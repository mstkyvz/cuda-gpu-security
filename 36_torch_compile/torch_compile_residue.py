"""
Experiment 36: torch.compile() / TorchInductor Intermediate Buffer Residue

torch.compile() (introduced PyTorch 2.0) transforms Python functions into
optimized CUDA kernels via TorchInductor. The generated code:
  - Fuses operations (e.g., linear + gelu + linear into one kernel)
  - Allocates its own intermediate "buf" tensors for fused ops
  - Manages these buffers via PyTorch's CachingAllocator

Key question: does TorchInductor's code generation read intermediate buffers
before writing them? If so, those buffers carry pool residue into computation.
Also: do TorchInductor's allocations themselves follow the same no-zeroing
pool pattern as eager mode?

Tests:
  A. Compiled vs eager residue path:
     Fill SECRET, del, run compiled forward → does compiled output depend
     on pool residue? (if compiled kernel reads buf before write: yes)

  B. Intermediate fusion buffer:
     Fused (linear + activation + linear) creates intermediate buf.
     del buf → re-alloc → does it contain fused computation residue?

  C. torch.compile buffer vs torch.empty — same pool?
     Confirm that torch.compile uses the same CachingAllocator pool
     (not a separate allocator), making Exp 33/35 attacks apply to
     compiled models too.

  D. Persistent compiled graph buffer:
     CUDA graph capture with torch.compile creates STATIC buffers.
     These are never freed during repeated calls. Do they retain
     old values between different input sequences?

  E. compile() with different backends — inductor vs eager:
     Compare residue behavior: backend='eager', 'aot_eager', 'inductor'
     Does the backend affect whether intermediate buffers are zeroed?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

DEVICE  = torch.device("cuda:1")
SECRET  = 3.14159
TOL     = 0.01

def separator(t):
    print(f"\n{'='*60}\n  {t}\n{'='*60}")

def count_secret(x, val=SECRET, tol=TOL):
    return int(((x - val).abs() < tol).sum().item())

# ─────────────────────────────────────────────────────────────
# Model: simple MLP with fuse-able ops
# Linear → GELU → Linear (TorchInductor fuses these)
# ─────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, d=1024):
        super().__init__()
        self.fc1 = nn.Linear(d, d * 4, bias=False)
        self.fc2 = nn.Linear(d * 4, d, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

# ═══════════════════════════════════════════════════════════════
# Test A: Does compiled model output change when pool has SECRET?
# If compiled kernel reads intermediate buf before writing:
# output with SECRET-polluted pool ≠ output with clean pool
# ═══════════════════════════════════════════════════════════════
def test_a(passes=5):
    separator("Test A: Compiled output stability — pool residue affects computation?")
    d   = 1024
    B   = 32
    mdl = MLP(d).to(DEVICE).eval()
    mdl_c = torch.compile(mdl, backend="inductor", fullgraph=True)

    # Warm up compiled model
    dummy = torch.randn(B, d, device=DEVICE)
    with torch.no_grad():
        _ = mdl_c(dummy)
    torch.cuda.synchronize(DEVICE)

    inp = torch.randn(B, d, device=DEVICE)

    for p in range(1, passes+1):
        # Clean run: empty_cache first, then run compiled model
        torch.cuda.empty_cache()
        with torch.no_grad():
            out_clean = mdl_c(inp).clone()

        # Pollute pool with SECRET
        junk = torch.full((B, d * 4), SECRET, device=DEVICE)  # same as fc1 output
        del junk
        torch.cuda.synchronize(DEVICE)

        # Run again — if compiled kernel reads buf before write, output differs
        with torch.no_grad():
            out_polluted = mdl_c(inp)

        diff = (out_clean - out_polluted).abs().max().item()
        print(f"  Pass {p}: max_diff_clean_vs_polluted={diff:.2e}",
              "  [!!!] COMPUTATION AFFECTED BY RESIDUE" if diff > 1e-4
              else "  [OK] output identical — compiled kernel writes before read")
        del out_clean, out_polluted
        gc.collect()

    del mdl, mdl_c, inp

# ═══════════════════════════════════════════════════════════════
# Test B: TorchInductor intermediate buffer — is it in the same pool?
# After compiled forward, allocate same-size tensors and check for
# computation residue (intermediate GELU output values)
# ═══════════════════════════════════════════════════════════════
def test_b(passes=5):
    separator("Test B: TorchInductor fusion buffer → pool residue scan")
    d   = 1024
    B   = 32
    mdl = MLP(d).to(DEVICE).eval()
    mdl_c = torch.compile(mdl, backend="inductor", fullgraph=True)

    inp = torch.randn(B, d, device=DEVICE)

    # Warm up
    with torch.no_grad():
        _ = mdl_c(inp)
    torch.cuda.synchronize(DEVICE)

    # Get reference: what does the eager fc1 output look like?
    with torch.no_grad():
        fc1_out_ref = F.gelu(mdl.fc1(inp)).clone()  # [B, d*4]
    ref_mean = fc1_out_ref.mean().item()
    ref_std  = fc1_out_ref.std().item()

    for p in range(1, passes+1):
        torch.cuda.empty_cache()  # clean pool

        # Run compiled forward (inductor allocates intermediate buf [B, d*4])
        with torch.no_grad():
            out = mdl_c(inp)
            out_shape = out.shape
        del out
        torch.cuda.synchronize(DEVICE)

        # Drain: allocate tensors of the intermediate buffer size [B, d*4]
        candidates = [torch.empty(B, d * 4, device=DEVICE) for _ in range(8)]
        torch.cuda.synchronize(DEVICE)

        # Check each: does any look like gelu(fc1(inp))?
        best_sim = 0.0
        best_idx = -1
        for i, c in enumerate(candidates):
            c_norm   = F.normalize(c.view(-1), dim=0)
            ref_norm = F.normalize(fc1_out_ref.view(-1), dim=0)
            sim = (c_norm * ref_norm).sum().item()
            if abs(sim) > abs(best_sim):
                best_sim = sim
                best_idx = i

        print(f"  Pass {p}: ref_mean={ref_mean:.3f} ref_std={ref_std:.3f}  "
              f"best_block={best_idx} cosine_sim={best_sim:.4f}",
              "  [!!!] INDUCTOR BUF IN POOL" if abs(best_sim) > 0.5
              else "  [~] no clear inductor residue")
        for c in candidates:
            del c
        gc.collect()

    del mdl, mdl_c, inp, fc1_out_ref

# ═══════════════════════════════════════════════════════════════
# Test C: torch.compile uses same CachingAllocator pool
# Prove that compiled model output goes into the SAME pool as
# torch.empty — making Exp 33/35 attacks applicable to
# compiled models too
# ═══════════════════════════════════════════════════════════════
def test_c(passes=5):
    separator("Test C: torch.compile output in same pool as torch.empty")
    d = 512
    B = 16
    layer = nn.Linear(d, d, bias=False).to(DEVICE).eval()
    layer_c = torch.compile(layer, backend="inductor")

    inp = torch.randn(B, d, device=DEVICE)

    # Warm up
    with torch.no_grad():
        _ = layer_c(inp)
    torch.cuda.synchronize(DEVICE)

    for p in range(1, passes+1):
        torch.cuda.empty_cache()

        # Run compiled layer → output [B, d]
        with torch.no_grad():
            out_c = layer_c(inp)
        out_mean = out_c.mean().item()
        out_std  = out_c.std().item()
        out_shape = out_c.shape
        del out_c
        torch.cuda.synchronize(DEVICE)

        # torch.empty same shape — gets compiled output from pool?
        residue = torch.empty(out_shape, device=DEVICE)
        torch.cuda.synchronize(DEVICE)

        r_mean = residue.mean().item()
        r_std  = residue.std().item()
        # If same pool: r_mean ≈ out_mean, r_std ≈ out_std
        mean_match = abs(r_mean - out_mean) < 0.05 * abs(out_mean) + 1e-4
        std_match  = abs(r_std  - out_std)  < 0.05 * abs(out_std)  + 1e-4

        print(f"  Pass {p}: compiled(mean={out_mean:.4f} std={out_std:.4f})  "
              f"residue(mean={r_mean:.4f} std={r_std:.4f})  "
              f"match={'YES' if mean_match and std_match else 'NO'}",
              "  [!!!] COMPILED OUTPUT IN SHARED POOL" if mean_match and std_match
              else "  [~]")
        del residue
        gc.collect()

    del layer, layer_c, inp

# ═══════════════════════════════════════════════════════════════
# Test D: backend comparison — inductor vs aot_eager vs eager
# Do different compile backends affect residue characteristics?
# ═══════════════════════════════════════════════════════════════
def test_d(passes=3):
    separator("Test D: Backend comparison — residue across compile backends")
    d  = 512
    B  = 16
    N  = B * d

    inp = torch.randn(B, d, device=DEVICE)
    backends = ["eager", "aot_eager", "inductor"]

    for backend in backends:
        layer   = nn.Linear(d, d, bias=False).to(DEVICE).eval()
        layer_c = torch.compile(layer, backend=backend)

        # Warm up
        with torch.no_grad():
            _ = layer_c(inp)
        torch.cuda.synchronize(DEVICE)

        total_match = 0
        for p in range(1, passes+1):
            torch.cuda.empty_cache()
            with torch.no_grad():
                out = layer_c(inp)
            true_std = out.std().item()
            del out
            torch.cuda.synchronize(DEVICE)

            residue  = torch.empty(B, d, device=DEVICE)
            torch.cuda.synchronize(DEVICE)
            r_std = residue.std().item()
            match = abs(r_std - true_std) < 0.01 * true_std + 1e-5
            if match:
                total_match += 1
            del residue

        print(f"  backend={backend:10s}: pool reuse {total_match}/{passes} passes",
              "  [!!!] RESIDUE" if total_match == passes else "  [~]")
        del layer, layer_c
        gc.collect()

    del inp

# ═══════════════════════════════════════════════════════════════
# Test E: NEW FINDING — torch.compile static buffer across calls
# When compile() with mode='reduce-overhead' uses cudagraph internally,
# it creates STATIC input/output buffers. These are reused across calls
# without zeroing. Can we read a previous call's output from the static buf?
# ═══════════════════════════════════════════════════════════════
def test_e(passes=5):
    separator("Test E: torch.compile reduce-overhead — static buffer across calls")
    d  = 512
    B  = 4    # small batch for cudagraph compatibility
    N  = B * d

    layer = nn.Linear(d, d, bias=False).to(DEVICE).eval()
    # reduce-overhead mode uses CUDA graphs internally → static buffers
    layer_c = torch.compile(layer, mode="reduce-overhead")

    # Warm up with different inputs to populate static buffer
    dummy = torch.zeros(B, d, device=DEVICE)
    with torch.no_grad():
        _ = layer_c(dummy)
    torch.cuda.synchronize(DEVICE)

    for p in range(1, passes+1):
        # Call 1: run with SECRET-filled input
        inp_secret = torch.full((B, d), SECRET, device=DEVICE)
        with torch.no_grad():
            out1 = layer_c(inp_secret)
        out1_mean = out1.mean().item()

        # Call 2: run with ZERO input
        inp_zero = torch.zeros(B, d, device=DEVICE)
        with torch.no_grad():
            out2 = layer_c(inp_zero)
        out2_mean = out2.mean().item()

        # Expected: out1_mean ≈ SECRET * weight_sum ≠ 0, out2_mean ≈ 0
        # If static buffer leaks: out2 might contain remnants of out1
        leakage = abs(out2_mean) / (abs(out1_mean) + 1e-8)
        print(f"  Pass {p}: secret_out_mean={out1_mean:.4f}  "
              f"zero_out_mean={out2_mean:.6f}  "
              f"leakage_ratio={leakage:.4f}",
              "  [!!!] STATIC BUF LEAKS" if leakage > 0.01
              else "  [OK] zero input → zero output (static buf overwritten)")

        del out1, out2, inp_secret, inp_zero
        gc.collect()

    del layer, layer_c, dummy

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== Experiment 36: torch.compile() / TorchInductor Buffer Residue ===")
    print(f"    PyTorch {torch.__version__}  GPU: cuda:1")
    print(f"    torch.compile available: {hasattr(torch, 'compile')}\n")

    test_a()
    test_b()
    test_c()
    test_d()
    test_e()

    print("\n=== Summary ===")
    print("A: Pool residue affects compiled computation?")
    print("B: TorchInductor fusion buf in shared pool?")
    print("C: Compiled output recoverable via torch.empty?")
    print("D: Backend comparison (eager / aot_eager / inductor)")
    print("E: reduce-overhead static buffer cross-call leakage?")
    print("\n[Done]")
