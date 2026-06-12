"""
Experiment 37: F.scaled_dot_product_attention (SDPA / FlashAttention) Buffer Residue

PyTorch's F.scaled_dot_product_attention uses FlashAttention kernel on H100.
This experiment tests whether the output tensor and internal scratch buffers
are in PyTorch's CachingAllocator pool (no-zeroing reuse).

Tests A-E use simpler, more isolated setups to avoid pool competition
from transpose/contiguous intermediate tensors.

All runs: GPU 1, H100, PyTorch 2.12 (FlashAttention kernel via SDPA)
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import gc

DEVICE = torch.device("cuda:1")
SECRET = 3.14159

def separator(t):
    print(f"\n{'='*60}\n  {t}\n{'='*60}")

# ═══════════════════════════════════════════════════════════════
# Test A: SDPA output in pool — CLEAN (no transpose intermediates)
# Use q,k,v already in [B, H, S, D] format → no extra allocs
# ═══════════════════════════════════════════════════════════════
def test_a(passes=5):
    separator("Test A: SDPA output (O) residue — clean measurement")
    B, H, S, D = 2, 8, 64, 64  # already [B,H,S,D] for SDPA

    for p in range(1, passes+1):
        torch.cuda.empty_cache()

        q = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float16)

        with torch.no_grad():
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out_mean = out.float().mean().item()
        out_std  = out.float().std().item()
        out_shape = out.shape   # [B, H, S, D]
        del out
        torch.cuda.synchronize(DEVICE)

        # Q, K, V still alive → only out's block in pool for this shape
        residue = torch.empty(out_shape, dtype=torch.float16, device=DEVICE)
        torch.cuda.synchronize(DEVICE)

        r_mean = residue.float().mean().item()
        r_std  = residue.float().std().item()
        match  = (abs(r_mean - out_mean) < 0.05 * abs(out_mean) + 1e-4 and
                  abs(r_std  - out_std)  < 0.05 * abs(out_std)  + 1e-4)

        print(f"  Pass {p}: out(mean={out_mean:.5f} std={out_std:.4f})  "
              f"residue(mean={r_mean:.5f} std={r_std:.4f})  "
              f"match={'YES' if match else 'NO'}",
              "  [!!!] SDPA OUTPUT IN POOL" if match else "  [~]")

        del q, k, v, residue
        gc.collect()

# ═══════════════════════════════════════════════════════════════
# Test B: SDPA computation SAFE — pool residue does NOT affect output
# Verify: pollute pool with SECRET-sized block, run SDPA → same result
# ═══════════════════════════════════════════════════════════════
def test_b(passes=5):
    separator("Test B: SDPA computation stable despite pool residue")
    B, H, S, D = 2, 8, 64, 64
    q = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float16)
    k = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float16)
    v = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float16)

    for p in range(1, passes+1):
        torch.cuda.empty_cache()
        with torch.no_grad():
            out_clean = F.scaled_dot_product_attention(q, k, v).clone()

        # Pollute pool with SECRET block of exact output shape
        junk = torch.full((B, H, S, D), SECRET, device=DEVICE, dtype=torch.float16)
        del junk
        torch.cuda.synchronize(DEVICE)

        with torch.no_grad():
            out_polluted = F.scaled_dot_product_attention(q, k, v)

        diff = (out_clean - out_polluted.to(out_clean.dtype)).abs().max().item()
        print(f"  Pass {p}: max_diff={diff:.2e}",
              "  [!!!] COMPUTATION AFFECTED" if diff > 1e-3
              else "  [OK] SDPA writes output before returning — no contamination")
        del out_clean, out_polluted
        gc.collect()

    del q, k, v

# ═══════════════════════════════════════════════════════════════
# Test C: Multi-head attention — per-head output residue
# nn.MultiheadAttention uses SDPA internally; output is [S, B, E]
# After del → re-alloc same shape
# ═══════════════════════════════════════════════════════════════
def test_c(passes=5):
    separator("Test C: nn.MultiheadAttention output residue (same pool)")
    E, H, S, B = 512, 8, 64, 4
    mha = nn.MultiheadAttention(E, H, batch_first=False).to(DEVICE).eval()

    q = torch.randn(S, B, E, device=DEVICE)
    k = torch.randn(S, B, E, device=DEVICE)
    v = torch.randn(S, B, E, device=DEVICE)

    # Warm up
    with torch.no_grad():
        _ = mha(q, k, v)
    torch.cuda.synchronize(DEVICE)

    for p in range(1, passes+1):
        torch.cuda.empty_cache()
        with torch.no_grad():
            attn_out, _ = mha(q, k, v)  # [S, B, E]
        out_mean  = attn_out.mean().item()
        out_std   = attn_out.std().item()
        out_shape = attn_out.shape
        del attn_out
        torch.cuda.synchronize(DEVICE)

        residue  = torch.empty(out_shape, device=DEVICE)
        torch.cuda.synchronize(DEVICE)
        r_mean   = residue.mean().item()
        r_std    = residue.std().item()
        match    = (abs(r_mean - out_mean) < 0.05 * abs(out_mean) + 1e-4 and
                    abs(r_std  - out_std)  < 0.05 * abs(out_std)  + 1e-4)

        print(f"  Pass {p}: mha_out(mean={out_mean:.5f} std={out_std:.4f})  "
              f"residue(mean={r_mean:.5f} std={r_std:.4f})  "
              f"match={'YES' if match else 'NO'}",
              "  [!!!] MHA OUTPUT IN POOL" if match else "  [~]")
        del residue
        gc.collect()

    del mha, q, k, v

# ═══════════════════════════════════════════════════════════════
# Test D: SECRET Q·K product → attention output residue
# When Q=K=V=SECRET, all attention weights are uniform (softmax of equal scores)
# Output = softmax * V = V = SECRET (approximately)
# After del → re-alloc → does residue ≈ SECRET?
# ═══════════════════════════════════════════════════════════════
def test_d(passes=5):
    separator("Test D: SECRET V → FA output = SECRET → residue has SECRET value?")
    B, H, S, D = 1, 4, 16, 32

    for p in range(1, passes+1):
        torch.cuda.empty_cache()

        # Q, K arbitrary; V = SECRET (all uniform → output ≈ SECRET)
        q = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float32)
        k = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float32)
        v = torch.full((B, H, S, D), SECRET, device=DEVICE, dtype=torch.float32)

        with torch.no_grad():
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        out_mean  = out.mean().item()  # should be ≈ SECRET
        out_shape = out.shape
        del out
        torch.cuda.synchronize(DEVICE)

        residue = torch.empty(out_shape, device=DEVICE, dtype=torch.float32)
        torch.cuda.synchronize(DEVICE)
        r_mean = residue.mean().item()
        err    = abs(r_mean - out_mean)

        print(f"  Pass {p}: expected_mean≈{out_mean:.4f}≈SECRET  "
              f"residue_mean={r_mean:.4f}  err={err:.4f}",
              f"  [!!!] SECRET VALUE IN RESIDUE" if err < 0.05 * abs(out_mean) + 0.01
              else "  [~]")

        del q, k, v, residue
        gc.collect()

# ═══════════════════════════════════════════════════════════════
# Test E: Sequential attention requests — embedding analogy
# Request A: SDPA on SECRET-filled V (output ≈ SECRET)
# del → Request B: torch.empty → gets A's attention output?
# Then try to detect SECRET value in pool block
# ═══════════════════════════════════════════════════════════════
def test_e(passes=5):
    separator("Test E: Sequential requests — B's empty gets A's attention output")
    B, H, S, D = 2, 8, 48, 64
    N = B * H * S * D

    for p in range(1, passes+1):
        torch.cuda.empty_cache()

        # Request A: uniform V=SECRET → out ≈ SECRET everywhere
        q_a = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float32)
        k_a = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float32)
        v_a = torch.full((B, H, S, D), SECRET, device=DEVICE, dtype=torch.float32)

        with torch.no_grad():
            out_a = F.scaled_dot_product_attention(q_a, k_a, v_a)
        true_mean = out_a.mean().item()
        out_shape = out_a.shape
        del out_a, q_a, k_a, v_a
        torch.cuda.synchronize(DEVICE)

        # B: torch.empty → gets A's output buffer
        residue = torch.empty(out_shape, device=DEVICE, dtype=torch.float32)
        torch.cuda.synchronize(DEVICE)

        r_mean = residue.mean().item()
        hits   = int(((residue - SECRET).abs() < 0.05).sum().item())
        err    = abs(r_mean - true_mean)

        print(f"  Pass {p}: A_out_mean≈{true_mean:.4f}  "
              f"residue_mean={r_mean:.4f}  "
              f"err={err:.5f}  near_secret={hits}/{N}",
              "  [!!!] A's ATTENTION OUTPUT IN B's POOL BLOCK" if err < 0.01 * abs(true_mean) + 0.01
              else "  [~]")

        del residue
        gc.collect()


if __name__ == "__main__":
    print("=== Experiment 37: SDPA / FlashAttention Buffer Residue ===")
    print(f"    PyTorch {torch.__version__}  GPU: cuda:1")
    print(f"    SDPA backends: flash={torch.backends.cuda.flash_sdp_enabled()}")
    print()

    test_a()
    test_b()
    test_c()
    test_d()
    test_e()

    print("\n=== Summary ===")
    print("A: SDPA output [B,H,S,D] → del → torch.empty → pool residue?")
    print("B: SDPA computation stable with pool residue? (writes before read)")
    print("C: nn.MultiheadAttention output → del → pool residue?")
    print("D: V=SECRET → attention output ≈ SECRET → residue has SECRET?")
    print("E: Sequential: A's attention output → B's torch.empty")
    print("\n[Done]")
