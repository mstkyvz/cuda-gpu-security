"""
Experiment 33: PyTorch CachingAllocator Residue

PyTorch manages its own GPU memory pool on top of CUDA's allocator.
When a tensor is deleted (del / goes out of scope), PyTorch does NOT
immediately return memory to CUDA -- it caches the block internally.
The next torch.empty() of the same size on the same stream gets the
cached block WITHOUT zeroing.

Tests:
  A. Basic residue:    fill SECRET → del → torch.empty same size → check
  B. empty_cache():   fill → del → torch.cuda.empty_cache() → alloc → check
                      (empty_cache returns to CUDA; new cudaMalloc is zeroed)
  C. Cross-stream:    fill on stream-A → del → alloc on default stream → check
  D. Grad buffer:     tensor.backward() allocates grad buffer → del → new alloc → check
  E. Real layer:      nn.Linear forward with SECRET input → del activations →
                      new alloc same size → check for activation residue

All tests run on GPU 1 (GPU 0 has Nemotron training).
Each test runs 5 times for consistency.
"""

import torch
import torch.nn as nn
import math

DEVICE   = torch.device("cuda:1")
SECRET   = 3.14159
TOL      = 0.01

def count_secret(t: torch.Tensor, val: float = SECRET, tol: float = TOL) -> int:
    return int(((t - val).abs() < tol).sum().item())

def pct(hits: int, total: int) -> str:
    return f"{hits}/{total} ({100*hits/total:.1f}%)"

def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ──────────────────────────────────────────────────────────────
# Test A: Basic pool residue
# fill → del → torch.empty (same size, same default stream)
# ──────────────────────────────────────────────────────────────
def test_a(N: int = 1024*1024, passes: int = 5):
    separator("Test A: Basic pool residue (del → torch.empty, no sync)")
    for p in range(1, passes+1):
        x = torch.full((N,), SECRET, dtype=torch.float32, device=DEVICE)
        torch.cuda.synchronize(DEVICE)   # ensure write is complete
        del x                            # back to PyTorch pool, NOT zeroed

        y = torch.empty(N, dtype=torch.float32, device=DEVICE)
        torch.cuda.synchronize(DEVICE)
        hits = count_secret(y)
        print(f"  Pass {p}: SECRET={pct(hits, N)}",
              "  [!!!] LEAK" if hits > N//4 else "  [safe]")
        del y

# ──────────────────────────────────────────────────────────────
# Test B: empty_cache() — does it sanitize?
# fill → del → empty_cache → torch.empty
# empty_cache returns blocks to CUDA; new cudaMalloc is driver-zeroed
# ──────────────────────────────────────────────────────────────
def test_b(N: int = 1024*1024, passes: int = 5):
    separator("Test B: empty_cache() before re-alloc (expect SAFE)")
    for p in range(1, passes+1):
        x = torch.full((N,), SECRET, dtype=torch.float32, device=DEVICE)
        torch.cuda.synchronize(DEVICE)
        del x
        torch.cuda.empty_cache()         # return to CUDA driver → driver zeros on next alloc

        y = torch.empty(N, dtype=torch.float32, device=DEVICE)
        torch.cuda.synchronize(DEVICE)
        hits = count_secret(y)
        print(f"  Pass {p}: SECRET={pct(hits, N)}",
              "  [!!!] LEAK" if hits > N//4 else "  [safe] — empty_cache zeroed")
        del y

# ──────────────────────────────────────────────────────────────
# Test C: Cross-stream residue
# fill on stream-A → del → alloc on default stream
# PyTorch's allocator is stream-aware: cross-stream alloc should be safe
# ──────────────────────────────────────────────────────────────
def test_c(N: int = 1024*1024, passes: int = 5):
    separator("Test C: Cross-stream residue (stream-A → default stream)")
    s = torch.cuda.Stream(device=DEVICE)
    for p in range(1, passes+1):
        with torch.cuda.stream(s):
            x = torch.full((N,), SECRET, dtype=torch.float32, device=DEVICE)
        s.synchronize()
        del x                            # freed on stream-A's pool

        # Alloc on DEFAULT stream — different stream pool
        y = torch.empty(N, dtype=torch.float32, device=DEVICE)
        torch.cuda.synchronize(DEVICE)
        hits = count_secret(y)
        print(f"  Pass {p}: SECRET={pct(hits, N)}",
              "  [!!!] LEAK" if hits > N//4 else "  [safe]")
        del y

# ──────────────────────────────────────────────────────────────
# Test D: Gradient buffer residue
# x.backward() causes PyTorch to allocate a gradient buffer.
# After del, does the next same-size alloc get the grad buffer?
# ──────────────────────────────────────────────────────────────
def test_d(N: int = 1024*1024, passes: int = 5):
    separator("Test D: Gradient buffer residue (backward → del grad → re-alloc)")
    for p in range(1, passes+1):
        # Create a simple computation graph
        x = torch.full((N,), SECRET, dtype=torch.float32,
                       device=DEVICE, requires_grad=True)
        loss = x.sum()
        loss.backward()                  # allocates x.grad buffer with SECRET
        torch.cuda.synchronize(DEVICE)

        grad_val = x.grad.mean().item()
        # x.grad should be all 1.0 (d(sum)/dx = 1), but grad buffer was
        # allocated in PyTorch's pool and may have been reused
        x.grad = None
        del x, loss

        # Alloc same size — does it get the gradient buffer?
        y = torch.empty(N, dtype=torch.float32, device=DEVICE)
        torch.cuda.synchronize(DEVICE)
        hits = count_secret(y)
        print(f"  Pass {p}: grad_mean={grad_val:.2f}  "
              f"re-alloc SECRET={pct(hits, N)}",
              "  [!!!] LEAK" if hits > N//4 else "  [safe]")
        del y

# ──────────────────────────────────────────────────────────────
# Test E: Real layer activation residue
# nn.Linear forward pass with SECRET input.
# Intermediate activation tensors allocated internally by PyTorch.
# After del, new alloc same size — activation residue?
# ──────────────────────────────────────────────────────────────
def test_e(batch: int = 512, dim: int = 2048, passes: int = 5):
    separator("Test E: nn.Linear activation residue (forward → del → re-alloc)")
    N = batch * dim
    layer = nn.Linear(dim, dim, bias=False).to(DEVICE)

    for p in range(1, passes+1):
        # Forward pass with SECRET-valued input
        inp = torch.full((batch, dim), SECRET, dtype=torch.float32, device=DEVICE)
        out = layer(inp)                 # allocates output buffer from pool
        torch.cuda.synchronize(DEVICE)

        out_mean = out.mean().item()
        del out, inp                     # output (activation) back to pool

        # Re-alloc same shape — does it get the activation buffer?
        y = torch.empty(batch, dim, dtype=torch.float32, device=DEVICE)
        torch.cuda.synchronize(DEVICE)
        hits = count_secret(y)
        print(f"  Pass {p}: out_mean={out_mean:.4f}  "
              f"re-alloc SECRET={pct(hits, N)}",
              "  [!!!] LEAK" if hits > N//4 else "  [safe]")
        del y

    del layer

# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.cuda.set_device(DEVICE)
    print("=== Experiment 33: PyTorch CachingAllocator Residue ===")
    print(f"    PyTorch {torch.__version__}  CUDA {torch.version.cuda}")
    p = torch.cuda.get_device_properties(DEVICE)
    print(f"    Device: {p.name}  Memory: {p.total_memory // (1024**3)} GB")
    print(f"    SECRET={SECRET}  N=1M floats (4MB) per test\n")

    test_a()
    test_b()
    test_c()
    test_d()
    test_e()

    print("\n=== Summary ===")
    print("A: Basic del → empty (same stream)     — pool residue?")
    print("B: del → empty_cache → empty           — empty_cache sanitizes?")
    print("C: del on stream-A → empty (default)   — cross-stream isolated?")
    print("D: backward grad buffer → del → empty  — grad buf residue?")
    print("E: Linear forward output → del → empty — activation residue?")
    print("\n[Done]")
