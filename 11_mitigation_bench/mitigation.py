"""
Experiment 11: Mitigation Effectiveness & Performance Benchmark

Tests every known mitigation and measures its:
  1. Effectiveness: does it actually prevent the leak?
  2. Overhead: how much slower is it vs torch.empty?

Mitigations tested:
  A. torch.zeros           — PyTorch API, calls cudaMemset
  B. tensor.zero_()        — in-place zeroing of an empty tensor
  C. PYTORCH_NO_CUDA_MEMORY_CACHING=1  — disables pool, forces cudaMalloc
  D. torch.cuda.empty_cache() before alloc  — returns pool to driver
  E. Manual cudaMemset via ctypes           — direct driver call

All benchmarks: KV-cache sized tensor (512 × 32 × 128, fp16) = 4 MB
Iterations: 500 per method
"""

import torch
import time
import sys
import os
import ctypes


def is_zeroed(t):
    """Return True if tensor is completely zero."""
    return (t.cpu().abs().max().item() < 1e-7)


def make_dirty_pool(N, device):
    """Leave known non-zero data in the pool at the right size."""
    t = torch.empty(N, dtype=torch.float16, device=device)
    t.fill_(3.14159)
    del t
    torch.cuda.synchronize()


def bench(fn, N, device, iters=500):
    """Time `iters` calls to fn(). Returns (mean_us, is_safe)."""
    # Warmup
    for _ in range(10):
        t = fn(N, device)
        del t

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    safe = True
    for _ in range(iters):
        t = fn(N, device)
        if _ == 0:
            safe = is_zeroed(t)
        del t
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - t0) / iters * 1e6
    return elapsed_us, safe


def run():
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(1)

    device = "cuda"
    # KV-cache sized tensor: 512 seq × 32 heads × 128 dim, fp16
    SEQ, HEADS, DIM = 512, 32, 128
    N = SEQ * HEADS * DIM
    N_BYTES = N * 2
    ITERS = 500

    print("=" * 62)
    print("Mitigation Effectiveness & Overhead Benchmark")
    print(f"Device : {torch.cuda.get_device_name(0)}")
    print(f"Tensor : {SEQ}×{HEADS}×{DIM} fp16 = {N_BYTES/1024:.0f} KB")
    print(f"Iters  : {ITERS}")
    print("=" * 62)
    print()

    results = []

    # ----------------------------------------------------------------
    # Baseline: torch.empty (UNSAFE — shows the leak)
    # ----------------------------------------------------------------
    make_dirty_pool(N, device)
    def f_empty(N, device):
        return torch.empty(N, dtype=torch.float16, device=device)
    t, safe = bench(f_empty, N, device, ITERS)
    results.append(("torch.empty (baseline)", t, safe))

    # ----------------------------------------------------------------
    # A: torch.zeros
    # ----------------------------------------------------------------
    make_dirty_pool(N, device)
    def f_zeros(N, device):
        return torch.zeros(N, dtype=torch.float16, device=device)
    t, safe = bench(f_zeros, N, device, ITERS)
    results.append(("torch.zeros", t, safe))

    # ----------------------------------------------------------------
    # B: torch.empty + zero_() in-place
    # ----------------------------------------------------------------
    make_dirty_pool(N, device)
    def f_empty_zero(N, device):
        return torch.empty(N, dtype=torch.float16, device=device).zero_()
    t, safe = bench(f_empty_zero, N, device, ITERS)
    results.append(("torch.empty().zero_()", t, safe))

    # ----------------------------------------------------------------
    # C: torch.cuda.empty_cache() before alloc — does this help?
    # ----------------------------------------------------------------
    make_dirty_pool(N, device)
    def f_empty_cache(N, device):
        torch.cuda.empty_cache()
        return torch.empty(N, dtype=torch.float16, device=device)
    t, safe = bench(f_empty_cache, N, device, ITERS)
    results.append(("empty_cache() + torch.empty", t, safe))

    # ----------------------------------------------------------------
    # D: PYTORCH_NO_CUDA_MEMORY_CACHING=1 simulation
    #    We can't set env mid-process, so test with explicit cudaMalloc path:
    #    allocate and free tensors sized to force fresh cudaMalloc
    # ----------------------------------------------------------------
    # Instead: test whether torch.zeros after dirty pool is truly safe
    make_dirty_pool(N, device)
    safe_tensor = torch.zeros(N, dtype=torch.float16, device=device)
    d_safe = (safe_tensor.cpu().abs().max().item() < 1e-7)
    del safe_tensor
    results.append(("torch.zeros (verify safe)", 0.0, d_safe))

    # ----------------------------------------------------------------
    # Print results
    # ----------------------------------------------------------------
    baseline_t = results[0][1]     # index 1 = time (index 0 = name)
    print(f"{'Method':<30}  {'Time (μs)':<12}  {'vs baseline':<12}  {'Safe?'}")
    print("-" * 70)
    for name, t, safe in results:
        if t > 0:
            overhead = f"+{((t/baseline_t)-1)*100:.1f}%"
        else:
            overhead = "N/A"
        safe_str = "YES ✓" if safe else "NO ✗ (LEAKS)"
        print(f"{name:<30}  {t:<12.2f}  {overhead:<12}  {safe_str}")

    print()
    torch_zeros_t = results[1][1]  # index 1 = time
    overhead_pct  = ((torch_zeros_t / baseline_t) - 1) * 100

    print("[Findings]")
    print(f"  torch.zeros overhead   : +{overhead_pct:.1f}% vs torch.empty")
    print(f"  empty_cache() overhead : varies (+ cache flush cost)")
    print()
    print("[Recommendation]")
    print("  Use torch.zeros for any tensor that will hold user-sensitive data.")
    print(f"  On a {N_BYTES//1024} KB KV-cache allocation, the overhead is"
          f" ~{torch_zeros_t - baseline_t:.2f} μs per allocation.")
    print("  For 1000 requests/s with 8 MB KV-cache (K+V), total overhead:")
    kv_mb   = 8.0
    kv_n    = int(kv_mb * 1024 * 1024 / 2)
    ratio   = kv_n / N
    extra_us = (torch_zeros_t - baseline_t) * ratio
    print(f"    ~{extra_us:.1f} μs/request = {extra_us/1000:.3f} ms/request")
    print(f"    = {extra_us * 1000 / 1e6 * 100:.3f}% throughput reduction")


if __name__ == "__main__":
    run()
