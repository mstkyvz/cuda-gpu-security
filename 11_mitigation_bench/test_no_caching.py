"""
Test PYTORCH_NO_CUDA_MEMORY_CACHING=1 as a mitigation.
Must be run with the env var set:
  PYTORCH_NO_CUDA_MEMORY_CACHING=1 python3 test_no_caching.py
"""
import os
import torch
import time

mode = os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING", "0")
print(f"PYTORCH_NO_CUDA_MEMORY_CACHING = {mode}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print()

N = 512 * 1024  # 2 MB fp32

ROUNDS = 5
total_nz = 0

print(f"{'Round':<6}  {'Ptr-reuse':<10}  {'Non-zero':<14}  {'Match%'}")
print("-" * 45)

for r in range(ROUNDS):
    secret = 2.71828 * (r + 1)
    victim = torch.empty(N, dtype=torch.float32, device="cuda")
    victim.fill_(secret)
    v_ptr = victim.data_ptr()
    del victim
    torch.cuda.synchronize()

    leaked = torch.empty(N, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    a_ptr = leaked.data_ptr()

    leaked_cpu = leaked.cpu()
    nz = (leaked_cpu.abs() > 1e-6).sum().item()
    matches = (leaked_cpu - secret).abs() < 1e-4
    match_n = matches.sum().item()
    total_nz += nz

    print(f"{r+1:<6}  {'YES' if a_ptr == v_ptr else 'no':<10}  "
          f"{nz:<7}/{N:<6}  {100.0*match_n/N:.1f}%")
    del leaked

pct = 100.0 * total_nz / (ROUNDS * N)
print()
if pct < 1.0:
    print(f"[SAFE] Non-zero: {total_nz}/{ROUNDS*N} ({pct:.1f}%)")
    print("       PYTORCH_NO_CUDA_MEMORY_CACHING=1 prevents the leak")
    print("       (cudaMalloc is used directly, driver zeroes each allocation)")
else:
    print(f"[LEAK] Non-zero: {total_nz}/{ROUNDS*N} ({pct:.1f}%)")
    print("       Caching is still active — check env var")

# Benchmark: overhead vs caching
print()
print("--- Performance overhead of disabling cache ---")
ITERS = 200

t0 = time.perf_counter()
for _ in range(ITERS):
    t = torch.empty(N, dtype=torch.float32, device="cuda")
    del t
torch.cuda.synchronize()
t1 = time.perf_counter()
time_us = (t1 - t0) / ITERS * 1e6
print(f"  torch.empty time (no-cache mode): {time_us:.1f} μs / alloc")
print(f"  (cached torch.empty baseline was ~22 μs on this GPU)")
print(f"  Overhead: ~{max(0, time_us - 22):.1f} μs/alloc = "
      f"{max(0, (time_us/22 - 1)*100):.0f}% slower than cached empty")
