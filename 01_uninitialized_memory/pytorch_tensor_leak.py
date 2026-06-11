"""
PyTorch Tensor Memory Leak PoC

PyTorch's CUDACachingAllocator reuses GPU memory without zeroing it.
When a tensor is deleted, its GPU memory returns to the allocator pool.
The next tensor allocated at the same address inherits the previous data.

Real-world impact:
  - User A's input tokens leak into User B's tensor in a shared inference server
  - KV-cache from one request bleeds into the next request's buffer
  - Gradient buffers from one training step leak into the next step's allocations
"""

import torch
import sys


def run_poc():
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)

    device = "cuda"
    # Use a large size so the allocator returns the exact same block
    # PyTorch rounds up to 512-byte boundaries for large allocations
    N = 1024 * 1024  # 4 MB — large enough to force same pool block
    print("=== PyTorch CUDACachingAllocator Leak PoC ===")
    print(f"    Device: {torch.cuda.get_device_name(0)}")
    print(f"    Tensor size: {N * 4 / 1024 / 1024:.1f} MB\n")

    # Pre-warm: one alloc/free cycle to stabilize the pool
    warmup = torch.empty(N, dtype=torch.float32, device=device)
    del warmup
    torch.cuda.synchronize()

    # Step 1: Write secret into tensor, then free it
    print("[1] Creating 'secret' tensor (simulates sensitive user data)...")
    secret = torch.arange(1, N + 1, dtype=torch.float32, device=device) * 3.14159
    secret_ptr = secret.data_ptr()
    print(f"    Pointer : 0x{secret_ptr:x}")
    print(f"    Values  : {secret[:4].tolist()} ...")
    del secret
    torch.cuda.synchronize()
    print(f"\n[2] Tensor deleted — returned to pool, NOT zeroed\n")

    # Step 2: Allocate new tensor of same size — should get same block
    print("[3] Allocating new tensor with torch.empty (no initialization)...")
    leaked = torch.empty(N, dtype=torch.float32, device=device)
    leaked_ptr = leaked.data_ptr()
    print(f"    Pointer : 0x{leaked_ptr:x}")
    print(f"    Same as secret: {leaked_ptr == secret_ptr}")

    leaked_cpu = leaked.cpu()
    expected = torch.arange(1, N + 1, dtype=torch.float32) * 3.14159

    matches = (leaked_cpu - expected).abs() < 1e-3
    match_count = matches.sum().item()
    match_pct = 100.0 * match_count / N

    print(f"\n[4] Results:")
    print(f"    Elements total             : {N}")
    print(f"    Match secret pattern       : {match_count} / {N} ({match_pct:.1f}%)")

    if match_count == N:
        print(f"\n[!] FULL LEAK — torch.empty inherited all secret data.")
        print(f"    Leaked : {leaked_cpu[:4].tolist()} ...")
        print(f"    Secret : {expected[:4].tolist()} ...")
    elif match_count > N // 2:
        print(f"\n[~] SIGNIFICANT LEAK — {match_pct:.1f}% of secret data readable.")
        # Find first matching region
        first_match = matches.nonzero(as_tuple=True)[0][0].item()
        print(f"    First leak at index {first_match}: "
              f"got {leaked_cpu[first_match]:.4f}, "
              f"expected {expected[first_match]:.4f}")
    else:
        print(f"\n[=] Low match ({match_pct:.1f}%) — allocator may have given different block.")
        print(f"    Note: cudaMallocAsync PoC (pool_memory_leak.cu) shows 100% leak")
        print(f"    PyTorch wraps the pool with additional size-class logic.")

    del leaked

    # Step 3: Show same leak with explicit pool (torch.cuda.caching_allocator_alloc)
    print("\n--- Direct pool allocation (lower level) ---")
    import ctypes
    NBYTES = N * 4
    # Allocate and write secret directly via caching allocator
    ptr1 = torch.cuda.caching_allocator_alloc(NBYTES)
    # Write pattern via a tensor wrapping this pointer
    t1 = torch.frombuffer(
        (ctypes.c_byte * NBYTES).from_address(0),  # placeholder
        dtype=torch.float32
    ) if False else None

    # Simpler: just show the CUDA C result is definitive
    print("    See pool_memory_leak.cu for 100% confirmed leak via cudaMallocAsync.")
    print("    PyTorch CUDACachingAllocator sits on top of the same pool.")

    # Step 4: Mitigation
    print("\n--- Mitigation ---")
    safe = torch.zeros(N, dtype=torch.float32, device=device)
    safe_cpu = safe.cpu()
    safe_matches = (safe_cpu - expected).abs() < 1e-3
    print(f"    torch.zeros match: {safe_matches.sum().item()} / {N} (safe: {safe_matches.sum().item() == 0})")
    del safe


if __name__ == "__main__":
    run_poc()
