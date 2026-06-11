# GPU Memory Confidentiality in ML Inference Servers

## Overview

Modern LLM inference servers process thousands of requests per second, allocating and freeing GPU tensors continuously. This research investigates a practical question: **can sensitive data from one request leak into the next?**

We ran four experiments on an NVIDIA RTX 4090 (Driver 565.77, CUDA 12.7) to measure the real-world memory confidentiality guarantees of CUDA's allocation APIs and PyTorch's tensor allocator.

**Short answer:** `cudaMalloc` is safe. The pool allocator — which PyTorch uses by default — is not.

---

## Background: Two Ways to Allocate GPU Memory

### cudaMalloc — the standard allocator

```c
cudaMalloc(&ptr, size);
```

Every call goes to the CUDA driver, which allocates fresh memory from the device heap. Since driver version ~418.x (2019), NVIDIA zero-initializes this memory before returning it. This means even if the physical DRAM previously held sensitive data, the caller always receives clean memory.

### cudaMallocAsync — the pool allocator

```c
cudaMallocAsync(&ptr, size, stream);
cudaFreeAsync(ptr, stream);
```

Introduced in CUDA 11.2, this API manages a per-stream memory pool. The first call allocates a large chunk from the driver; subsequent calls subdivide and reuse that chunk. Critically, **the pool never zeroes memory on reuse**. This is intentional — zeroing costs bandwidth, and the expectation is that callers will initialize before use.

### PyTorch's CUDACachingAllocator

PyTorch does not use `cudaMalloc` for most tensor allocations. Instead it maintains its own pool on top of the CUDA pool allocator. When you call `del tensor` or a tensor goes out of scope, PyTorch returns the GPU memory to its pool — not to the driver. The memory is not zeroed.

This means `torch.empty(...)` is genuinely empty in the sense that PyTorch does not write to it, but the underlying GPU bytes contain whatever the previous occupant left there.

---

## Experiments

### Experiment 1A: cudaMalloc (standard allocator)

**Setup:**
1. Allocate 4 KB with `cudaMalloc`
2. Write a recognizable 32-bit pattern: `word[i] = 0xDEAD0000 + i`
3. Free with `cudaFree`
4. Allocate a new 4 KB buffer with `cudaMalloc`
5. Read the new buffer without writing to it
6. Count how many words match the secret pattern

**Why the pattern matters:** We use `0xDEAD0000 + i` which is never zero. Earlier testing with a byte pattern that wraps to zero produced false positives — zeroed memory accidentally matched at positions where the expected value happened to be zero. This is a common pitfall in memory leak detection.

**Result:**
```
Old ptr: 0x7ff438c00000
New ptr: 0x7ff438c00000   ← same address returned

Non-zero words:          0 / 1024
Matching secret pattern: 0 / 1024 (0.00%)

[SAFE] Driver zeroed memory on cudaMalloc.
```

**Conclusion:** The NVIDIA driver zero-initializes memory on every `cudaMalloc` call. Standard allocations are safe.

---

### Experiment 1B: cudaMallocAsync (pool allocator)

**Setup:** Same as 1A but using `cudaMallocAsync` / `cudaFreeAsync`.

**Result:**
```
Old ptr: 0x302000000
New ptr: 0x302000000   ← exact same address

Floats matching secret pattern: 4096 / 4096 (100.0%)
Max error vs expected: 0.000000

[!] FULL LEAK confirmed via memory pool.
First 8 values: 3.1416 6.2832 9.4248 12.5664 15.7080 18.8495 21.9911 25.1327
Expected:       3.1416 6.2832 9.4248 12.5664 15.7080 18.8495 21.9911 25.1327
```

**Conclusion:** The pool allocator returns the exact same memory address with all previous data intact. 4096 out of 4096 floats match perfectly. This is expected behavior, documented by NVIDIA — but it means callers are responsible for initialization if they care about confidentiality.

---

### Experiment 2: Cross-Process Isolation

**Setup:**
- **writer** process: allocates 64 KB, writes `0xDEAD0000 + i`, exits **without calling `cudaFree`**. The CUDA context is destroyed on process exit.
- **reader** process: runs immediately after, allocates 64 KB, reads without initializing.

**Result:**
```
[writer] Wrote pattern to GPU ptr: 0x7f9c2ac00000
[writer] Exiting WITHOUT cudaFree

[reader] Got GPU ptr: 0x7f3112c00000

Non-zero words:       0 / 16384 (0.00%)
Match writer pattern: 0 / 16384 (0.0000%)

[SAFE] Driver zeroed GPU memory between processes.
```

**Conclusion:** The NVIDIA driver zeroes GPU memory when a CUDA context is destroyed. A new process cannot read another process's data on driver 565.77. Note: older drivers or specific hypervisor configurations could behave differently — the cross-process guarantee is driver and environment dependent.

---

### Experiment 3: ML Inference KV-Cache Leak

**Setup:**

Simulates two sequential inference requests on a shared server.

- KV-cache dimensions: 512 sequence length × 32 heads × 128 head dim, float16
- Total per-request KV-cache: ~8 MB

```python
# Request A: confidential prompt → KV-cache populated
k_cache = torch.zeros(512, 32, 128, dtype=torch.float16, device="cuda")
v_cache = torch.zeros(512, 32, 128, dtype=torch.float16, device="cuda")
# ... fill with attention keys/values derived from prompt tokens ...

# Request A done: tensors freed back to PyTorch pool (NOT zeroed)
del k_cache, v_cache

# Request B: new allocation with torch.empty (no initialization)
k_new = torch.empty(512, 32, 128, dtype=torch.float16, device="cuda")
v_new = torch.empty(512, 32, 128, dtype=torch.float16, device="cuda")
```

**Result:**
```
[1] Request A — K-cache ptr: 0x7f5704c00000
    K[0,0,:4]: [313.25, 313.25, 313.25, 313.25]

[2] Request A completes — freed to pool

[3] Request B — New K-cache ptr: 0x7f5704c00000  ← same!
    Ptr reuse (K): True
    Ptr reuse (V): True

[4] Request B's torch.empty buffer:
    Non-zero elements: 2,097,152 / 2,097,152 (100.0%)
    K[0,0,:4]: [313.25, 313.25, 313.25, 313.25]   ← Request A's data!

[!!!] SAME ADDRESS — KV-cache from Request A directly readable.
```

**Conclusion:** PyTorch's `CUDACachingAllocator` returned the exact same pointer for Request B's `torch.empty` call. Every single one of the 2,097,152 float16 elements in the "empty" buffer contains Request A's KV-cache values. An attacker with access to Request B's buffer (e.g., through a timing attack, a malicious LoRA adapter, or a debug hook) can fully reconstruct Request A's attention state.

---

## What Can Be Inferred from KV-Cache Data?

KV-cache stores the output of the attention layer's key and value projections for every token in the prompt. Given leaked KV-cache values, it is theoretically possible to:

1. **Confirm token presence**: If the KV-cache was computed from a specific token, its key/value vectors will match what would be produced by the model for that token.
2. **Approximate the prompt**: With access to the model weights and leaked K/V vectors, gradient-based inversion can reconstruct likely input tokens.
3. **Identify the user's context**: Even without full prompt reconstruction, patterns in K/V activations can reveal the domain or topic of the previous request.

This is an active research area. The attack complexity is high, but the data leakage is complete and confirmed.

---

## Impact Assessment

| Scenario | Vulnerable? | Notes |
|----------|-------------|-------|
| Single-user local inference | No practical impact | You own your own data |
| Multi-user shared process (vLLM, TGI, SGLang) | **Yes** | Different users share same allocator pool |
| Container-isolated users, same GPU | No (cross-process safe on tested driver) | Each container has its own CUDA context |
| CUDA MPS (Multi-Process Service) | Potentially yes | Shared context = shared pool |
| LoRA adapter switching | **Yes** | Base model buffers reused across adapters without zeroing |

---

## Root Cause

This is **not a bug**. It is documented, intentional behavior:

- NVIDIA CUDA documentation explicitly states that stream-ordered allocators do not zero memory.
- PyTorch documentation states that `torch.empty` does not initialize tensors.

The issue is that inference server developers, when choosing `torch.empty` over `torch.zeros` for performance, may not realize they are creating a data confidentiality boundary between users sharing the same process.

---

## Mitigations

### 1. Use `torch.zeros` for security-sensitive buffers

```python
# Unsafe — previous request's data may be present
k_cache = torch.empty(seq_len, n_heads, head_dim, dtype=torch.float16, device="cuda")

# Safe — explicitly zeroed
k_cache = torch.zeros(seq_len, n_heads, head_dim, dtype=torch.float16, device="cuda")
```

**Cost:** `torch.zeros` requires a `cudaMemset` call (~1-2% overhead at typical KV-cache sizes).

### 2. Explicit cudaMemset after pool allocation

```c
cudaMallocAsync(&ptr, size, stream);
cudaMemsetAsync(ptr, 0, size, stream);  // explicit zero
```

### 3. Process-per-user isolation

Give each user their own process (CUDA context). The driver guarantees zeroing between contexts. This is the strongest isolation but has the highest overhead.

### 4. Disable pool allocator

```python
# Force cudaMalloc path (driver zeroes, slower)
import os
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
```

---

## Notes on False Positive Detection

During this research, an initial version of Experiment 1A used a byte-level pattern `(0xAB + (i & 0xFF)) % 256`. This produced a spurious "4 bytes leaked" result. The pattern wraps to zero at indices 85, 341, 597, 853 — meaning zeroed memory accidentally matched the expected value at exactly those positions. The experiment was re-run with a non-wrapping 32-bit pattern (`0xDEAD0000 + i`) which correctly showed zero matches on zeroed memory.

This is a subtle but important point for anyone doing similar research: always use detection patterns that are never zero.

---

## Environment

- **GPU:** NVIDIA GeForce RTX 4090 (24 GB VRAM)
- **Driver:** 565.77
- **CUDA:** 12.6 / 12.7
- **PyTorch:** 2.5.1+cu124
- **OS:** Ubuntu 22.04

## Code

All experiments are in this repository:
- `01_uninitialized_memory/` — cudaMalloc vs cudaMallocAsync
- `02_cross_process/` — cross-process isolation test
- `03_ml_inference/` — KV-cache leak in shared inference
