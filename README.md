# CUDA GPU Memory Security Research

This repository documents hands-on security research into GPU memory safety on NVIDIA hardware (RTX 4090, CUDA 12.x). All experiments were run live, results are reproducible.

## Motivation

Modern ML inference servers allocate and free GPU tensors thousands of times per second. We investigated whether sensitive data (user prompts, KV-caches, model weights) can leak between allocations — either within the same process or across processes — due to the way CUDA memory management works.

## Findings Summary

| Scenario | Method | Result |
|----------|--------|--------|
| Same-process, standard alloc | `cudaMalloc` → `cudaFree` → `cudaMalloc` | ✅ SAFE (driver zeroes, 0/1024) |
| Same-process, pool (no sync) | `cudaMallocAsync` back-to-back on same stream | 🔴 **100% FULL LEAK** |
| Same-process, pool (with sync) | `cudaMallocAsync` + `cudaStreamSynchronize` + realloc | ✅ SAFE (pool zeroes after sync) |
| Same-process, PyTorch (with sync) | `torch.empty` + `torch.cuda.synchronize()` + `torch.empty` | 🔴 **100% FULL LEAK** |
| ML inference KV-cache | Two sequential requests, same PyTorch process | 🔴 **100% FULL LEAK** |
| Two models, same process | Model A unload → Model B allocates hidden buffers | 🔴 **100% FULL LEAK** |
| Persistent scanner | 10 rounds: victim fill → attacker `torch.empty` | 🔴 **100% every round** |
| Cross-process | Process A exits → Process B allocates | ✅ SAFE (driver zeroes) |
| Cross-stream (cudaMallocAsync) | Stream A free → Stream B alloc (with sync) | ✅ SAFE (pool zeroes) |

## Key Insight

`cudaMalloc` and the raw `cudaMallocAsync` pool (with stream sync) both zero-initialize GPU memory. **PyTorch's `CUDACachingAllocator` does not** — it maintains its own free list in Python/C++ space and bypasses driver-level zeroing entirely. This means `torch.empty()` is genuinely uninitialized and always contains whatever the previous occupant left there.

Critical nuance: `torch.cuda.synchronize()` (which every inference server calls between requests) **does not prevent the leak** — PyTorch's allocator ignores it.

## Repository Structure

```
01_uninitialized_memory/    Experiments 1A/1B: cudaMalloc vs cudaMallocAsync
02_cross_process/           Experiment 2: Cross-process isolation test  
03_ml_inference/            Experiment 3: KV-cache leak in shared inference
04_multi_stream/            Experiment 4: CUDA stream pool sharing
05_two_models/              Experiment 5: Cross-model data leak
06_persistent_scan/         Experiment 6: Persistent GPU pool scanner
```

## Environment

- GPU: NVIDIA GeForce RTX 4090 (24 GB)
- Driver: 565.77
- CUDA: 12.6 / 12.7
- PyTorch: 2.5.1+cu124

## Mitigation

| Unsafe | Safe |
|--------|------|
| `torch.empty(...)` | `torch.zeros(...)` |
| `cudaMallocAsync` without init | `cudaMemset` after alloc |
| Reusing pool buffers across requests | Explicit zero-fill between requests |

---

## Experiment 1 — Uninitialized Memory Read

**Directory:** `01_uninitialized_memory/`

### Background

CUDA provides two primary allocation paths:
- `cudaMalloc`: Allocates from the device heap. NVIDIA drivers zero-initialize this memory before returning it to the caller (since driver ~418.x).
- `cudaMallocAsync` (stream-ordered memory pool): Allocates from a pre-allocated pool. Returns memory **without zeroing** — by design, for performance.

PyTorch's `CUDACachingAllocator` is built on top of the pool allocator. When you call `del tensor` or let a tensor go out of scope, the underlying GPU memory is returned to PyTorch's internal pool — **not zeroed**.

### Test 1A: `cudaMalloc` (uninit_memory_leak.cu)

Write a byte pattern to a GPU buffer, free it with `cudaFree`, allocate a new buffer of the same size, read without initializing.

**Result:**
```
Old ptr: 0x7ff438c00000
New ptr: 0x7ff438c00000   ← same address returned

Non-zero words:          0 / 1024
Matching secret pattern: 0 / 1024 (0.00%)

[SAFE] Driver zeroed memory on cudaMalloc.
```

**Analysis:** The NVIDIA driver zero-initializes memory on every `cudaMalloc` call. Standard allocations are safe. Note: an earlier version of this test used a byte pattern that wraps to zero, producing 4 false-positive matches. The final version uses `0xDEAD0000 + i` (never zero) to eliminate this class of error.

### Test 1B: `cudaMallocAsync` Pool Allocator (pool_memory_leak.cu)

Write a float pattern via `cudaMallocAsync`, free with `cudaFreeAsync`, immediately reallocate from the same pool.

**Result:**
```
Old ptr: 0x302000000
New ptr: 0x302000000   ← exact same address

Floats matching secret pattern: 4096 / 4096 (100.0%)
Max error vs expected: 0.000000

[!] FULL LEAK confirmed via memory pool.
First 8 values: 3.1416 6.2832 9.4248 12.5664 15.7080 18.8495 21.9911 25.1327
Expected first 8: 3.1416 6.2832 9.4248 12.5664 15.7080 18.8495 21.9911 25.1327
```

**Analysis:** 100% of previous data is readable. The pool allocator never zeroed the memory. This is documented behavior — `cudaMallocAsync` explicitly skips initialization for performance.

---

## Experiment 2 — Cross-Process Isolation

**Directory:** `02_cross_process/`

### Background

If NVIDIA drivers do not zero GPU memory when a CUDA context is destroyed (process exit), a new process could potentially read data left by the previous process. This would be critical in multi-tenant cloud GPU environments.

### Method

- **writer.cu**: Allocates GPU memory, writes `0xDEAD0000 + index` pattern (non-zero, non-wrapping), exits **without calling `cudaFree`**. The CUDA context is destroyed on process exit.
- **reader.cu**: Immediately after writer exits, allocates same-sized buffer, reads without initializing.

**Result:**
```
[writer] Wrote 0xDEAD0000+i pattern to GPU ptr: 0x7f9c2ac00000
[writer] Exiting WITHOUT cudaFree

[reader] Got GPU ptr: 0x7f3112c00000

Non-zero words:       0 / 16384 (0.00%)
Match writer pattern: 0 / 16384 (0.0000%)

[SAFE] Driver zeroed GPU memory between processes.
```

**Analysis:** Modern NVIDIA drivers (565.77) zero GPU memory when a CUDA context is destroyed, before returning it to the OS allocator. Cross-process isolation is intact on this driver version. Note: older drivers and some virtualized environments may not behave the same way.

---

## Experiment 3 — ML Inference KV-Cache Leak

**Directory:** `03_ml_inference/`

### Background

In a real LLM inference server:
1. Request A is processed — a KV-cache tensor is allocated and filled
2. Request A finishes — the KV-cache tensor is deleted (`del`, goes out of scope)
3. PyTorch returns the GPU memory to `CUDACachingAllocator` (not zeroed)
4. Request B arrives — a new KV-cache tensor is allocated with `torch.empty`
5. Request B gets the same memory address — Request A's KV-cache is still there

### Result

```
[1] Request A — K-cache ptr: 0x7f5704c00000
    K[0,0,:4]: [313.25, 313.25, 313.25, 313.25]

[2] Request A completes — tensors freed to pool

[3] Request B — New K-cache ptr: 0x7f5704c00000  ← same!
    Ptr reuse (K): True
    Ptr reuse (V): True

[4] Request B's 'empty' buffer:
    K non-zero elements: 2097152 / 2097152 (100.0%)
    K[0,0,:4]: [313.25, 313.25, 313.25, 313.25]   ← Request A's data!

[!!!] SAME ADDRESS — KV-cache from Request A directly readable.

[5] Mitigation (torch.zeros):
    Non-zero elements: 0 / 2097152 ✓
```

**Analysis:** When a shared inference server (e.g., vLLM, TGI, SGLang) reuses GPU memory between requests without explicit zeroing, one user's KV-cache is directly readable in the next user's allocation. The data includes attention keys and values derived from input tokens — potentially allowing reconstruction of parts of the previous user's prompt.

### Real-World Exposure

This affects:
- Any inference server that uses `torch.empty` for KV-cache allocation
- Systems where multiple users share the same GPU process (not just containers)
- Batch inference where request buffers are reused without zeroing

### CVE Status

This is **documented behavior**, not a driver bug. NVIDIA and PyTorch both state that pool allocators do not zero memory. However:

- It constitutes a **data confidentiality risk** in multi-user inference servers
- Inference server developers should use `torch.zeros` or explicit `cudaMemset` for security-sensitive buffers
- Some systems already do this (vLLM has options for secure allocation) but it is not the default

---

## Experiment 4 — Multi-Stream Pool Behavior

**Directory:** `04_multi_stream/`

### Background

CUDA stream-ordered pools (`cudaMallocAsync`) are per-device, not per-stream. But the pool's
zeroing behavior depends on whether the CPU synchronized between free and realloc.

### Results

```
[TEST 1] Writer and reader on the SAME CUDA stream:
    Writer ptr : 0x302000000
    Reader ptr : 0x302000000  (same: YES)
    Non-zero   : 0 / 4096
    --> SAFE (pool zeroed after stream sync)

[TEST 2] Writer on Stream A, reader on Stream B (different streams):
    Writer ptr : 0x302000000
    Reader ptr : 0x302000000  (same: YES)
    Non-zero   : 0 / 4096
    --> SAFE (pool zeroed cross-stream)
```

**Analysis:** When `cudaStreamSynchronize()` is called between `cudaFreeAsync` and `cudaMallocAsync`,
the CUDA driver zeroes the block before returning it — in both same-stream and cross-stream cases.
This is a safety mechanism in the raw CUDA pool allocator.

**Critical nuance:** PyTorch's `CUDACachingAllocator` does NOT use `cudaMallocAsync`. It maintains
its own free list in C++ space, bypassing the driver's zeroing. This is why PyTorch shows 100% leak
even after `torch.cuda.synchronize()`, while raw `cudaMallocAsync` tests show safe results when synced.

---

## Experiment 5 — Two Models, Same Process

**Directory:** `05_two_models/`

### Background

In a multi-model serving system (e.g., running multiple LLM adapters), models are loaded and unloaded
sequentially. If Model A's tensors are freed to the PyTorch pool and Model B allocates from the same pool,
Model B's buffers contain Model A's activation data.

### Result

```
[1] Loading Model A (hidden=8192, batch=256)...
    Model A hidden ptr   : 0x7f16dce00000  (size 4096 KB)
    Model A hidden[0,:4] : [1.2861, 0.8237, -1.3125, 1.0205]

[2] Unloading Model A — returned to pool, NOT zeroed

[3] Loading Model B...
    Model B hidden ptr : 0x7f16dce00000   ← same address!
    Same as Model A    : True

[4] Model B 'empty' hidden buffer:
    Non-zero elements : 2,096,896 / 2,097,152 (100.0%)
    Buffer[0,:4]      : [1.2861, 0.8237, -1.3125, 1.0205]   ← Model A's data!
    Value match [0,:4]: [True, True, True, True]

[5] Mitigation torch.zeros: 0 / 2,097,152 ✓
```

**Analysis:** Model B's first hidden-layer buffer was allocated with `torch.empty` and returned the
exact same pointer as Model A's hidden layer. All 2,096,896 elements are non-zero and bit-identical
to Model A's hidden activations. In a LoRA adapter serving setup, this means base model residual
buffers can contain the previous adapter's private fine-tuned activations.

---

## Experiment 6 — Persistent Pool Scanner

**Directory:** `06_persistent_scan/`

### Background

Simulates a persistent attacker on a shared GPU server that repeatedly allocates `torch.empty`
tensors after each request to scan for residual data. Runs 10 rounds with different victim patterns.

### Result

```
Round   Secret      Ptr-reuse   Leaked/Total      Match%
1       2.71828     YES         524,286/524,288    100.0%
         Leaked  [0:4]: ['2.99011', '2.71828', '2.71828', '2.71828']
         Expected     : 2.71828 (all elements should be 2.71828)
         Marker[0]    : 2.99011  (expected 2.99011)   ← custom marker preserved!
         Marker[-1]   : 26.91097 (expected 26.91097)  ← end marker preserved!
2       3.14159     YES         524,286/524,288    100.0%
3       1.41421     YES         524,286/524,288    100.0%
[... 7 more rounds, all 100% ...]

Total leaked: 5,242,860/5,242,880 (100.0%)
torch.cuda.synchronize() between rounds: YES — does NOT prevent leak
```

**Analysis:** Across all 10 rounds, the attacker recovered 100% of the victim's tensor data. The
custom marker values (`secret * 1.1` at index 0, `secret * 9.9` at index -1) were recovered exactly,
confirming that full tensor contents — not just statistical residue — are accessible. `synchronize()`
between rounds has no effect because PyTorch's allocator never zeroes memory.

---

## Reproducing

### Requirements
- NVIDIA GPU (any modern)
- CUDA Toolkit 12.x
- Python 3.10+ with PyTorch

### Run all experiments

```bash
# Experiment 1A — cudaMalloc (safe)
nvcc -O2 -o uninit_memory_leak 01_uninitialized_memory/uninit_memory_leak.cu
./uninit_memory_leak

# Experiment 1B — cudaMallocAsync pool (100% leak)
nvcc -O2 -o pool_memory_leak 01_uninitialized_memory/pool_memory_leak.cu
./pool_memory_leak

# Experiment 2 — cross-process isolation (safe)
nvcc -O2 -o writer 02_cross_process/writer.cu
nvcc -O2 -o reader 02_cross_process/reader.cu
./writer && ./reader

# Experiment 3 — KV-cache leak (100% leak)
python3 03_ml_inference/kv_cache_leak.py

# Experiment 4 — multi-stream behavior
nvcc -O2 -o stream_leak 04_multi_stream/stream_leak.cu
./stream_leak

# Experiment 5 — two models, same process (100% leak)
python3 05_two_models/two_model_leak.py

# Experiment 6 — persistent pool scanner (100% every round)
python3 06_persistent_scan/persistent_scan.py
```
