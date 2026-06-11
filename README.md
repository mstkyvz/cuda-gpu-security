# CUDA GPU Memory Security Research

This repository documents hands-on security research into GPU memory safety on NVIDIA hardware (RTX 4090, CUDA 12.x). All experiments were run live, results are reproducible.

## Motivation

Modern ML inference servers allocate and free GPU tensors thousands of times per second. We investigated whether sensitive data (user prompts, KV-caches, model weights) can leak between allocations — either within the same process or across processes — due to the way CUDA memory management works.

## Findings Summary

| Scenario | Method | Result |
|----------|--------|--------|
| Same-process, standard alloc | `cudaMalloc` → `cudaFree` → `cudaMalloc` | ⚠️ 0.4% partial residue |
| Same-process, pool alloc | `cudaMallocAsync` → `cudaFreeAsync` → `cudaMallocAsync` | 🔴 **100% FULL LEAK** |
| Same-process, PyTorch tensors | `torch.empty` after `del tensor` | 🔴 **100% FULL LEAK** |
| ML inference KV-cache | Two requests sharing pool | 🔴 **100% FULL LEAK** |
| Cross-process | Process A exits → Process B allocates | ✅ SAFE (driver zeroes) |

## Key Insight

`cudaMalloc` zero-initializes GPU memory (secure). But `cudaMallocAsync` and PyTorch's `CUDACachingAllocator` **do not** — they return previously used memory directly from a pool for performance. This means `torch.empty()` is not actually empty.

## Repository Structure

```
01_uninitialized_memory/    Experiment 1: cudaMalloc vs cudaMallocAsync
02_cross_process/           Experiment 2: Cross-process isolation test  
03_ml_inference/            Experiment 3: KV-cache leak in shared inference
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
Old ptr: 0x7f62c4c00000
New ptr: 0x7f62c4c00000   ← same address returned

Bytes matching secret pattern: 4 / 1024 (0.4%)
[~] PARTIAL LEAK: 4 bytes of previous data still readable.
```

**Analysis:** The NVIDIA driver scrubs most of the memory on allocation, but 4 bytes were not zeroed. This may be coincidental (the pattern happens to match) or a genuine residue. Either way, `cudaMalloc` is largely safe due to driver-level scrubbing.

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

## Reproducing

### Requirements
- NVIDIA GPU (any modern)
- CUDA Toolkit 12.x
- Python 3.10+ with PyTorch

### Run all experiments

```bash
# Experiment 1A
nvcc -O2 -o uninit_memory_leak 01_uninitialized_memory/uninit_memory_leak.cu
./uninit_memory_leak

# Experiment 1B
nvcc -O2 -o pool_memory_leak 01_uninitialized_memory/pool_memory_leak.cu
./pool_memory_leak

# Experiment 2
nvcc -O2 -o writer 02_cross_process/writer.cu
nvcc -O2 -o reader 02_cross_process/reader.cu
./writer && ./reader

# Experiment 3
python3 03_ml_inference/kv_cache_leak.py
```
