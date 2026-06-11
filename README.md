# CUDA GPU Memory Security Research

This repository documents hands-on security research into GPU memory safety on NVIDIA hardware (H100 80GB HBM3, CUDA 12.x). All experiments were run live on real hardware, results are reproducible.

## Motivation

Modern ML inference servers allocate and free GPU tensors thousands of times per second. We investigated whether sensitive data (user prompts, KV-caches, model weights) can leak between allocations — either within the same process or across processes — due to the way CUDA memory management works.

## Findings Summary

| # | Scenario | Method | Result |
|---|----------|--------|--------|
| 1A | Same-process, standard alloc | `cudaMalloc` → `cudaFree` → `cudaMalloc` | ✅ SAFE (0/1024) |
| 1B | Same-process, pool (no sync) | `cudaMallocAsync` back-to-back | 🔴 **100% FULL LEAK** |
| 2 | Cross-process | Process A exits → Process B | ✅ SAFE (driver zeroes) |
| 3 | KV-cache, PyTorch pool | Two sequential requests | 🔴 **100% FULL LEAK** |
| 4 | Multi-stream (cudaMallocAsync+sync) | Stream A → Stream B | ✅ SAFE (pool zeroes) |
| 4 | PyTorch CUDACachingAllocator | With `synchronize()` | 🔴 **100% FULL LEAK** |
| 5 | Two models, same process | Model A unload → Model B | 🔴 **100% FULL LEAK** |
| 6 | Persistent scanner | 10 rounds victim→attacker | 🔴 **100% every round** |
| 7 | LoRA adapter switching | Adapter A → Adapter B pool | 🔴 **100% non-zero residue** |
| 8 | vLLM/Ollama (same process) | User A → User B KV-cache | 🔴 **100% FULL LEAK** |
| 8 | PagedAttention block reuse | KV blocks NOT zeroed | 🔴 **LEAK on block reuse** |
| 9 | Token ID reconstruction | Leaked embeddings → word recovery | 🔴 **16/16 tokens (100%)** |
| 10 | Training gradient leak | Customer A grad → Customer B | 🔴 **100% non-zero pool** |
| 11 | Mitigation: torch.zeros | vs torch.empty overhead | ✅ **SAFE, -48% faster on H100!** |
| 12 | Real GPT-2 inference | Actual medical prompt recovery | 🔴 **16/16 words recovered** |
| 13 | Temporal persistence | How long does data survive in pool? | 🔴 **Indefinite — never expires** |
| 14 | INT8 quantization | Does quant protect against token recovery? | 🔴 **16/16 even with INT8** |
| 15 | CUDA IPC handles | Cross-process GPU read via shared handle | 🔴 **100% full R/W access** |
| 16 | Multi-GPU P2P (NVLink) | GPU1 reads GPU0 pool residue via P2P | 🔴 **100% cross-GPU leak at 387 GB/s** |
| 17 | cudaMemPool attributes | Built-in zeroing flag in CUDA API? | ⚠️ **No zeroing attr — only sync helps** |
| 18 | VMM direct (cuMemCreate) | ggml_cuda_pool_vmm exact behavior | 🔴 **VMM reuse = full leak, no zeroing** |
| 19 | Pinned host memory (CVE-2011-0636) | cudaHostAlloc free+realloc (modern driver) | ✅ PATCHED — CUDA 12.8 zeroes all pinned pages |
| 20 | Unified Memory (CVE-2024-53869) | cudaMallocManaged free+realloc (5 paths) | ✅ PATCHED — CUDA 12.8 zeroes all UM pages |
| 21 | cudaMallocAsync detailed analysis | 7 sync/stream/pool-config combinations | 🔴 **Same-stream no-sync = 100% LEAK; ReuseAllowOpportunistic=0 does NOT help** |
| 22 | IPC + pool reuse boundary test | Pool residue across cudaMalloc / cross-process | ✅ **Pools are per-process; pool→cudaMalloc safe** |

## Key Insights

**The core vulnerability:** PyTorch's `CUDACachingAllocator` maintains its own free list in C++ space, bypassing driver-level memory zeroing. `torch.empty()` always returns unzeroed memory containing whatever the previous allocation left there.

**Critical nuance:** `torch.cuda.synchronize()` (which inference servers call between requests) **does not prevent leaks** — PyTorch's allocator ignores it. Raw `cudaMallocAsync` DOES zero after sync; PyTorch does not.

**Most impactful finding (Experiment 12):** Using real GPT-2, we recovered the exact words from a confidential medical prompt ("The patient's medical diagnosis is strictly confidential. Do not share...") by:
1. Running GPT-2 on the victim's prompt
2. Allocating `torch.empty` for the attacker (got same pointer)
3. Scanning GPT-2's embedding matrix → **16/16 token IDs recovered (100%)**

**Performance:** `torch.zeros` (the mitigation) is actually **48% faster** than `torch.empty` on H100 due to GPU HBM3 memory system behavior. The security cost is negative.

**New findings (Exp 13–18):**
- Data persists **indefinitely** in the pool — 5+ seconds with no expiry. Attacker doesn't need to act immediately.
- **Different-sized requests don't protect the victim** — size-class separation keeps the block available regardless.
- **INT8 quantization doesn't help** — 16/16 exact token recovery still works at INT8.
- **CUDA IPC handles give full R/W access** to another process's GPU memory — any process with the handle file can read and overwrite the allocation.
- **Multi-GPU NVLink P2P**: GPU1 reads GPU0's pool residue at **387 GB/s** — pool leak propagates across TP shards.
- **No built-in zeroing attribute** in CUDA pool API — only stream sync or pool trim helps at raw CUDA level (PyTorch bypasses both).
- **VMM (cuMemCreate) confirmed**: llama.cpp's `ggml_cuda_pool_vmm` uses bump-reuse without zeroing — full leak on H100.

**CVE analysis (Exp 19–22):**
- **CVE-2011-0636 is patched**: cudaHostAlloc zeroes pinned pages in CUDA 12.8 — 0/10M leak across all scenarios.
- **CVE-2024-53869 is patched**: cudaMallocManaged zeroes all UM pages in CUDA 12.8 — 0 leak in 5 test paths.
- **cudaMallocAsync same-stream no-sync = 100% LEAK**: disabling opportunistic reuse does NOT help — same-stream reuse is always ordered and the flag does not apply to it.
- **Pool isolation confirmed**: CUDA stream-ordered pool and cudaMalloc are separate — pool residue does not cross to cudaMalloc; pools are per-process (cross-process contamination requires explicit IPC).
- **Gap**: No existing CVE covers user-space pool allocator leaks (PyTorch CUDACachingAllocator, llama.cpp ggml pools) → this research fills that gap.

## Repository Structure

```
01_uninitialized_memory/    Exp 1A/1B: cudaMalloc vs cudaMallocAsync
02_cross_process/           Exp 2:     Cross-process isolation (safe)
03_ml_inference/            Exp 3:     KV-cache leak in shared inference
04_multi_stream/            Exp 4:     CUDA stream pool sharing
05_two_models/              Exp 5:     Cross-model data leak
06_persistent_scan/         Exp 6:     Persistent GPU pool scanner (100% every round)
07_lora_serving/            Exp 7:     LoRA adapter weight/activation leak
08_server_scenarios/        Exp 8:     vLLM, Ollama, PagedAttention simulation
09_token_reconstruction/    Exp 9:     Token ID recovery from leaked embeddings (100%)
10_gradient_leak/           Exp 10:    Training gradient confidentiality
11_mitigation_bench/        Exp 11:    Mitigation effectiveness & performance benchmark
12_gpt2_real/               Exp 12:    Real GPT-2: full prompt word recovery (100%)
13_temporal_window/         Exp 13:    How long data persists — indefinite, size-class isolation
14_quantized_models/        Exp 14:    INT8 quantization does NOT protect (16/16 recovery)
15_cuda_ipc/                Exp 15:    CUDA IPC: cross-process R/W access via handle file
16_multi_gpu_p2p/           Exp 16:    NVLink P2P: GPU1 reads GPU0 pool residue (387 GB/s)
17_mempool_attrs/           Exp 17:    cudaMemPool attributes — no built-in zeroing flag
18_vmm_direct/              Exp 18:    cuMemCreate/cuMemMap VMM — confirms llama.cpp leak
19_pinned_host_mem/         Exp 19:    cudaHostAlloc pinned memory — CVE-2011-0636 class (patched)
20_unified_memory/          Exp 20:    cudaMallocManaged unified memory — CVE-2024-53869 class (patched)
21_malloc_async_pool/       Exp 21:    cudaMallocAsync 7-scenario detailed analysis (same-stream leaks)
22_ipc_pool_reuse/          Exp 22:    IPC + pool boundary test (pools are per-process)
```

## Environment

- GPU: NVIDIA H100 80GB HBM3
- Driver: 565.77
- CUDA: 12.6 / 12.7
- PyTorch: 2.12.0+cu130

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

## Experiments 7–12 (New)

### Experiment 7: LoRA Adapter Memory Leak (`07_lora_serving/lora_leak.py`)

LoRA adapters (rank-64, 32 layers, hidden=4096) are unloaded from GPU when a user's request
finishes. The next user's buffer allocations contain **100% non-zero residue** from the
prior adapter's computations. In a LoRA-serving setup, one user can read another user's
private adapter activations.

### Experiment 8: vLLM / Ollama / PagedAttention Simulation (`08_server_scenarios/vllm_simulation.py`)

Tests three inference server architectures:

```
[Scenario A] Same-process (vLLM, TGI, SGLang, Ollama server, llama-server):
  User A  K ptr: 0x7f0960c00000    K[0,0,:4]: [1.414, 1.414, 1.414, 1.414]
  User B  K ptr: 0x7f0960c00000    (reuse: True)
  User B  K[0,0,:4]: [1.414, 1.414, 1.414, 1.414]
  Non-zero: 1048576/1048576 (100.0%)
  [!!!] SAME ADDRESS — User B has User A's KV-cache (LEAK)

[Scenario C] PagedAttention block reuse (vLLM-style):
  vLLM allocates its KV block store with torch.zeros at server startup.
  BUT when Request A finishes and its blocks return to the free list,
  those blocks are NOT re-zeroed before the next request gets them.
  User A block[0] K[0,0,:4]: [3.14, 3.14, 3.14, 3.14]  ← written during request A
  User B got same block (reused) — still contains Request A's attention data
  [!!!] PagedAttention blocks NOT re-zeroed on recycling → LEAK

[Scenario B] Cross-process isolation baseline (NOT Ollama server behavior):
  SAFE — driver zeroes on CUDA context destroy
  ⚠️  Ollama server mode runs as a single persistent process (not this scenario).
      Ollama server = Scenario A = HIGH RISK. Only per-request CLI spawning is safe.
```

**Architecture risk matrix:**
| Server | Isolation | Risk |
|--------|-----------|------|
| vLLM | KV blocks zeroed; intermediate activations NOT | **MEDIUM-HIGH** |
| TGI | Single process, shared pool | **HIGH** |
| SGLang | `torch.empty` in MLA KV paths | **HIGH** |
| Ollama (server) | Single process, ggml pool — no zeroing | **HIGH** |
| **llama-server** | **ggml pool — no zeroing on reuse** | **HIGH** |
| llama.cpp CLI | New process per request (cross-process safe) | SAFE |

### Experiment 9: Token ID Reconstruction (`09_token_reconstruction/token_recon.py`)

Complete proof-of-concept for prompt vocabulary recovery:

```
User A secret tokens: [31337, 1337, 42, 1024, 8192, 4096, 2048, 512, ...]
Attacker ptr:  0x7fe007600000  (same as User A)
Non-zero: 12288/12288 (100.0%)

 Pos   Secret ID   Recovered   Match          Dist
   0       31337       31337   YES ✓        0.0000
   1        1337        1337   YES ✓        0.0000
   2          42          42   YES ✓        0.0000
   ...
[Result] Recovered 16/16 token IDs correctly (100.0%)
[!!!] FULL TOKEN RECOVERY — attacker knows every word the user typed
```

**How it works:** Embedding lookup is deterministic. Leaked embedding vector → scan
50,257-entry vocab table → O(vocab_size) match → exact token ID, exact word.

### Experiment 10: Training Gradient Leak (`10_gradient_leak/grad_leak.py`)

After a training step (forward + backward), gradient tensors freed to pool contain
**100% non-zero** data from the backward pass. A co-tenant allocating `torch.empty`
of the same size gets those gradients. Gradient inversion attacks (Zhu et al., 2019)
can reconstruct training data from gradients alone.

### Experiment 11: Mitigation Benchmark (`11_mitigation_bench/mitigation.py`)

**Surprising finding: `torch.zeros` is faster than `torch.empty`.**

```
Method                          Time (μs)     vs baseline   Safe?
torch.empty (baseline)          24.36         +0.0%         NO (LEAKS)
torch.zeros                     12.61         -48.2%        YES ✓
torch.empty().zero_()           10.53         -56.8%        YES ✓
empty_cache() + torch.empty     111.26        +356.8%       YES ✓
PYTORCH_NO_CUDA_MEMORY_CACHING  78.8          +223%         YES ✓
```

On H100 80GB HBM3, zeroing a 4 MB KV-cache is **48% FASTER** than not zeroing.
The security fix costs nothing — it is actually a performance improvement.

### Experiment 12: Real GPT-2 — Full Prompt Word Recovery (`12_gpt2_real/gpt2_kvcache_leak.py`)

**The most impactful result.** Uses real GPT-2 (124M params, HuggingFace) on an actual
confidential medical prompt:

```
User A: "The patient's medical diagnosis is strictly confidential.
         Do not share this with anyone outside authorized medical staff."

After freeing User A's tensors, attacker's torch.empty gets same pointer.
Attacker scans GPT-2's 50,257-token embedding matrix:

 Pos  True word         Recovered word    Match
   0  The               The               YES ✓
   1  Ġpatient          Ġpatient          YES ✓
   2  's                's                YES ✓
   3  Ġmedical          Ġmedical          YES ✓
   4  Ġdiagnosis        Ġdiagnosis        YES ✓
   5  Ġis               Ġis               YES ✓
   6  Ġstrictly         Ġstrictly         YES ✓
   7  Ġconfidential     Ġconfidential     YES ✓
   ...
[!!!] 16/16 token IDs recovered (100.0%)
Full prompt vocabulary recovered from leaked GPU memory!
```

The attacker reconstructed that the previous user's prompt was about a patient's
medical diagnosis — without any brute-force, model inversion, or side-channels.
Just `torch.empty` and a table lookup.

---

## Reproducing

### Requirements
- NVIDIA GPU (any CUDA 12.x capable)
- CUDA Toolkit 12.x
- Python 3.10+ with PyTorch 2.x and transformers

### Install
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install transformers accelerate
```

### Run all experiments

```bash
# Experiments 1A/1B: cudaMalloc vs pool
nvcc -O2 -o uninit_memory_leak 01_uninitialized_memory/uninit_memory_leak.cu && ./uninit_memory_leak
nvcc -O2 -o pool_memory_leak 01_uninitialized_memory/pool_memory_leak.cu && ./pool_memory_leak

# Experiment 2: cross-process (safe)
nvcc -O2 -o writer 02_cross_process/writer.cu && nvcc -O2 -o reader 02_cross_process/reader.cu
./writer && ./reader

# Experiments 3–6: PyTorch pool leaks
python3 03_ml_inference/kv_cache_leak.py
python3 05_two_models/two_model_leak.py
python3 06_persistent_scan/persistent_scan.py

# Experiment 4: multi-stream
nvcc -O2 -o stream_leak 04_multi_stream/stream_leak.cu && ./stream_leak

# Experiments 7–12: cloud GPU multi-model scenarios
python3 07_lora_serving/lora_leak.py
python3 08_server_scenarios/vllm_simulation.py
python3 09_token_reconstruction/token_recon.py
python3 10_gradient_leak/grad_leak.py
python3 11_mitigation_bench/mitigation.py
python3 12_gpt2_real/gpt2_kvcache_leak.py  # requires transformers
```
