# GPU Memory Confidentiality in Multi-Tenant LLM Inference

**Author:** mustafayvz-3211@hotmail.com  
**Hardware:** NVIDIA H100 80GB HBM3  
**CUDA / Driver:** 12.6 / 565.77  
**PyTorch:** 2.12.0+cu130  
**Date:** June 2026

---

## Executive Summary

We ran thirteen controlled experiments on a production-grade GPU to measure real-world memory confidentiality guarantees when running multiple LLM inference requests on the same GPU. The core finding:

> **PyTorch's `CUDACachingAllocator` never zeroes GPU memory. Any tensor allocated with `torch.empty` contains the previous occupant's data verbatim. In multi-user inference servers (vLLM, TGI, SGLang, Ollama), this means one user can read another user's prompt tokens, KV-cache, and intermediate activations — with 100% fidelity and zero false positives.**

We escalate this from "data is there" to "we can recover exact words." Experiment 12 demonstrates live recovery of the full vocabulary of a GPT-2 prompt about a *"strictly confidential medical diagnosis"* from leaked GPU memory with 16/16 exact words.

The mitigation is also counterintuitive: **`torch.zeros` is 48% faster than `torch.empty`** on H100 due to GPU memory sub-system behavior. The "zero overhead" mitigation actually improves performance.

---

## Background

### The Two CUDA Allocation Paths

**Path 1 — `cudaMalloc` (safe)**

Every call goes directly to the CUDA driver, which allocates fresh memory from the device heap. Since driver ~418.x (2019), NVIDIA zero-initializes this memory before returning the pointer. The caller always receives clean, zeroed memory regardless of what the physical DRAM held previously.

**Path 2 — `cudaMallocAsync` / pool allocator (unsafe by default)**

Introduced in CUDA 11.2, this API maintains a per-stream memory pool. The first call allocates a large block from the driver; subsequent calls subdivide and reuse that block without zeroing. This is intentional — NVIDIA documents that pool allocators do not initialize memory. The caller is responsible for initialization if confidentiality matters.

**Path 3 — PyTorch `CUDACachingAllocator` (always unsafe)**

PyTorch does not use `cudaMalloc` for most tensor allocations. It maintains its own free-list (the `CUDACachingAllocator`) on top of the CUDA pool. When `del tensor` or scope exit frees a tensor, the memory goes back to PyTorch's pool — never to the driver. The allocator does not zero on free or on the next allocation. `torch.empty(...)` always contains previous occupant data if the pool has a matching block.

This is not a bug. It is the documented, intended behavior. The problem is that ML framework users often assume "empty" means "zero" and that this distinction matters only for performance, not for confidentiality.

### Why This Matters for Inference Servers

A production LLM inference server processes thousands of requests per second:

```
Request A (user Alice) → allocate KV-cache → forward pass → del KV-cache
                                                                    ↓
                                                          pool: [block 0x...00 → 8MB, dirty]
                                                                    ↓
Request B (user Bob)  → torch.empty(KV-cache shape) ← same block returned
                        Bob's "empty" buffer = Alice's exact KV-cache values
```

The pool does not know or care that Alice and Bob are different users. The block is the right size, it is in the free list, it is returned. Bob's tensor contains Alice's prompt information — every byte of it.

---

## Experiments

### Experiment 1A: `cudaMalloc` — Standard Driver Allocator

**Goal:** Verify that the standard CUDA allocator zeroes memory between allocations.

**Method:**
1. Allocate 4 KB with `cudaMalloc`
2. Write pattern: `word[i] = 0xDEAD0000 + i` (never zero; avoids false-positive matches on zeroed memory)
3. Free with `cudaFree`
4. Allocate a fresh 4 KB with `cudaMalloc`
5. Count non-zero words and pattern matches

**Result (H100):**
```
Old ptr: 0x7ff438c00000
New ptr: 0x7ff438c00000   ← same physical address reused

Non-zero words          : 0 / 1024
Matching secret pattern : 0 / 1024  (0.00%)

[SAFE] Driver zeroed memory on cudaMalloc.
```

**Conclusion:** NVIDIA zeroes on `cudaMalloc`. Despite returning the same physical address, all 1024 words are zero. The standard allocator is safe.

---

### Experiment 1B: `cudaMallocAsync` — Pool Allocator

**Goal:** Verify pool allocator leaks.

**Method:** Identical to 1A but using `cudaMallocAsync` / `cudaFreeAsync`.

**Result (H100):**
```
Old ptr: 0x302000000
New ptr: 0x302000000   ← same address

Floats matching secret pattern : 4096 / 4096  (100.0%)
Max error vs expected          : 0.000000

[!!!] FULL LEAK confirmed via memory pool.
First 8 values: 3.1416  6.2832  9.4248  12.5664  15.7080  18.8495  21.9911  25.1327
Expected      : 3.1416  6.2832  9.4248  12.5664  15.7080  18.8495  21.9911  25.1327
```

**Conclusion:** Pool allocator returns the same pointer with all data intact. 4096/4096 floats match exactly. This is expected behavior per CUDA documentation — the point is that any ML framework built on this allocator inherits the leakage.

---

### Experiment 2: Cross-Process Isolation

**Goal:** Test whether a new process can read a previous process's GPU data.

**Method:**
- **Writer process:** Allocates 64 KB, writes `0xDEAD0000 + i`, then exits *without calling `cudaFree`*. CUDA context is destroyed on process exit.
- **Reader process:** Starts immediately after, allocates 64 KB, reads without writing.

**Result (H100):**
```
[writer] Wrote pattern to GPU ptr: 0x7f9c2ac00000
[writer] Exiting WITHOUT cudaFree

[reader] Got GPU ptr: 0x7f3112c00000

Non-zero words       : 0 / 16384  (0.00%)
Match writer pattern : 0 / 16384  (0.0000%)

[SAFE] Driver zeroed GPU memory between processes.
```

**Conclusion:** The CUDA driver zeroes memory when a context is destroyed. Different-process scenarios (separate Docker containers, separate `llama.cpp` invocations) are safe on driver 565.77.

**Important nuance:** This guarantee is driver- and hypervisor-dependent. CUDA MPS (Multi-Process Service), which shares a single CUDA context across processes for throughput, removes this protection — MPS users share one pool.

---

### Experiment 3: ML Inference KV-Cache Leak

**Goal:** Simulate a real inference server scenario and measure the leak.

**Setup:**
- KV-cache: `512 × 32 × 128` float16 (8 MB per request)
- Request A allocates, fills with recognizable values, is freed to pool
- Request B allocates with `torch.empty`, reads the result

**Result (H100):**
```
[1] Request A — K-cache ptr: 0x7f5704c00000
    K[0,0,:4]: [313.25, 313.25, 313.25, 313.25]

[2] Request A completes — freed to pool (NOT zeroed)

[3] Request B — New K-cache ptr: 0x7f5704c00000  ← same!
    Ptr reuse (K): True
    Ptr reuse (V): True

[4] Request B's torch.empty buffer:
    Non-zero elements : 2,097,152 / 2,097,152  (100.0%)
    K[0,0,:4]         : [313.25, 313.25, 313.25, 313.25]  ← Request A's data!

[!!!] SAME ADDRESS — KV-cache from Request A directly readable in Request B.
```

**Conclusion:** 2,097,152/2,097,152 float16 elements — the full 8 MB KV-cache — are readable in Request B's "empty" tensor. This is the core vulnerability for multi-user inference servers.

---

### Experiment 4: Multi-Stream Safety (`cudaMallocAsync` with Synchronization)

**Goal:** Test whether using different CUDA streams provides isolation.

**Method:**
- `cudaMallocAsync` + `cudaFreeAsync` with explicit `cudaStreamSynchronize` between free and re-allocation
- Tested for same-stream and cross-stream scenarios

**Result (H100):**
```
TEST1 (same stream + sync):
  Old ptr: 0x7f6c0dc00000  New ptr: 0x7f6c0dc00000
  Non-zero: 0/4096  [SAFE]

TEST2 (different stream + sync):
  Old ptr: 0x7f6c0dc00000  New ptr: 0x7f6c0dc00000
  Non-zero: 0/4096  [SAFE]
```

**Conclusion:** When `cudaStreamSynchronize` is called between `cudaFreeAsync` and the next `cudaMallocAsync`, the CUDA pool runtime zeroes the block before returning it. This is safe.

**Critical nuance:** PyTorch's `CUDACachingAllocator` does **not** call `cudaStreamSynchronize` between free and re-allocation in normal operation. PyTorch calls its own pool directly, bypassing the CUDA pool's sync-triggered zeroing. This is why `torch.cuda.synchronize()` between requests in a Python server does **not** prevent the leak.

---

### Experiment 5: Two Models on Same GPU

**Goal:** Can Model B read Model A's activations on a shared GPU?

**Setup:**
- Two `SmallMLP` instances (hidden=8192, batch=256), both on same GPU, same process
- Model A forward pass → hidden state freed → Model B forward pass with `torch.empty` intermediate

**Result (H100):**
```
Model A hidden ptr : 0x7f16dce00000  (4096 KB)
Model B hidden ptr : 0x7f16dce00000  ← SAME

Non-zero elements : 2,096,896 / 2,097,152  (100.0%)
Value match [0,:4]: [True, True, True, True]
```

**Conclusion:** When two models run in the same process sequentially, Model B's intermediate activation buffers contain Model A's data. In a multi-model serving scenario (e.g., multiple LoRA adapters behind one process, or batched requests processed by different model versions), activations are not isolated.

---

### Experiment 6: Persistent Pool Scanner

**Goal:** Prove the leak is deterministic and persistent across many rounds, even with explicit GPU synchronization between them.

**Method:**
- 10 rounds: victim fills `secret × different_value` → `del victim` → `torch.cuda.synchronize()` → `torch.empty` attacker
- Custom markers at index 0 (`secret × 1.1`) and last index (`secret × 9.9`) for exact verification

**Result (H100):**
```
Buffer: 2048 KB × 10 rounds  |  Sync: YES after every victim del

Round  Secret      Ptr-reuse  Leaked/Total      Match%
------------------------------------------------------
1      2.71828     YES        524286 /524288    100.0%
         Leaked [0:4]: ['2.99011', '2.71828', '2.71828', '2.71828']
         Marker[0]  : 2.99011  (expected 2.99011)  ✓
         Marker[-1] : 26.91097  (expected 26.91097) ✓
2      3.14159     YES        524286 /524288    100.0%
3      1.41421     YES        524286 /524288    100.0%
4      1.61803     YES        524286 /524288    100.0%
[... rounds 5-10 identical: 524286/524288 = 100.0% every round ...]

Total leaked: 5,242,860 / 5,242,880  (100.0%)
Custom markers recovered exactly in all 10 rounds.
```

**Conclusion:** `torch.cuda.synchronize()` between requests does not prevent the leak. The PyTorch allocator pool is independent of CUDA stream synchronization. The leak is deterministic and 100% reproducible across every round with different secret values. The exact marker values are recovered, proving this is real data extraction — not statistical noise.

---

### Experiment 7: LoRA Adapter Serving

**Goal:** Test whether switching LoRA adapters leaks between them.

**Setup:**
- LoRA adapter A (rank=64, hidden=4096, 32 layers, 64 MB) loaded and used for forward pass
- Pool freed, adapter B loaded, activations allocated with `torch.empty`

**Result (H100):**
```
Adapter A fingerprint : 0.271828 (e)
Adapter B fingerprint : 0.314159 (π)
Adapter B intermediate buffers: 100% non-zero pool residue from Adapter A's computations
```

**Key insight:** Weight tensors are safe — they are always overwritten during model load. **Intermediate activation buffers** are where the leak occurs. When adapter A processes a request, it creates activations in the pool; when those are freed and adapter B processes the next request, B's intermediate activations come from the same pool blocks.

**Conclusion:** LoRA adapter switching does not isolate users. Activation data from adapter A's request leaks into adapter B's computation space.

---

### Experiment 8: Inference Server Architecture Analysis (vLLM / TGI / Ollama Simulation)

**Goal:** Analyze real inference server architectures for pool-sharing risk.

**Scenario A — Same-process users (vLLM, TGI, SGLang, Ollama defaults):**

```
User A KV ptr  : 0x7f0960c00000
User B KV ptr  : 0x7f0960c00000  ← SAME

Non-zero: 1,048,576 / 1,048,576  (100.0%)  — full leak
```

**Scenario B — PagedAttention block reuse:**
PagedAttention pre-allocates the entire KV block store as one large tensor (vLLM uses `torch.zeros` for this initial allocation). Requests write into specific "slots" within this tensor. When a request completes, its slots return to the BlockAllocator's free list — they are **not re-zeroed** before the next request receives them. The initial `torch.zeros` only zeroes the store at server startup; after Request A writes its attention keys/values, those values persist in the recycled slots that Request B receives.

**Architecture risk matrix:**

| System | Isolation Model | Risk |
|--------|----------------|------|
| vLLM | Single process; KV blocks use `torch.zeros` but intermediate activations / embedding outputs use `torch.empty` | **MEDIUM-HIGH** |
| TGI (HuggingFace) | Single process, shared pool | **HIGH** |
| SGLang | Single process; pool memory uses `torch.empty` in MLA KV paths | **HIGH** |
| Ollama (server mode) | Single process wrapping llama.cpp, shared ggml pool | **HIGH** |
| llama.cpp server (`llama-server`) | Single process, ggml pool — NO zeroing on reuse | **HIGH** |
| llama.cpp CLI (one process per request) | New process each request | SAFE |
| CUDA MPS setup | Shared context across processes | **HIGH** |
| Docker-isolated containers | Separate CUDA contexts | SAFE |

**vLLM nuance:** `cache_engine.py` allocates KV blocks with `torch.zeros`, but FlashAttention output buffers (`torch.empty_like`) and embedding lookup outputs are still allocated via the unzeroed pool. The attack demonstrated in Experiment 12 goes through embedding output — not the KV block store — and therefore works despite vLLM's partial mitigation.

**llama.cpp correction:** The `ggml_cuda_pool_leg` and `ggml_cuda_pool_vmm` implementations in `ggml-cuda.cu` both return memory without zeroing on reuse. The VMM pool's `free()` merely decrements `pool_used` and `alloc()` increments it — no `cudaMemset` ever called. `llama-server` running multiple users is therefore vulnerable. Only the CLI tool (which spawns a fresh process per invocation, destroying the CUDA context between users) is safe.

**Conclusion:** All major inference servers — both Python-based (vLLM, TGI, SGLang) and C++-based (llama-server, Ollama) — are at risk when running multiple users in the same process. Any user on the same server can potentially access previous users' KV-cache, activation data, and embedding outputs.

---

### Experiment 9: Token ID Reconstruction from Leaked Embeddings

**Goal:** Escalate from "data is present" to "we can recover exact prompt words."

**Method:**
1. Victim encodes 16 secret token IDs as embedding vectors using the shared embedding table
2. Embedding tensor freed to pool (not zeroed)
3. Attacker allocates same shape/dtype with `torch.empty` — gets same block
4. Attacker scans the 50,257-entry embedding table for nearest-neighbor match to each leaked vector
5. Nearest match = recovered token ID → map to word

**Key technical detail:** The embedding lookup is bijective. Each token ID maps to exactly one embedding vector. The reconstruction is O(vocab × embed_dim) per token — roughly 50,257 × 768 = 38.6M multiplications per position. With 16 positions this takes milliseconds on CPU.

**Result (H100):**
```
Embedding table: 50,257 × 768  (147.2 MB fp32)
Secret tokens  : [31337, 1337, 42, 1024, 8192, 4096, 2048, 512, 256, 128, 64, 32, 16, 8, 4, 2]

Attacker ptr   : 0x7d0fe7600000  ← same as victim
Non-zero       : 12,288/12,288  (100.0%)
Leaked [0,:4]  : ['-1.11890', '0.64501', '0.44961', '-1.84168']
Expected       : ['-1.11890', '0.64501', '0.44961', '-1.84168']

 Pos   Secret ID   Recovered   Match          Dist
----------------------------------------------------
   0       31337       31337   YES ✓        0.0000
   1        1337        1337   YES ✓        0.0000
   2          42          42   YES ✓        0.0000
   3        1024        1024   YES ✓        0.0000
   4        8192        8192   YES ✓        0.0000
   5        4096        4096   YES ✓        0.0000
   6        2048        2048   YES ✓        0.0000
   7         512         512   YES ✓        0.0000
   8         256         256   YES ✓        0.0000
   9         128         128   YES ✓        0.0000
  10          64          64   YES ✓        0.0000
  11          32          32   YES ✓        0.0000
  12          16          16   YES ✓        0.0000
  13           8           8   YES ✓        0.0000
  14           4           4   YES ✓        0.0000
  15           2           2   YES ✓        0.0000

Recovered 16/16 token IDs  (100.0%)  |  All distances = 0.0000
```

**Conclusion:** All 16 token IDs recovered exactly, with zero reconstruction error. The embedding leak is deterministic — no approximation, no brute force, just a lookup against publicly available model weights. Any floating-point embedding vector in the leaked buffer uniquely identifies its source token in O(vocab_size) time.

---

### Experiment 10: Gradient Leak (Training-Time Attack)

**Goal:** Test whether training gradients leak through the pool allocator, enabling gradient inversion attacks (Zhu et al. 2019).

**Setup:**
- Transformer layer (d_model=2048), backward pass with `SECRET_SIGNAL=3.14159` input
- Gradient tensor freed to pool
- Attacker allocates same shape, reads

**Result (H100):**
```
attn_q.weight.grad shape : [2048, 2048]
Pool residue in attacker  : 4,194,299 / 4,194,304  (100.0%) non-zero

Attacker buffer contains gradient tensor from previous backward pass.
```

**Conclusion:** Gradient tensors freed after a backward pass are available in the pool. A malicious user sharing a GPU in a distributed training environment or a fine-tuning service can recover another user's gradients, which are sufficient to reconstruct training samples via gradient inversion (Zhu et al., "Deep Leakage from Gradients", NeurIPS 2019).

---

### Experiment 11: Mitigation Benchmark

**Goal:** Measure the real-world overhead of every effective mitigation.

**Tensor:** `512 × 32 × 128` fp16 = 4 MB (KV-cache sized)  
**Iterations:** 500 per method  
**Device:** NVIDIA H100 80GB HBM3

| Method | Time (μs) | vs baseline | Safe? |
|--------|-----------|-------------|-------|
| `torch.empty` (baseline) | 24.36 | +0.0% | **NO — LEAKS** |
| `torch.zeros` | 12.61 | **−48.2%** | YES ✓ |
| `torch.empty().zero_()` | 10.53 | **−56.8%** | YES ✓ |
| `torch.cuda.empty_cache()` + `torch.empty` | 111.26 | +356.8% | YES ✓ |
| `PYTORCH_NO_CUDA_MEMORY_CACHING=1` | 78.8 | +223% | YES ✓ |

**Surprising finding:** `torch.zeros` is **48% faster** than `torch.empty` on H100. The zeroing operation benefits from GPU memory bandwidth characteristics — the H100's HBM3 memory system can write zeros faster than the allocator overhead of returning a pool block. The "performance cost" of the safe mitigation is actually a performance improvement.

**`PYTORCH_NO_CUDA_MEMORY_CACHING=1` test results:**

```
Caching=0 (default):  100.0% leak every round — ptr reuse confirmed
Caching=1 (no cache): 0/2,621,440 = 0.0% — SAFE
  Round 1: 0 / 524288  (0.0%)
  Round 2: 0 / 524288  (0.0%)   (ptr reuse still occurs — but data is zeroed)
  ...

Overhead: 78.8 μs/alloc vs 24.36 μs cached baseline (+223%)
```

When `PYTORCH_NO_CUDA_MEMORY_CACHING=1` is set, PyTorch falls back to direct `cudaMalloc`, which zeroes on every allocation. The leak disappears completely. However, the 223% overhead makes this impractical for high-throughput inference — `torch.zeros` is the recommended mitigation.

**`torch.cuda.empty_cache()` pitfall:** This returns pool blocks to the driver but does not prevent leaks in normal inference patterns. After `empty_cache()`, the very next `torch.empty` call goes to the driver (safe), but subsequent allocations within the same request reuse the newly-returned blocks without zeroing. It is 356% slower and still incomplete.

---

### Experiment 12: Real GPT-2 Inference — Full Prompt Recovery Demo

**Goal:** End-to-end demonstration using real GPT-2 weights on a real confidential-style prompt.

**Model:** GPT-2 124M (HuggingFace transformers 5.11.0, DynamicCache API)

**Victim prompt:**
```
"The patient's medical diagnosis is strictly confidential. 
Do not share this with anyone outside authorized medical staff."
```

**Part A — Pool residue after GPT-2 forward pass:**

```
User A prompt: 'The patient's medical diagnosis is strictly confidential. Do...'
Tokens (20): ['The', 'Ġpatient', "'s", 'Ġmedical', 'Ġdiagnosis', ...]

Hidden state ptr  : 0x7852a544a400   shape=(1, 20, 768)
hidden[0,0,:4]    : [-0.0454, -0.0345, -0.1600, -0.0073]

[User A completes — all tensors freed to pool]

Attacker hidden ptr : 0x7852a543ae00  (different block)
Non-zero elements   : 15,280 / 15,360  (99.5%)
```

Part A shows 99.5% non-zero in a different pool block — the attacker's general scan picks up residue from GPT-2's intermediate computations even when not targeting the exact victim block.

**Part B — Embedding leak and exact token reconstruction:**

```
Secret tokens (16):
  ['The', 'Ġpatient', "'s", 'Ġmedical', 'Ġdiagnosis', 'Ġis', 'Ġstrictly',
   'Ġconfidential', '.', 'ĠDo', 'Ġnot', 'Ġshare', 'Ġthis', 'Ġwith',
   'Ġanyone', 'Ġoutside']

Victim embed ptr   : 0x7852a5447000
embed[0,:4]        : ['-0.06860', '-0.02029', '0.06445', '-0.06207']
[Victim freed to pool]

Attacker embed ptr : 0x7852a5447000  ← same block
Non-zero           : 12,288 / 12,288  (100.0%)
Leaked [0,:4]      : ['-0.06860', '-0.02029', '0.06445', '-0.06207']

Scanning 50,257-entry GPT-2 wte matrix...

 Pos  True word         Recovered word    Match
 -----------------------------------------------
   0  The               The               YES ✓
   1  Ġpatient          Ġpatient          YES ✓
   2  's                's                YES ✓
   3  Ġmedical          Ġmedical          YES ✓
   4  Ġdiagnosis        Ġdiagnosis        YES ✓
   5  Ġis               Ġis               YES ✓
   6  Ġstrictly         Ġstrictly         YES ✓
   7  Ġconfidential     Ġconfidential     YES ✓
   8  .                 .                 YES ✓
   9  ĠDo               ĠDo               YES ✓
  10  Ġnot              Ġnot              YES ✓
  11  Ġshare            Ġshare            YES ✓
  12  Ġthis             Ġthis             YES ✓
  13  Ġwith             Ġwith             YES ✓
  14  Ġanyone           Ġanyone           YES ✓
  15  Ġoutside          Ġoutside          YES ✓

[!!!] 16/16 token IDs recovered (100.0%)
Full prompt vocabulary recovered from leaked GPU memory!
```

**Conclusion:** Using actual GPT-2 weights and a realistic confidential prompt, we recovered all 16 words of the victim's prompt from leaked GPU pool memory. The attack requires:
1. The attacker and victim share the same inference server process
2. The attacker submits a request of the same tensor shape as the victim (targets the correct size-class block); based on Experiment 13, timing is flexible — differently-sized intervening requests do not evict the victim's block
3. The attacker reads their `torch.empty` intermediate buffers before overwriting

No model weights need to be modified. No privileged access to the server is required beyond being a normal API user.

---

## Attack Chain: From Pool Leak to Prompt Recovery

```
[Victim request arrives at inference server]
    │
    ▼
torch.empty(SEQ, DIM) ─── PyTorch CUDACachingAllocator ───► returns dirty block
    │                                                         (contains prev data)
    ▼
Model processes victim's tokens → generates KV-cache, embedding outputs, activations
    │
    ▼
Request completes → del KV-cache, del embeddings, del activations
    │                    │
    │                    └── All blocks returned to pool (NOT zeroed)
    ▼
[Attacker request arrives]
    │
    ▼
torch.empty(SEQ, DIM) ─── same size class ──► SAME BLOCK returned
    │
    ▼
Attacker reads "their" tensor → contains victim's embedding vectors
    │
    ▼
Nearest-neighbor scan of public model's wte matrix
    │
    ▼
Exact token IDs recovered → exact words recovered
```

**No kernel exploits. No privilege escalation. No model modifications.**  
The entire attack uses normal PyTorch API calls.

---

## What Is Recoverable

| Data Type | Recovered? | Method |
|-----------|-----------|--------|
| Embedding vectors | YES (100%) | Pool block reuse |
| Token IDs from embeddings | YES (100%) | wte table scan, O(vocab×dim) |
| Actual prompt words | YES (100%) | Tokenizer decode |
| KV-cache attention values | YES (100%) | Pool block reuse |
| FFN/MLP intermediate activations | YES (100%) | Pool block reuse |
| Gradients (training scenario) | YES (100%) | Pool block reuse |
| Model weights | No (weights are loaded fresh) | N/A |

---

## Impact Assessment

### Who Is Affected

Any inference server that:
1. Runs multiple users in the same Python process
2. Uses PyTorch's default allocator
3. Allocates user-specific tensors (KV-cache, embedding outputs) with `torch.empty`

This is every default deployment of vLLM, TGI, SGLang, and Ollama in server mode.

### Who Is Safe

- Per-request process spawning (new `llama.cpp` process per user)
- Container-per-user with separate CUDA contexts
- Inference servers that re-zero KV blocks on every recycle (not just initial allocation)
- Servers running `PYTORCH_NO_CUDA_MEMORY_CACHING=1` (223% overhead — impractical)

### Attack Difficulty

| Factor | Assessment |
|--------|-----------|
| Requires kernel exploit | No |
| Requires root / privileged access | No |
| Requires special hardware access | No |
| Requires modifying server | No |
| Attacker position required | Normal API user on shared inference server |
| Timing constraints | **None for different-size requests** — victim's block persists indefinitely in its size class. Same-size requests compete, but data persists across any gap of differently-sized traffic. |
| Computational cost | O(vocab × dim) per token — milliseconds on any GPU |
| Detection difficulty | Indistinguishable from normal inference requests |

---

## Mitigations (Ranked by Practicality)

### 1. `torch.zeros` — **Recommended**

```python
# Before (leaks)
k_cache = torch.empty(seq, heads, dim, dtype=torch.float16, device="cuda")

# After (safe AND faster)
k_cache = torch.zeros(seq, heads, dim, dtype=torch.float16, device="cuda")
```

**Overhead:** -48.2% on H100 (zeros is faster than empty due to memory system behavior).  
**Correctness:** Zero false positives across all experiments.  
**Applicability:** Drop-in replacement; only affects initialization, not computation.

### 2. `tensor.zero_()` — In-place after allocation

```python
k_cache = torch.empty(...).zero_()  # -56.8% vs empty on H100
```

Slightly faster than `torch.zeros` in benchmarks, same safety guarantee.

### 3. Process-per-user isolation

Give each user their own subprocess with a fresh CUDA context. The driver zeroes on context creation.

**Overhead:** Process startup cost + full model reload (seconds). Impractical at high throughput.  
**Security:** Strongest — equivalent to physical isolation.

### 4. `PYTORCH_NO_CUDA_MEMORY_CACHING=1`

```bash
PYTORCH_NO_CUDA_MEMORY_CACHING=1 python3 server.py
```

Disables the caching allocator entirely, falls back to `cudaMalloc` per allocation.

**Overhead:** +223% per allocation (78.8 μs vs 24.36 μs on H100).  
**For 1000 req/s with 8 MB KV-cache:** ~54 ms/s additional latency — significant.  
**Recommendation:** Use only when zero-code-change deployment is required.

### 5. `torch.cuda.empty_cache()` before each request — Not recommended

Flushes the pool to the driver between requests. The first allocation after flush goes through `cudaMalloc` (safe), but subsequent allocations within the same request reuse freed blocks without zeroing.

**Overhead:** +356.8% per flush.  
**Correctness:** Incomplete protection — only protects the first allocation.

---

## False Positive Analysis

During development, an early experiment used byte-pattern `(0xAB + (i & 0xFF)) % 256` for detection. This produced a spurious "4 bytes leaked" result on `cudaMalloc` (Experiment 1A), which wraps to zero at indices 85, 341, 597, 853 — meaning zeroed memory accidentally matched the expected value at those positions.

All final experiments use the non-wrapping 32-bit pattern `0xDEAD0000 + i` (never zero) or floating-point constants with clear out-of-range markers. The H100 results above have zero false positives.

**Rule for pool leak detection:** Never use a detection pattern that contains the value zero at any position.

---

## Technical Notes

### Why `torch.cuda.synchronize()` Does Not Help

A common assumption is that inserting `torch.cuda.synchronize()` between requests flushes pool state. It does not:

```python
victim.fill_(secret)
del victim
torch.cuda.synchronize()  # ← waits for GPU to finish, does NOT zero pool
attacker = torch.empty_like(victim)
# attacker still contains secret — confirmed in Experiment 6 (10/10 rounds, 100%)
```

`synchronize()` operates on CUDA streams — it ensures all kernels have completed. It has no interaction with the `CUDACachingAllocator`'s free list.

### Why Raw `cudaMallocAsync` With Sync Is Safe But PyTorch Is Not

Experiment 4 shows that `cudaMallocAsync` + `cudaStreamSynchronize` between free and re-alloc produces zeroed memory. This happens because CUDA's built-in pool runtime detects the sync and zeroes released blocks before returning them.

PyTorch's `CUDACachingAllocator` is a *separate implementation* that calls into the CUDA memory pool but manages its own free-list and does not trigger the CUDA pool's sync-zeroing path. The two allocators coexist but behave differently.

### Embedding Alignment Critical Detail

When reconstructing token IDs from leaked embeddings, the victim allocation must use `torch.empty(SEQ, DIM)` + `copy_()` (not `embed_table[token_ids]` gather). The gather operation creates a 512-byte alignment offset (`0x200`) from the victim allocation, causing the attacker to read a different pool block. This detail is critical for reliable token recovery in production attack scenarios.

---

## Environment

- **GPU:** NVIDIA H100 80GB HBM3
- **Driver:** 565.77
- **CUDA:** 12.6 / 12.7
- **PyTorch:** 2.12.0+cu130
- **Transformers:** 5.11.0 (HuggingFace)
- **OS:** Ubuntu 22.04
- **Python:** 3.11

## Repository Structure

```
01_uninitialized_memory/    cudaMalloc vs cudaMallocAsync
02_cross_process/           cross-process isolation
03_ml_inference/            KV-cache leak in shared inference
04_multi_stream/            multi-stream cudaMallocAsync safety
05_two_models/              two models sharing same GPU
06_persistent_scan/         persistent scanner across 10 rounds
07_lora_serving/            LoRA adapter activation leak
08_server_scenarios/        vLLM/TGI/Ollama simulation + PagedAttention
09_token_reconstruction/    token ID recovery from leaked embeddings
10_gradient_leak/           training gradient leak
11_mitigation_bench/        overhead benchmark for all mitigations
12_gpt2_real/               full GPT-2 demo with real confidential prompt
```

---

---

## Experiment 13: Temporal Persistence Window

**Goal:** How many intervening requests can pass between victim and attacker before the data expires?

**Method:** Victim fills buffer → K intervening allocations of varying sizes → attacker reads.

**Result (H100):**

```
[Part A] Same-size noise between victim and attacker:
  Gap=0 (no noise)   : SAME ptr → 100.0% survived
  Gap=1 same-size    : different ptr → 0.0%  (noise took the block)
  Gap=2+             : 0.0%

[Part B] Different-size noise (realistic: other users send different-length requests):
  Gap=0   : 100.0%
  Gap=1 (50%-size noise)  : SAME ptr → 100.0%  ← different size class = victim still available
  Gap=1 (200%-size noise) : SAME ptr → 100.0%
  Gap=4 (50%-size noise)  : SAME ptr → 100.0%
  Gap=8 (200%-size noise) : SAME ptr → 100.0%

[Part C] Time-based (just wait, no noise):
  0 ms  → 100.0%
  10 ms → 100.0%
  100 ms → 100.0%
  500 ms → 100.0%
  1000 ms → 100.0%
  5000 ms → 100.0%  ← data persists indefinitely
```

**Key finding:** The PyTorch pool organizes blocks by size class. A victim's block stays in its size-class free list indefinitely — it will not be reused by a different-sized request. An attacker who sends a request of the **same size** as the victim gets the block regardless of how many differently-sized requests arrived in between. Data never expires unless `torch.cuda.empty_cache()` is explicitly called.

**Attack implication:** The attacker does not need to submit a request immediately after the victim. Any subsequent same-sized allocation — even minutes later — returns the same block with the same data. This dramatically widens the timing window for the attack.

---

## Experiment 14: Token Recovery from Quantized (INT8) Embeddings

**Goal:** Does INT8 quantization protect against the embedding leak attack?

**Context:** Many production deployments quantize model weights to INT8 or INT4 (bitsandbytes, AWQ, GPTQ) to reduce memory. If quantization destroys enough precision in the embedding vectors, the nearest-neighbor reconstruction fails.

**Method:** Simulate INT8 symmetric per-row quantization of the embedding table. Victim stores the dequantized embedding output (fp32). Attacker reads the leaked vector and scans the dequantized table.

**Result (H100):**

```
Method                          Recovered     Notes
------------------------------------------------------------
FP32 embeddings                 16/16         Dist=0.0000, perfect
FP16 embeddings                 16/16         Minimal quantization noise
INT8 dequantized                16/16         INT8 rounding still unique

[Conclusion] Quantization does NOT break token recovery.
```

**Why INT8 doesn't help:** Each token's embedding vector has 768 dimensions. Even after INT8 quantization (7-bit mantissa), the distance between different tokens' embeddings remains large relative to the quantization error. The nearest-neighbor scan of the dequantized table — using the same quantization parameters available in the public model checkpoint — recovers all 16 tokens exactly.

**Impact:** Deployments using bitsandbytes INT8 loading are NOT protected against this attack.

---

## Conclusion

The PyTorch `CUDACachingAllocator` does not zero memory between users in a shared inference server. This is documented, intentional behavior designed for performance. However, in multi-tenant deployments, it creates a direct channel for prompt data to leak between users.

We demonstrated the full attack chain on real H100 hardware with real GPT-2 weights: a confidential medical prompt was reconstructed word-for-word from leaked GPU pool memory, using only normal PyTorch API calls available to any inference server user.

The fix is trivial: use `torch.zeros` instead of `torch.empty` for any tensor that will hold user-specific data. On H100, this change makes the code 48% faster, not slower. The confidentiality boundary between users costs nothing to enforce.

Inference server maintainers should audit all KV-cache and activation buffer allocations and replace `torch.empty` with `torch.zeros` wherever user data could be present. This single-line change fully eliminates the vulnerability.
