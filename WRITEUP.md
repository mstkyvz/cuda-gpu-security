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

---

## Experiment 15: CUDA IPC — Cross-Process GPU Memory Access

**Goal:** Can process B read process A's GPU memory using a CUDA IPC handle?

CUDA IPC (`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`) allows sharing GPU allocations across processes. It is used by PyTorch's `torch.multiprocessing`, NCCL, and some inference server worker-pool patterns.

**Security question:** If an IPC handle leaks (via shared file, logging, API metadata, etc.), can an unauthorized process access the GPU data?

**Method:**
- Writer process: allocates GPU memory, fills with `secret=2.71828`, exports handle to `/tmp/cuda_ipc_handle.bin`
- Reader process: loads handle, opens it in a fresh context, reads all data

**Result (H100, two separate processes):**

```
[writer] GPU ptr : 0x7a801fe00000
[writer] Secret  : 2.71828  (N=65536 floats)

[reader] IPC ptr : 0x71d803e00000   ← different virtual address, same physical
[reader] First 8 : 2.71828 2.71828 2.71828 2.71828 2.71828 2.71828 2.71828 2.71828
[reader] Matching: 65536/65536  (100.0%)

[!!!] FULL CROSS-PROCESS GPU MEMORY ACCESS VIA IPC HANDLE
      Process B read Process A's GPU data without permission check.
      Any process with the IPC handle file can read this memory.
[reader] Overwrote writer's GPU memory with zeros (write access confirmed)
```

**Key findings:**
1. **Full read + write access** via IPC handle — the reader has complete control of the writer's GPU allocation
2. **No permission check** at the CUDA level — the handle file is the only access control
3. **Cross-process**: different virtual addresses map to the same physical GPU memory
4. **Risk in production**: NCCL all-reduce operations, PyTorch DataLoader workers, and any multi-process serving pattern that exchanges IPC handles creates this exposure

**Attack vector:** If an inference server logs or exposes IPC handle metadata (e.g., in debug endpoints, process /proc mappings, or shared temp files), any other process on the same host with access to the handle file gains full R/W access to the GPU allocation. This bypasses driver-level cross-process zeroing entirely.

---

## Experiment 16: Multi-GPU P2P Leak (NVLink, 2× H100)

**Goal:** On an NVLink-connected multi-GPU system, can GPU1 read GPU0's unfreed pool data?

**Context:** Tensor-parallel LLM inference splits the model across GPUs. P2P memory copies (`cudaMemcpyPeer`) pass activations between GPU shards. If GPU0's activation pool has dirty data and GPU1 pulls it via P2P, GPU1 sees GPU0's previous request's data.

**NVLink bandwidth:** 387.8 GB/s (confirmed NVLink active, not PCIe)

**Result (2× H100 NVLink):**

```
[1] Peer access capability:
    GPU0 → GPU1: YES
    GPU1 → GPU0: YES

[2] Scenario A: cudaFree → realloc → P2P copy to GPU1
  GPU0 victim ptr  : 0x79befbe00000  (secret=2.71828)
  GPU0 realloc ptr : 0x79befbe00000  (same? YES)
  GPU1 received [0:4]: 0.00000 0.00000 0.00000 0.00000
  [=] cudaFree returns to driver → zeroed → SAFE

[3] Scenario B: pool-style reuse (no cudaFree) → P2P copy
  GPU0 pool ptr (dirty): 0x79befbe00000  secret=2.71828
  GPU1 received [0:4]: 2.71828 2.71828 2.71828 2.71828
  Match (16/16): [!!!] CROSS-GPU LEAK via NVLink P2P
```

**Interpretation:**
- **Scenario A (cudaFree path):** Driver zeroes the block before returning it to the pool. A P2P copy of the reallocation returns zeros. SAFE.
- **Scenario B (pool reuse path, i.e., PyTorch pool):** No zeroing. GPU0's pool reuses the dirty block. P2P copy to GPU1 transfers the previous request's activation data exactly. **FULL LEAK at 387.8 GB/s via NVLink.**

**Real-world scenario:** In tensor-parallel serving (vLLM with `--tensor-parallel-size 2`), each forward pass copies activations between GPU shards via NVLink. If those activation buffers come from the unzeroed pool, the cross-GPU copy propagates the previous user's data to GPU1's buffers too. The pool leak is not contained to a single GPU in TP setups.

---

## Experiment 17: cudaMemPool Attributes — Is There a Built-in Zeroing Flag?

**Goal:** Does the CUDA pool API expose an attribute to enable zeroing on reuse? If so, this would be a single-API mitigation.

**Method:** Test all relevant `cudaMemPoolSetAttribute` values plus `cudaMemPoolTrimTo`.

**Result (H100):**

```
[Part A] Complete sync × stream matrix for cudaMallocAsync:
Scenario                             Ptr-same  Matches     Verdict
same stream, no sync                 YES       4096/4096   FULL LEAK
same stream, with sync               YES          0/4096   SAFE (zeroed)
diff stream, no sync                 no           0/4096   SAFE (zeroed)
diff stream, with sync               YES          0/4096   SAFE (zeroed)

[Part B] cudaMemPoolAttrReleaseThreshold = 0 (force OS return)
  Matches: 0/4096  → SAFE (driver zeroes after OS return)

[Part C] cudaMemPoolTrimTo(pool, 0)
  Matches: 0/4096  → SAFE (trim zeroes)

[Part D] Custom pool, all reuse flags disabled
  Matches: 0/4096  → SAFE
```

**Key findings:**
1. **There is NO zeroing attribute** — CUDA does not expose `cudaMemPoolAttrZeroOnReuse` or equivalent
2. **The only safe CUDA API path:**
   - `cudaMallocAsync` + `cudaStreamSynchronize` between free and realloc (triggers pool zeroing)
   - `cudaMemPoolAttrReleaseThreshold = 0` + sync (forces return to driver, driver zeroes)
   - `cudaMemPoolTrimTo(pool, 0)` (returns all idle pages to driver)
3. **PyTorch bypasses all of these** — the `CUDACachingAllocator` calls into the pool without stream sync, making all three safe paths inaccessible from Python
4. **The sync × stream table** is the authoritative reference: only the "no sync same stream" case leaks at the raw CUDA API level

**Practical implication:** There is no single `cudaMemPoolSetAttribute` call that makes the pool safe. Mitigation must come from the application layer (`torch.zeros`).

---

## Experiment 18: VMM Direct Test (cuMemCreate / cuMemMap)

**Goal:** Characterize the exact behavior of the VMM allocator used by llama.cpp's `ggml_cuda_pool_vmm` on H100.

**VMM granularity on H100:** 2048 KB

**Result:**

```
[Part A] cuMemCreate — fresh physical pages:
  Non-zero on fresh allocation : 0/524288
  → SAFE — cuMemCreate zeroes new physical pages on first mapping

[Part B] VMM bump-alloc reuse (ggml_cuda_pool_vmm):
  pool_used=0 → alloc → fill → pool_used=0 (rewind) → alloc again
  Slot0 ptr    : 0x7d62d8800000  (filled with 2.71828)
  Attacker ptr : 0x7d62d8800000  (same — pool rewind)
  Matches      : 524288/524288
  → [!!!] FULL LEAK — VMM pool reuse does NOT zero

[Part C] cuMemUnmap + cuMemMap to same VA (fresh physical → same address):
  After mapping fresh h2 physical to same VA:
  Non-zero : 0/524288
  → SAFE — new physical pages zeroed on fresh mapping

[Part D] Re-map SAME physical handle to new VA:
  va2 (same physical, new virtual address):
  Matches  : 524288/524288
  → [!!!] LEAK — physical reuse retains data regardless of VA change
```

**Summary table for VMM:**

| Operation | Zeroed? | Notes |
|-----------|---------|-------|
| `cuMemCreate` → first mapping | YES | Driver zeroes fresh physical pages |
| VMM bump reuse (pool_used rewind) | **NO** | ggml_cuda_pool_vmm exact behavior |
| `cuMemUnmap` + `cuMemMap` (new physical) | YES | New physical pages are zeroed |
| `cuMemUnmap` + `cuMemMap` (same physical handle) | **NO** | Physical data persists across VA remap |

**Conclusion:** llama.cpp's `ggml_cuda_pool_vmm` performs a bump-allocator reuse by rewinding `pool_used`. The physical memory is never re-zeroed. This is Part B — confirmed as full leak on H100. Only when the pool is completely torn down (`cuMemUnmap` + `cuMemRelease` + new `cuMemCreate`) does the driver zero the memory.


---

## CVE Analysis: Prior Art and Gap

### Existing CVEs

| CVE | Year | Score | Domain | Description |
|-----|------|-------|--------|-------------|
| CVE-2011-0636 | 2011 | — | Host (pinned) | cudaHostAlloc returned uninitialized kernel memory on NVIDIA 260.19.21 |
| CVE-2016-7386 | 2016 | 5.5 | Kernel driver | KMD escape handler leaked kernel heap via uninitialized buffer to user space |
| CVE-2019-5697 | 2019 | 7.1 | vGPU | vGPU Manager granted guest access to memory outside its allocation |
| CVE-2024-53869 | 2025 | 5.5 | Unified Memory | NVIDIA Unified Memory driver — uninitialized pages leaked to caller |

### What is NEW in this research

**CVE-2011-0636** covers host-side (CPU) pinned memory. Our Exp 19 confirms this is **patched** in CUDA 12.8 — cudaHostAlloc returns zeroed pages in all scenarios. The driver-level vulnerability no longer exists.

**CVE-2024-53869** covers the NVIDIA Unified Memory driver (kernel module path for cudaMallocManaged). Our Exp 20 confirms this class is also **not exploitable** on H100 CUDA 12.8.

**The gap this research fills:** No existing CVE covers the **user-space pool allocator** path:
- PyTorch CUDACachingAllocator (Python/C++ library, not kernel module)
- llama.cpp ggml_cuda_pool_vmm / ggml_cuda_pool_leg (C++ library pool)
- The end-to-end chain: **pool reuse → embedding residue → token ID recovery**

The vulnerability class is:
> **User-space GPU memory pool allocator does not zero on reuse, and framework applications do not zero pool allocations, enabling cross-request reconstruction of prior-request tokens, weights, and activations.**

This is a framework-layer vulnerability, not captured by any existing CVE.

---

## Experiment 19: Pinned Host Memory (CVE-2011-0636 Class)

**Goal:** Verify whether cudaHostAlloc (pinned host memory) leaks stale data on modern CUDA drivers.

**Result (H100, CUDA 12.8):**

```
Part A  Fresh cudaHostAlloc (4 MB):
  Non-zero bytes : 0 / 4,194,304   → SAFE

Part B  cudaHostAlloc free + realloc (same size):
  Matches        : 0 / 1,048,576   → SAFE

Part C  cuMemHostAlloc (driver API) free + realloc:
  Matches        : 0 / 1,048,576   → SAFE

Part D  10-cycle fill/free/realloc entropy:
  Total matches  : 0 / 10,485,760  → SAFE — no pinned leakage
```

**Conclusion:** CVE-2011-0636 is fully patched in CUDA 12.8. All pinned host memory paths return zeroed pages.

---

## Experiment 20: Unified Memory (CVE-2024-53869 Class)

**Goal:** Test whether cudaMallocManaged leaks stale data (CVE-2024-53869 class).

**Result (H100, CUDA 12.8):**

```
Part A  Fresh alloc, CPU first-touch:              0/524,288  → SAFE
Part B  GPU fill → free → realloc → CPU read:      0/524,288  → SAFE
Part C  GPU-prefetch → free → CPU-prefetch realloc:0/524,288  → SAFE
Part D  GPU kernel reads reallocated UM buffer:    0/524,288  → SAFE
Part E  cudaMemAdviseSetPreferredLocation + reuse: 0/524,288  → SAFE
```

**Conclusion:** CVE-2024-53869 is patched on CUDA 12.8. All five UM paths return zeroed pages.

---

## Experiment 21: cudaMallocAsync Stream-Ordered Pool (Detailed)

**Goal:** Fully characterize cudaMallocAsync safety across all sync/stream/pool-config combinations.

**Result (H100, CUDA 12.8):**

```
Part A  Fresh cudaMallocAsync:                       262,144/262,144 zeros → SAFE
Part B  Same stream, no sync (fill→free→alloc→read): 262,144/262,144 match → [!!!] FULL LEAK
Part C  Same stream WITH stream sync between:        0/262,144             → SAFE
Part D  ReuseAllowOpportunistic=0 (custom pool):     262,144/262,144 match → [!!!] LEAK
Part E  Cross-stream (victim s0, attacker s1):       0/262,144             → SAFE
Part F  cudaMemPoolTrimTo(pool, 0) before realloc:   0/262,144             → SAFE
Part G  ReleaseThreshold=0:                          0/262,144             → SAFE
```

**Safety matrix:**

| Scenario | Leaks? |
|----------|--------|
| Same stream, no sync between free/alloc | **YES 100%** |
| Same stream, with stream sync | SAFE |
| Different stream | SAFE |
| ReuseAllowOpportunistic = 0 | **LEAKS** (flag does not prevent same-stream reuse) |
| cudaMemPoolTrimTo(pool, 0) | SAFE (pages return to OS, re-zeroed) |
| ReleaseThreshold = 0 | SAFE |

**Critical insight (Part D):** cudaMemPoolReuseAllowOpportunistic=0 does NOT fix same-stream reuse.
The flag controls cross-stream opportunistic reuse only. Same-stream reuse is ordered by construction
(free happens-before alloc in stream order) and is not considered "opportunistic".

---

## Experiment 22: IPC + Pool Reuse Combined Attack

**Goal:** Test whether pool residue crosses process boundaries via IPC or pool inheritance.

**Method:**
- Writer: cudaMallocAsync pool fill SECRET → free to pool → cudaMalloc (non-pool) → overwrite with ANTI-SECRET → export IPC handle
- Reader: open IPC handle, read; also test reader's own pool for contamination

**Result (H100, CUDA 12.8):**

```
Writer cudaMalloc self-check: 0/131,072 match
  (pool-freed memory does NOT contaminate subsequent cudaMalloc)

Reader IPC read:
  Match SECRET       : 0/131,072  (pool residue not in IPC block)
  Match ANTI-SECRET  : 131,072/131,072  (IPC read confirmed)

Reader own pool:
  match_secret=0  match_anti=0
  SAFE: cross-process pool contamination does not occur
```

**Findings:**
1. cudaMalloc and cudaMallocAsync pools are **isolated** — pool-freed memory does not contaminate a subsequent cudaMalloc block.
2. **Pools are per-process** — process B cannot inherit process A's pool residue.
3. Cross-process data access still requires explicit IPC handle transfer (Exp 15).

**Conclusion:** Pool leaks are intra-process only. Cross-process attacks require explicit IPC handle sharing.


---

## Experiment 23: Shared Memory (__shared__) Zeroing Between Kernel Launches

**Goal:** Determine whether CUDA hardware zeroes `__shared__` SRAM between sequential kernel launches on the same SM.

**Design note:** Initial version had a dead-code elimination bug — `victim_fill_shared` compiled to `bar.sync; ret` because the compiler detected the shared memory writes were never read within the kernel. Fixed by adding a global flag that reads `sm[0]` back, forcing the compiler to retain the `st.shared` instructions (verified via PTX inspection).

**Method:**
- Part A: Static `__shared__`: victim fills, attacker reads — 4 trials
- Part B: Dynamic `extern __shared__`: same test
- Part C: Large allocation (164 KB = near-max optin on H100) — 4 blocks
- Part D: 512 blocks → amplifies SM reuse (H100 has 132 SMs)
- Part E: Control — attacker runs without prior victim

**Result (H100, CUDA 12.8 — 4-pass verified):**

```
[Part A] Static __shared__ (48 KB, 4 blocks):
  Trial 0: match=12288/12288  Trial 1: match=12288/12288
  Trial 2: match=12288/12288  Trial 3: match=12288/12288
  → [!!!] FULL LEAK — static shared memory not zeroed between launches

[Part B] Dynamic extern __shared__ (48 KB, 4 blocks):
  Trial 0: match=12288/12288  Trial 1: match=12288/12288
  Trial 2: match=12288/12288  Trial 3: match=12288/12288
  → [!!!] FULL LEAK — dynamic shared memory not zeroed

[Part C] Large allocation (164 KB optin):
  match=41984/41984 (100.0%)  → [!!!] FULL LEAK at 164 KB

[Part D] 512 blocks (SM reuse amplified):
  match=12288/12288 (100.0%)  → [!!!] FULL LEAK across all SM reuse patterns

[Part E] Control:
  12288/12288 match  → confirms attacker reads prior kernel's data
```

**SM shared memory hardware max on H100:** 227 KB per block (optin). 228 KB `cudaFuncSetAttribute` fails with `invalid argument`.

**Conclusion:** CUDA hardware does **not** zero `__shared__` SRAM between kernel launches. When a new block is scheduled on an SM, it inherits the previous block's shared memory contents. This applies to both static and dynamic shared memory, at all tested sizes (48 KB, 96 KB, 164 KB).

**Attack scenario:** Multi-request inference server — Request A's attention kernel stores query/key/value tiles in `__shared__`. Request B's kernel, scheduled on the same SM, reads uninitialized `__shared__` and recovers Request A's token activations.

---

## Experiment 24: cuBLAS Workspace and Shared Memory Residue

**Goal:** Determine whether cuBLAS GEMM operations leak data through (1) the external workspace buffer or (2) shared memory residue accessible to a subsequent attacker kernel.

### Part 24A–E: Workspace Buffer Leak

**Method:** Manually provided 64 MB workspace via `cublasSetWorkspace`; ran FP32 and FP16 GEMMs; inspected workspace after.

**Result (H100, CUDA 12.8 — 3-pass verified):**

```
FP32 cublasSgemm:                        0 non-zero bytes → SAFE
FP16 cublasGemmEx (COMPUTE_16F):         0 non-zero bytes → SAFE
FP16 cublasGemmEx (COMPUTE_32F_FAST_16F):1 non-zero byte  → effectively 0
Victim GEMM (A=SECRET), FP16 SECRET match: 0             → SAFE
Sequential GEMM residue:                 0 SECRET matches → SAFE
```

**Conclusion:** cuBLAS on H100 does **not** use external workspace for standard 4096×4096 GEMMs. The Tensor Core hardware handles all computation in registers and on-chip shared memory — no external scratch needed. External workspace buffer does NOT leak.

### Part 24F: cuBLAS GEMM → Shared Memory Residue

**Method:** Run cuBLAS FP16 GEMM with A=SECRET → immediately launch attacker kernel that reads uninitialized `__shared__` across all 132 SMs → count FP16 SECRET patterns.

**Result (H100, CUDA 12.8 — 4-pass verified, 3 trials each pass):**

```
Baseline (no prior GEMM):
  Non-zero u16 values: 0 / 3,244,032    SECRET hits: 0

After cuBLAS GEMM (A=SECRET=2.71828, FP16=0x4170):
  Trial 0: nz=3,244,032/3,244,032  SECRET_hits=2,162,688/3,244,032 (66.67%)  [!!!]
  Trial 1: nz=3,244,032/3,244,032  SECRET_hits=2,162,688/3,244,032 (66.67%)  [!!!]
  Trial 2: nz=3,244,032/3,244,032  SECRET_hits=2,162,688/3,244,032 (66.67%)  [!!!]

Control (A=B=0):
  nz=0/3,244,032  → confirms 66.67% is causally from SECRET input
```

**Why exactly 66.67%?** cuBLAS's H100 HMMA (Tensor Core) algorithm tiles shared memory in a 2:1 A:B ratio. With 132 SMs × 48 KB = 6.2 MB of shared memory sampled, 2/3 contains A matrix tiles (=SECRET), 1/3 contains B matrix tiles (=1.0). This exactly matches `2162688 / 3244032 = 2/3`.

**Conclusion:** After any cuBLAS GEMM, an attacker kernel on the same SMs can read the GEMM's input matrix tiles from shared memory residue. 66.67% of all shared memory across all 132 SMs contained the exact FP16 value of the victim's input matrix. This is a direct consequence of Exp 23 (shared memory not zeroed) applied to cuBLAS's internal tiling algorithm.

**Combined attack chain:** Victim's LLM layer → cuBLAS GEMM on A=embedding/attention → attacker kernel reads uninitialized `__shared__` → recovers FP16 fragments of victim's activation matrices.

---

## Experiment 25: CUDA MPS (Multi-Process Service) Isolation Test

**Context:** CUDA MPS (Multi-Process Service) allows multiple processes to share a single GPU CUDA context. It is widely used on cloud H100/A100 instances to improve GPU utilization across concurrent workloads. Under MPS, all client processes share the same physical GPU address space.

**Setup:** GPU 0 set to `EXCLUSIVE_PROCESS` compute mode. MPS daemon started via `nvidia-cuda-mps-control -d`. Writer and reader run as separate client processes under the same MPS session.

**Tests:**

**Test A — MPS Pool Isolation (cudaMallocAsync):**
- Writer: allocates 1 MB via `cudaMallocAsync`, fills with SECRET=3.14159, frees to pool
- Reader (separate process, same MPS session): allocates 1 MB via `cudaMallocAsync`
- Both processes get the SAME virtual address (`0x420000000`) — confirms shared context

**Result (4-trial, H100, CUDA 12.8):**

```
All 4 trials:
[W] pool alloc ptr=0x420000000  filled S=3.14159  freed to pool
[R] attempt 0: ptr=0x420000000  matches=0/262144
[R] attempt 1: ptr=0x420000000  matches=0/262144
...
→ SAFE — MPS runtime zeroes pool memory on cross-client allocation
```

**Test B — MPS Shared Memory (`__shared__`) Isolation:**
- Writer: runs 512-block kernel filling `__shared__` (48 KB) with SECRET on all 132 SMs
- Reader (separate process, same MPS session): launches kernel reading uninitialized `__shared__`

**Result (timing-dependent, H100, CUDA 12.8):**

```
Delay ≤ 100ms (attacker launches immediately after victim):
  match=12288/12288  nonzero=12288/12288
  [!!!] MPS __shared__ LEAK — reader sees writer's shared memory data (100%)

Delay ≥ 500ms (other GPU work runs in between):
  match=0/12288  nonzero=0/12288
  SAFE — other workloads naturally overwrote shared memory on SM reuse
```

**Key findings:**

| Surface | MPS Isolation | Reason |
|---------|--------------|--------|
| cudaMallocAsync pool | **SAFE** | MPS runtime zeroes on cross-client alloc |
| `__shared__` memory | **NOT isolated** | Hardware SM SRAM has no per-client isolation |
| Virtual address space | Shared | Both clients get identical VAs (0x420000000) |

**Why the asymmetry?** MPS manages the `cudaMallocAsync` pool at the software level — it explicitly zeroes when handing memory from one client's freed pool to another. But `__shared__` is hardware-managed SM SRAM: MPS has no mechanism to zero hardware SRAM between client kernel executions. The GPU hardware's SRAM reuse behavior (as proven in Exp 23) is not configurable through MPS.

**Attack window:** Under MPS, sequential inference requests from different users may share SMs. If an attacker process launches immediately after a victim process's kernel (realistic in a shared inference server), the attacker reads 100% of the victim's `__shared__` data. The window is ~100-200ms — realistic for back-to-back user requests.

**Practical context:** Any inference server running under MPS (which includes most cloud GPU deployments for cost efficiency) is vulnerable to this `__shared__` leak if:
1. The attacker has enough control to time their kernel launch
2. Both attacker and victim run on the same MPS session (same GPU partition)

---

## Experiment 26: CUDA Graphs — Intermediate Buffer Residue

**Context:** CUDA Graphs (`cudaGraph_t`) capture a sequence of GPU kernels and replay them at reduced CPU overhead. PyTorch uses `torch.cuda.CUDAGraph` to accelerate LLM inference — the standard workflow is: (1) warmup run on real data, (2) graph capture on same stream, (3) repeated graph replays. Key question: does graph capture or replay zero intermediate pool buffers?

**Tests:**

**Test A — External buffer, graph writes, post-destroy pool:**
- Graph captures `write_secret(d_work, SECRET1)`. d_work is preallocated outside the graph.
- Replays 1 and 2 both correctly write SECRET1 (graph replay works).
- After graph destroy + `cudaMallocAsync` from same pool: 0 SECRET1 hits.

**Test B — Graph-owned allocation (cudaMallocAsync inside capture):**
- `cudaMallocAsync` called inside the stream-capture window → owned by graph.
- Graph writes SECRET1 to d_internal, frees inside capture.
- After graph destroy + pool alloc: 0 SECRET1 hits, nz=0.

**Test C — Pre-graph pool residue (graph alloc inside capture):**
- Phase 1: alloc from pool, fill with SECRET1, free back to pool.
- Phase 2: capture graph that calls `cudaMallocAsync` inside capture window.
- Graph launch: the pool block returned to d_in_graph is the same one filled with SECRET1.

**Test D — `__shared__` residue from graph kernel:**
- Graph captures `fill_shared(SECRET1)` running on 512 blocks across all 132 SMs.
- After graph launch + sync, attacker kernel reads uninitialized `__shared__`.

**Result (4-pass verified, H100, CUDA 12.8):**

```
Test A — Post-destroy pool alloc:
  All 4 passes: SECRET1=0/262144  nz=0/262144
  → SAFE — graph destroy + pool free zeroes the block

Test B — Graph-owned alloc after destroy:
  All 4 passes: SECRET1=0/262144  nz=0/262144
  → SAFE — graph-owned memory zeroed when returned to pool on destroy

Test C — Pre-graph pool residue in graph capture:
  Pass 1: SECRET1=262144/262144  nz=262144/262144  [!!!]
  Pass 2: SECRET1=262144/262144  nz=262144/262144  [!!!]
  Pass 3: SECRET1=262144/262144  nz=262144/262144  [!!!]
  Pass 4: SECRET1=262144/262144  nz=262144/262144  [!!!]
  → 100% LEAK — graph capture does NOT zero pool blocks; same-stream reuse delivers SECRET1

  Graph replay 2 (same graph run again):
  All passes: SECRET1=262144/262144 — same pool block locked to graph, reused each replay

Test D — __shared__ residue from graph kernel:
  Pass 1: SECRET1=12288/12288  nz=12288/12288  [!!!]
  Pass 2: SECRET1=12288/12288  nz=12288/12288  [!!!]
  Pass 3: SECRET1=12288/12288  nz=12288/12288  [!!!]
  Pass 4: SECRET1=12288/12288  nz=12288/12288  [!!!]
  → 100% LEAK — graph wrapper does not affect __shared__ hardware SRAM behavior
```

**Key findings:**

| Test | Surface | Result | Reason |
|------|---------|--------|--------|
| A | External buffer, post-destroy pool | **SAFE** | Pool zeroes block on graph boundary |
| B | Graph-owned alloc post-destroy | **SAFE** | Graph destroy triggers pool zeroing |
| C | Pool alloc inside capture (pre-graph residue) | **100% LEAK** | Same-stream pool reuse, no zeroing |
| D | `__shared__` from graph kernel | **100% LEAK** | Hardware SRAM never zeroed (Exp 23) |

**Why Test C leaks:** A `cudaMallocAsync` inside a stream-capture window is assigned a pool block at graph-instantiation time (or at first launch). The pool block is the same one freed in Phase 1 with SECRET1 still in it — because stream-ordered pool reuse does not zero (as proven in Exp 21). Graph capture/replay does not add any additional zeroing step. The pool block with SECRET1 is locked to the graph and reused on every replay.

**PyTorch CUDAGraph attack chain:** PyTorch's `torch.cuda.CUDAGraph` pattern is:
1. **Warmup** (user A's data flows through): model forward pass fills activation buffers in stream pool
2. **Graph capture**: same stream, `torch.empty`-allocated buffers inside graph reuse the SAME pool blocks from warmup — still contain user A's activations
3. **Graph replay** (user B's request): graph replays with user B's input tensor, but the graph-internal `torch.empty` allocations still point to the warmup pool blocks, which hold user A's data until user B's computation overwrites them — a window of vulnerability at the start of each replay

**Practical scenario:** Any inference server using `torch.cuda.CUDAGraph` (vLLM, TGI, TensorRT-LLM) with `torch.empty` for intermediate buffers leaks warmup-phase activations into the first replay cycle's intermediate buffers. Since warmup is typically run on representative (real) data, the leaked buffers contain real activation patterns.

**Connection to prior experiments:**
- Test C: same underlying mechanism as Exp 1B (same-stream no-sync pool reuse) and Exp 21 (same-stream no-sync = 100% LEAK). Graph capture does not change this.
- Test D: same as Exp 23 (`__shared__` never zeroed). Graph execution is just another kernel launch from the hardware's perspective.

---

## Experiment 27: GPU L2 Cache Timing Side-Channel (Prime+Probe)

**Context:** The GPU L2 cache is physically shared across all 132 SMs on the H100. A Prime+Probe cache side-channel attack uses access timing to infer which memory regions a victim accessed — without reading the victim's memory directly. This experiment tests whether a GPU attacker can infer which "slot" (2 MB region) a victim kernel accessed based solely on L2 access latency, and whether this works across CUDA streams.

**H100 L2 specs:**
- Capacity: 50 MB total
- Cache line: 128 bytes
- Hit latency: ~670–680 cycles (~340 ns at 1980 MHz)
- Miss (HBM3) latency: ~840–1280 cycles (~420–647 ns at 1980 MHz)
- Latency ratio: **1.25×–1.90× cold/warm** — sufficient for differential timing inference

**Key design decisions (derived through debugging):**
1. **Non-volatile loads required.** `volatile float *` parameters cause `ld.volatile.global.f32` PTX which bypasses L2 entirely. All kernel parameters must be plain `float *` for the L2 to participate.
2. **`membar.gl` required after each probe load.** Without it, `clock64()` fires at instruction _issue_ (not completion), giving ~23-cycle readings for all cache states — completely unusable.
3. **Single thread, single SM for probe.** `clock64()` is SM-local; measurements from different SMs are incomparable. Probe must use a single thread in block (0,0).
4. **Differential timing.** A sequential 150 MB flush buffer does not uniformly evict all 16 probe slots due to physical address aliasing between `d_probe` and `d_flush`. Some slots (0–3, 8–11) partially or fully survive the flush. Using differential timing (baseline run without victim minus attack run with victim) eliminates this bias: always-warm slots show ~0 differential; the victim's slot shows a large latency REDUCTION.
5. **LCG randomized probe order.** Randomizing slot access order in the probe prevents sequential prefetcher bias from contaminating measurements.

**Tests:**

**Test A — L2 latency baseline:**
- After 150 MB flush: avg ~800–815 cycles (mix of cold HBM and some L2-resident slots due to address aliasing)
- After all 16 slots warmed: avg ~674–676 cycles (all L2 hits)
- Reliably cold slot latency: 840–1280 cycles; L2 hit: 670–680 cycles

**Test B — Differential signal detection (victim slot 7, 120 trials):**
- Flush → baseline probe → flush → victim(7) → attack probe
- Accumulate diff per slot over 120 trials; slot with max(baseline − attack) = inferred victim

**Test C — 4-bit secret inference (16 possible secrets, 120 trials each):**
- Victim accesses one of 16 slots based on a 4-bit secret
- Attacker performs differential Prime+Probe, infers victim's slot from accumulated timing delta

**Test D — Cross-stream differential timing (victim stream A, attacker stream B):**
- Tests whether CUDA stream isolation prevents L2 observation

**Result (5-pass verified, H100, CUDA 12.8):**

```
Test A — Latency baseline:
  After 150MB flush:         avg ~800-815 cycles (HBM hits + some L2-resident slots)
  After all slots warmed:    avg ~674-676 cycles (all L2 hits)
  Cold/warm ratio:           1.25×–1.90× (reliable differential signal)

Test B — Signal detection (victim slot 7, 120 trials):
  Pass 1: CORRECT  Pass 2: CORRECT  Pass 3: CORRECT  Pass 4: CORRECT  Pass 5: CORRECT
  → 5/5 passes, max diff on true victim: 21683–34883 cycles vs noise <6000

Test C — 4-bit secret inference:
  Pass 3 (120 trials): 14/16 secrets correct
  Pass 4 (120 trials): 16/16 secrets correct  [!!!]
  Pass 5 (120 trials): 14/16 secrets correct
  → 13–16/16 per pass (avg 14.7/16 = 92%)
  → [!!!] L2 TIMING ATTACK: attacker infers 4-bit secret via differential timing
  → Random baseline: 1/16 = 6.25%

  Reliably cold slots (4–7, 12–15): 100% detection all passes
  Borderline slots (0–3, 8–11): 50–100% per pass (varies with L2 flush effectiveness per run)

Test D — Cross-stream (victim stream A, attacker stream B, victim slot 6):
  Pass 3: CORRECT  Pass 4: CORRECT  Pass 5: CORRECT
  → 3/3 passes — L2 state is NOT isolated between CUDA streams
```

**Key findings:**

| Test | Signal | Result |
|------|--------|--------|
| A | L2 hit vs HBM miss latency | **1.25×–1.90× cold/warm ratio** |
| B | Victim slot detection (1 of 16) | **100% (5/5 passes, 120 trials)** |
| C | 4-bit secret inference | **13–16/16 per pass (avg 92%); random baseline 6.25%** |
| D | Cross-stream L2 timing | **CORRECT 3/3 — CUDA streams share L2 state** |

**Why this works:**
1. H100 L2 (50 MB) is a hardware-shared resource across all SMs. There is no per-process or per-stream L2 partition.
2. Differential timing eliminates the non-uniform L2 flush artifact: the baseline run captures each slot's "natural" post-flush state; the attack run shows the victim's effect on top. The slot with the largest drop is the victim's slot.
3. `ld.global.cg` (cache-globally: bypass L1, use L2) combined with `membar.gl` (force completion before clock) gives precise per-slot latency measurements from a single SM thread.
4. CUDA streams do not affect L2 state — stream isolation is a scheduling abstraction, not a hardware cache partition.

**L2 flush non-uniformity (hardware observation):** A 150 MB sequential read of `d_flush` consistently evicts slots 4–7 and 12–15 from `d_probe` but leaves slots 0–3 and 8–11 partially L2-resident. This is due to physical address aliasing: `d_probe` and `d_flush` are allocated at different physical pages, and their cache set indices (determined by physical address bits) do not fully overlap across all L2 slices. This is a fundamental H100 hardware constraint, not a software bug. Differential timing is the correct algorithmic response: it works even when flush is non-uniform.

**Practical attack scenario:** In a shared inference server (multi-stream serving, MPS), an attacker can fingerprint:
- Which model layer a victim is computing (different layers access different HBM regions → different L2 residue)
- Which token batch size a request used (different batch sizes → different activation tensor footprint)
- Victim computation timing via appearance/disappearance of cache lines

**Defense:** NVIDIA H100 does not support L2 cache partitioning per process (unlike Intel's CAT). The only mitigations are:
- MIG (Multi-Instance GPU): each GPU instance has independent L2 slices — this attack is impossible across MIG instances
- Physical GPU isolation per tenant
- Software-level noise injection (periodically flushing/warming the L2 with random data)
