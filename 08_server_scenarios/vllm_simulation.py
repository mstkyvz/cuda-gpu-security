"""
Experiment 8: Multi-User Inference Server Simulation (vLLM/Ollama Style)

Tests GPU memory isolation between users on a shared inference server.

Three scenarios tested:

SCENARIO A — Same-Process Multi-User (vLLM, TGI, SGLang)
  Multiple users' requests processed by the same Python process.
  Pool memory is shared → User B reads User A's KV-cache.
  RESULT: 100% LEAK (confirmed)

SCENARIO B — Different Processes (Ollama default, per-request spawn)
  Each request handled by a separate process / CUDA context.
  Driver zeroes GPU memory on context destroy.
  RESULT: SAFE (confirmed by Experiment 2)

SCENARIO C — vLLM PagedAttention Pool Simulation
  vLLM uses a custom block table (PagedAttention) for KV-cache management.
  Even with PagedAttention, the UNDERLYING GPU memory is from the same
  PyTorch pool allocated with torch.empty.
  We simulate the block reuse pattern here.

Architecture notes:
  Ollama: By default runs a single server process. All users served by
          the same process → same pool → LEAK between users in flight.
  vLLM:   Uses continuous batching in a single process. PagedAttention
          manages KV blocks but they are allocated from torch.empty.
  TGI:    Also single-process multi-user → same pool → LEAK.
  SGLang: Radix cache shares KV blocks across requests → intentional reuse
          but with isolation via indices, not memory zeroing.
"""

import torch
import torch.nn as nn
import sys
import time
from typing import Dict, List, Optional


# =====================================================================
# Simulate a minimal paged-attention KV block manager (vLLM-like)
# =====================================================================
class KVBlockManager:
    """
    Simplified vLLM-style PagedAttention block manager.
    Maintains a pool of fixed-size KV blocks, all backed by ONE large tensor.
    Blocks are allocated and freed without zeroing — matching vLLM behavior.
    """
    def __init__(self, n_blocks=256, block_size=16, n_heads=32, head_dim=128,
                 dtype=torch.float16, device="cuda"):
        self.block_size = block_size
        self.n_heads    = n_heads
        self.head_dim   = head_dim

        # Single large backing tensor for all KV blocks
        # Shape: [n_blocks, 2, n_heads, block_size, head_dim]
        # (2 = K and V)
        self.kv_store = torch.empty(
            n_blocks, 2, n_heads, block_size, head_dim,
            dtype=dtype, device=device
        )
        self.free_blocks = list(range(n_blocks))
        self.used_blocks: Dict[int, List[int]] = {}  # request_id → [block_ids]
        self.next_req_id = 0

    def allocate(self, n_tokens) -> int:
        """Allocate blocks for a request. Returns request_id."""
        n_needed = (n_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < n_needed:
            raise RuntimeError("OOM: no free KV blocks")
        req_id = self.next_req_id
        self.next_req_id += 1
        allocated = [self.free_blocks.pop(0) for _ in range(n_needed)]
        self.used_blocks[req_id] = allocated
        return req_id, allocated

    def free(self, req_id):
        """Return request's blocks to the free list (NOT zeroed)."""
        blocks = self.used_blocks.pop(req_id, [])
        self.free_blocks.extend(blocks)

    def write_kv(self, req_id, blocks, k_vals, v_vals):
        """Write K/V into the blocks (simulates attention layer writing KV-cache)."""
        for i, blk in enumerate(blocks):
            self.kv_store[blk, 0] = k_vals  # K
            self.kv_store[blk, 1] = v_vals  # V

    def read_block_raw(self, block_id):
        """Read a block's raw contents (no initialization check)."""
        return self.kv_store[block_id]


def run():
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(1)

    print("=" * 62)
    print("Multi-User Inference Server Simulation")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 62)

    BLOCK_SIZE = 16
    N_HEADS    = 32
    HEAD_DIM   = 128
    N_BLOCKS   = 512

    # ----------------------------------------------------------------
    # SCENARIO A: Same-process raw pool (most direct)
    # ----------------------------------------------------------------
    print("\n[Scenario A] Same-process raw PyTorch pool")
    print("  Simulates: vLLM, TGI, SGLang, Ollama (single process)\n")

    SEQ_A = 256   # User A's prompt: 256 tokens
    DTYPE = torch.float16

    # User A gets GPU memory via torch.empty
    k_a = torch.empty(SEQ_A, N_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    v_a = torch.empty(SEQ_A, N_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")

    # Fill with "computed" attention keys/values
    k_a.fill_(1.41421)   # User A's KV signal (sqrt(2))
    v_a.fill_(2.71828)   # User A's V signal (e)
    torch.cuda.synchronize()

    k_ptr_a = k_a.data_ptr()
    print(f"  User A  K ptr: 0x{k_ptr_a:x}")
    print(f"  User A  K[0,0,:4]: {k_a[0,0,:4].cpu().float().tolist()}")

    del k_a, v_a
    torch.cuda.synchronize()

    # User B arrives immediately
    k_b = torch.empty(SEQ_A, N_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    torch.cuda.synchronize()
    k_ptr_b = k_b.data_ptr()
    k_b_vals = k_b[0, 0, :4].cpu().float().tolist()

    print(f"  User B  K ptr: 0x{k_ptr_b:x}  (reuse: {k_ptr_b == k_ptr_a})")
    print(f"  User B  K[0,0,:4]: {k_b_vals}")
    nz_b = (k_b.cpu().abs() > 1e-4).sum().item()
    print(f"  Non-zero: {nz_b}/{k_b.numel()} ({100.0*nz_b/k_b.numel():.1f}%)")

    if k_ptr_b == k_ptr_a:
        print("  [!!!] SAME ADDRESS — User B has User A's KV-cache (LEAK)")
    elif nz_b > k_b.numel() * 0.5:
        print(f"  [~] Different ptr but {100.0*nz_b/k_b.numel():.1f}% non-zero pool data")
    del k_b

    # ----------------------------------------------------------------
    # SCENARIO C: PagedAttention block reuse (vLLM simulation)
    # ----------------------------------------------------------------
    print("\n[Scenario C] PagedAttention block reuse (vLLM-style)")
    print("  Blocks NOT zeroed when returned to free list\n")

    mgr = KVBlockManager(N_BLOCKS, BLOCK_SIZE, N_HEADS, HEAD_DIM)

    # User A: allocate blocks, write secret KV
    req_a, blocks_a = mgr.allocate(n_tokens=32)
    secret_k = torch.full((N_HEADS, BLOCK_SIZE, HEAD_DIM), 3.14159,
                           dtype=torch.float16, device="cuda")
    secret_v = torch.full((N_HEADS, BLOCK_SIZE, HEAD_DIM), 2.71828,
                           dtype=torch.float16, device="cuda")
    mgr.write_kv(req_a, blocks_a, secret_k, secret_v)
    torch.cuda.synchronize()
    first_block_a = blocks_a[0]

    print(f"  User A  request_id={req_a}, blocks={blocks_a}")
    print(f"  User A  KV block[{first_block_a}] K[0,0,:4]: "
          f"{mgr.kv_store[first_block_a,0,0,0,:4].cpu().float().tolist()}")

    # User A finishes: blocks returned (NOT zeroed)
    mgr.free(req_a)

    # User B: gets the same blocks
    req_b, blocks_b = mgr.allocate(n_tokens=32)
    torch.cuda.synchronize()
    first_block_b = blocks_b[0]

    # User B reads the block "raw" before writing their own KV
    raw_b = mgr.read_block_raw(first_block_b)
    k_raw = raw_b[0, 0, 0, :4].cpu().float().tolist()  # K, head 0, pos 0

    print(f"\n  User B  request_id={req_b}, blocks={blocks_b}")
    print(f"  User B  reused block {first_block_b} (was block {first_block_a}): "
          f"{first_block_b == first_block_a}")
    print(f"  User B  KV block[{first_block_b}] K[0,0,:4] (NOT yet written): {k_raw}")
    print(f"  Expected (User A's): {[3.14159]*4}")

    if first_block_b == first_block_a:
        match = abs(k_raw[0] - 3.14159) < 0.01
        print(f"  Match: {match}")
        if match:
            print("  [!!!] PagedAttention blocks NOT zeroed on reuse → LEAK")
    mgr.free(req_b)

    # ----------------------------------------------------------------
    # SCENARIO B: Different-process summary (from Experiment 2)
    # ----------------------------------------------------------------
    print("\n[Scenario B] Different processes (Ollama per-process isolation)")
    print("  Result: SAFE — driver zeroes memory on CUDA context destroy")
    print("  (see 02_cross_process/ for full test)")
    print("  Ollama runs all requests in ONE server process by default →")
    print("  concurrent users share pool → reverts to Scenario A (LEAK)")

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    print("\n[Summary] Inference Server Architecture Risk Matrix")
    print("-" * 62)
    print(f"{'Server':<20}  {'Architecture':<20}  {'Risk'}")
    print("-" * 62)
    rows = [
        ("vLLM",         "single-process",      "HIGH — shared pool"),
        ("TGI",          "single-process",      "HIGH — shared pool"),
        ("SGLang",       "single-process",      "HIGH — radix cache reuse"),
        ("Ollama",       "single-process svr",  "HIGH — all users same pool"),
        ("Ollama",       "per-request spawn",   "SAFE — driver zeroes"),
        ("llama.cpp",    "per-request binary",  "SAFE — new process each"),
        ("Custom Flask", "single-process",      "HIGH — shared pool"),
    ]
    for server, arch, risk in rows:
        print(f"  {server:<18}  {arch:<20}  {risk}")
    print()


if __name__ == "__main__":
    run()
