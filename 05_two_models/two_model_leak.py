"""
Experiment 5: Cross-Model Memory Leak (Two Models, Same Process)

Demonstrates that in a shared GPU inference server that runs multiple
different models in the same process, Model B's allocations can contain
activation data left by Model A.

This simulates a realistic multi-model serving scenario:
  - Model A (e.g., a user's private fine-tune) runs inference
  - Model A is unloaded to free GPU VRAM
  - Model B (a different user's request) is loaded
  - Model B's intermediate activation buffers contain Model A's data

No external dependencies beyond PyTorch — both models are custom MLP-style
networks so this test runs with no internet access required.
"""

import torch
import torch.nn as nn
import sys


class SmallMLP(nn.Module):
    """Simple MLP — stand-in for any neural network."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def run():
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(1)

    device = "cuda"
    print("=" * 62)
    print("Two-Model GPU Memory Leak PoC")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 62)

    # ------------------------------------------------------------------
    # Phase 1: Model A — runs inference with a "secret" batch
    # ------------------------------------------------------------------
    IN_DIM     = 1024
    HIDDEN_DIM = 8192  # large hidden layer to force big allocs
    OUT_DIM    = 512
    BATCH      = 256

    print(f"\n[1] Loading Model A (hidden={HIDDEN_DIM}, batch={BATCH})...")
    model_a = SmallMLP(IN_DIM, HIDDEN_DIM, OUT_DIM).to(device).half()

    # Secret input — recognizable pattern
    secret_batch = torch.full((BATCH, IN_DIM), fill_value=3.14159,
                              dtype=torch.float16, device=device)
    secret_batch += torch.arange(IN_DIM, dtype=torch.float16,
                                 device=device).unsqueeze(0) * 0.001

    print(f"    Secret input[0,:4] = {secret_batch[0, :4].tolist()}")

    with torch.no_grad():
        out_a = model_a(secret_batch)

    torch.cuda.synchronize()
    out_a_ptr    = out_a.data_ptr()
    out_a_sample = out_a[0, :4].cpu().tolist()
    hidden_size_bytes = BATCH * HIDDEN_DIM * 2  # fp16

    print(f"    Model A output ptr   : 0x{out_a_ptr:x}")
    print(f"    Model A output[0,:4] : {out_a_sample}")

    # Capture a reference to the hidden-layer output before freeing
    # (we'll compare against what Model B finds in its buffers)
    fc1_out = model_a.fc1(secret_batch)
    fc1_ptr  = fc1_out.data_ptr()
    fc1_vals = fc1_out[0, :4].cpu().tolist()
    print(f"    Model A hidden ptr   : 0x{fc1_ptr:x}  (size {hidden_size_bytes/1024:.0f} KB)")
    print(f"    Model A hidden[0,:4] : {fc1_vals}")

    # ------------------------------------------------------------------
    # Phase 2: Unload Model A — simulates model being swapped out
    # ------------------------------------------------------------------
    print("\n[2] Unloading Model A (del — returned to pool, NOT zeroed)...")
    del out_a, fc1_out, secret_batch, model_a
    torch.cuda.synchronize()
    print("    All Model A tensors deleted.")

    # ------------------------------------------------------------------
    # Phase 3: Model B loads and allocates intermediate buffers
    # ------------------------------------------------------------------
    print("\n[3] Loading Model B (different user's model, same dimensions)...")
    model_b = SmallMLP(IN_DIM, HIDDEN_DIM, OUT_DIM).to(device).half()

    # Allocate the same-sized intermediate buffer using torch.empty
    # This simulates the hidden activation buffer Model B would allocate
    leaked_hidden = torch.empty(BATCH, HIDDEN_DIM, dtype=torch.float16,
                                device=device)
    torch.cuda.synchronize()

    leaked_ptr = leaked_hidden.data_ptr()
    print(f"    Model B hidden ptr : 0x{leaked_ptr:x}")
    print(f"    Same as Model A    : {leaked_ptr == fc1_ptr}")

    # Read the buffer contents
    leaked_cpu = leaked_hidden.cpu().float()
    nonzero    = (leaked_cpu.abs() > 1e-4).sum().item()
    total      = leaked_cpu.numel()

    print(f"\n[4] Model B 'empty' hidden buffer contents:")
    print(f"    Total elements    : {total}")
    print(f"    Non-zero elements : {nonzero} ({100.0*nonzero/total:.1f}%)")
    print(f"    Buffer[0,:4]      : {leaked_cpu[0, :4].tolist()}")
    print(f"    Model A had       : {fc1_vals}")

    if leaked_ptr == fc1_ptr:
        # Exact same address — direct comparison
        match_a = [(abs(leaked_cpu[0, i].item() - fc1_vals[i]) < 0.1)
                   for i in range(4)]
        print(f"\n[!!!] SAME ADDRESS — Model B buffer = Model A hidden activations.")
        print(f"      Value match [0,:4]: {match_a}")
    elif nonzero > total * 0.5:
        print(f"\n[~] Different address but {100.0*nonzero/total:.1f}% of buffer is non-zero.")
        print(f"    Residual data from pool (prior operations) is visible.")
    else:
        print(f"\n[=] Allocator returned clean block for this size class.")

    # ------------------------------------------------------------------
    # Phase 5: Verify mitigation
    # ------------------------------------------------------------------
    print("\n[5] Mitigation — torch.zeros:")
    safe = torch.zeros(BATCH, HIDDEN_DIM, dtype=torch.float16, device=device)
    safe_nonzero = (safe.cpu().abs() > 1e-4).sum().item()
    print(f"    Non-zero elements : {safe_nonzero} / {safe.numel()} "
          f"({'SAFE' if safe_nonzero == 0 else 'UNSAFE'})")

    del leaked_hidden, safe, model_b
    print("\n[Done]")


if __name__ == "__main__":
    run()
