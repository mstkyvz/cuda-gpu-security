"""
Experiment 41: Autograd Backward Pass Residue

During backward(), PyTorch allocates gradient buffers and intermediate
tensors (needed to compute ∂L/∂W for each layer) in the CachingAllocator.
After backward() + del loss + optimizer.zero_grad(), these buffers return
to the pool WITHOUT zeroing.

These buffers are richer than inference outputs:
  - They encode the loss signal (from labels)
  - They carry per-sample gradient information
  - Gradient inversion attacks (Geiping 2020) can reconstruct training
    images from a single gradient batch

Tests (5 passes, 3 verification methods each):
  A. Output gradient residue      — backward writes ∂L/∂output → pool
  B. Hidden layer gradient residue — ∂L/∂hidden (Linear output)
  C. Input gradient residue        — ∂L/∂x (most downstream)
  D. Exact gradient verification   — compare residue vs torch.autograd.grad
  E. Gradient accumulation         — N steps, pool has N accumulated grads
  F. Full model: which layer first? — scan 8 pool blocks post-backward
"""

import torch, torch.nn as nn, torch.nn.functional as F, gc

DEVICE = torch.device("cuda:1")
SECRET = 3.14159

def separator(t):
    print(f"\n{'='*60}\n  {t}\n{'='*60}")

def verify3(true_t, residue, label=""):
    """3 methods: cosine, mean/std, element-wise"""
    t = true_t.float().view(-1)
    r = residue.float().view(-1)
    cos      = F.cosine_similarity(t.unsqueeze(0), r.unsqueeze(0)).item()
    mean_ok  = abs(r.mean() - t.mean()) < 0.05 * t.abs().mean() + 1e-5
    std_ok   = abs(r.std()  - t.std())  < 0.05 * (t.std() + 1e-8)
    elem_pct = ((r - t).abs() < 1e-4).float().mean().item() * 100
    leak = abs(cos) > 0.9 or (mean_ok and std_ok)
    return cos, mean_ok, std_ok, elem_pct, leak

# ══════════════════════════════════════════════════════════════
# Test A: Output gradient residue
# Forward: out = layer(inp); backward: loss.backward()
# del loss, out → torch.empty(out.shape) → ∂L/∂out in residue?
# ══════════════════════════════════════════════════════════════
def test_a(passes=5):
    separator("Test A: Output gradient (∂L/∂out) residue — 5 passes")
    d = 1024; B = 32
    layer = nn.Linear(d, d, bias=False).to(DEVICE)
    inp   = torch.randn(B, d, device=DEVICE)

    leaks = 0
    for p in range(1, passes+1):
        torch.cuda.empty_cache()

        out  = layer(inp)              # [B, d]
        loss = out.pow(2).mean()       # scalar loss
        loss.backward()

        # Reference: true ∂L/∂out = 2 * out / (B * d)
        with torch.no_grad():
            true_grad_out = 2 * out / (B * d)

        out_shape = out.shape
        del loss, out
        layer.zero_grad()
        torch.cuda.synchronize(DEVICE)

        residue = torch.empty(out_shape, device=DEVICE)
        torch.cuda.synchronize(DEVICE)

        cos, mo, so, ep, leak = verify3(true_grad_out, residue, f"A.p{p}")
        if leak: leaks += 1
        print(f"  Pass {p}: cos={cos:.4f}  mean_ok={mo}  std_ok={so}  "
              f"elem%={ep:.1f}",
              "  [!!!] ∂L/∂out IN POOL" if leak else "  [~]")
        del residue, true_grad_out
        gc.collect()

    del layer, inp
    print(f"  Result: {leaks}/{passes} LEAKED")

# ══════════════════════════════════════════════════════════════
# Test B: Hidden layer gradient residue
# 2-layer MLP. After backward, del intermediate hidden state.
# torch.empty(hidden.shape) → ∂L/∂hidden or hidden activation?
# ══════════════════════════════════════════════════════════════
def test_b(passes=5):
    separator("Test B: Hidden layer activation/gradient residue — 5 passes")
    d = 1024; B = 32

    class MLP2(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(d, d * 2, bias=False)
            self.fc2 = nn.Linear(d * 2, d, bias=False)
        def forward(self, x):
            self.hidden = F.gelu(self.fc1(x))   # save ref
            return self.fc2(self.hidden)

    mdl = MLP2().to(DEVICE)
    inp = torch.randn(B, d, device=DEVICE)

    # Reference: what does hidden look like?
    with torch.no_grad():
        ref_hidden = F.gelu(mdl.fc1(inp)).clone()
    ref_mean = ref_hidden.mean().item()
    ref_std  = ref_hidden.std().item()

    leaks = 0
    for p in range(1, passes+1):
        torch.cuda.empty_cache()

        out  = mdl(inp)
        loss = out.pow(2).mean()
        loss.backward()
        hidden_shape = mdl.hidden.shape   # [B, d*2]
        del loss, out
        mdl.zero_grad()
        torch.cuda.synchronize(DEVICE)

        # Drain: hidden [B, d*2] should be in pool
        residue = torch.empty(hidden_shape, device=DEVICE)
        torch.cuda.synchronize(DEVICE)

        r_mean = residue.mean().item()
        r_std  = residue.std().item()
        cos    = F.cosine_similarity(ref_hidden.view(-1).unsqueeze(0),
                                     residue.view(-1).unsqueeze(0)).item()
        mean_ok = abs(r_mean - ref_mean) < 0.05 * abs(ref_mean) + 1e-4
        std_ok  = abs(r_std  - ref_std)  < 0.05 * ref_std + 1e-4
        leak = abs(cos) > 0.9 or (mean_ok and std_ok)
        if leak: leaks += 1
        print(f"  Pass {p}: cos={cos:.4f}  mean_ok={mean_ok}  std_ok={std_ok}",
              "  [!!!] HIDDEN IN POOL" if leak else "  [~]")
        del residue
        gc.collect()

    del mdl, inp, ref_hidden
    print(f"  Result: {leaks}/{passes} LEAKED")

# ══════════════════════════════════════════════════════════════
# Test C: Input gradient residue (∂L/∂x)
# Compute ∂L/∂inp — this encodes how the loss changes w.r.t. input
# Meaningful in model-stealing and gradient inversion attacks
# ══════════════════════════════════════════════════════════════
def test_c(passes=5):
    separator("Test C: Input gradient (∂L/∂x) residue — 5 passes")
    d = 1024; B = 32
    layer = nn.Linear(d, d, bias=False).to(DEVICE)

    leaks = 0
    for p in range(1, passes+1):
        torch.cuda.empty_cache()

        inp = torch.randn(B, d, device=DEVICE, requires_grad=True)
        out = layer(inp)
        loss = out.pow(2).mean()
        loss.backward()

        # True ∂L/∂inp = (2 * out) @ W.T / (B * d)
        with torch.no_grad():
            true_grad_inp = inp.grad.clone()

        grad_shape = inp.grad.shape
        del loss, out
        inp.grad = None
        layer.zero_grad()
        del inp
        torch.cuda.synchronize(DEVICE)

        residue = torch.empty(grad_shape, device=DEVICE)
        torch.cuda.synchronize(DEVICE)

        cos, mo, so, ep, leak = verify3(true_grad_inp, residue, f"C.p{p}")
        if leak: leaks += 1
        print(f"  Pass {p}: cos={cos:.4f}  mean_ok={mo}  std_ok={so}  elem%={ep:.1f}",
              "  [!!!] ∂L/∂x IN POOL" if leak else "  [~]")
        del residue, true_grad_inp
        gc.collect()

    del layer
    print(f"  Result: {leaks}/{passes} LEAKED")

# ══════════════════════════════════════════════════════════════
# Test D: Exact verification — torch.autograd.grad vs residue
# Compute exact gradient via autograd.grad, compare byte-for-byte
# with pool residue. elem%=100 means perfect recovery.
# ══════════════════════════════════════════════════════════════
def test_d(passes=5):
    separator("Test D: Exact gradient verification (autograd.grad vs residue)")
    d = 512; B = 16
    layer = nn.Linear(d, d, bias=False).to(DEVICE)
    inp = torch.randn(B, d, device=DEVICE)

    leaks = 0
    for p in range(1, passes+1):
        torch.cuda.empty_cache()

        out = layer(inp)   # [B, d]
        loss = out.pow(2).mean()

        # Exact gradient via retain_graph (don't free graph yet)
        exact_grad = torch.autograd.grad(loss, out, retain_graph=True)[0]  # [B, d]
        exact_grad = exact_grad.detach().clone()

        out_shape = out.shape
        loss.backward()
        del loss, out
        layer.zero_grad()
        torch.cuda.synchronize(DEVICE)

        # Pool should have the backward output-gradient buffer
        residue = torch.empty(out_shape, device=DEVICE)
        torch.cuda.synchronize(DEVICE)

        # Exact comparison
        elem_exact = ((residue - exact_grad).abs() < 1e-5).float().mean().item() * 100
        cos = F.cosine_similarity(exact_grad.view(-1).unsqueeze(0),
                                  residue.view(-1).unsqueeze(0)).item()
        leak = abs(cos) > 0.9 or elem_exact > 50
        if leak: leaks += 1
        print(f"  Pass {p}: cos={cos:.4f}  elem_exact%={elem_exact:.1f}",
              "  [!!!] EXACT GRADIENT IN POOL" if leak else "  [~]")
        del residue, exact_grad
        gc.collect()

    del layer, inp
    print(f"  Result: {leaks}/{passes} LEAKED")

# ══════════════════════════════════════════════════════════════
# Test E: Gradient accumulation — N steps
# With accumulation, gradient is summed over N batches before optimizer.
# Each step adds to .grad without clearing pool. After N steps:
# does pool have each step's individual gradient?
# ══════════════════════════════════════════════════════════════
def test_e(passes=5):
    separator("Test E: Gradient accumulation residue (N=4 steps) — 5 passes")
    d = 512; B = 16; ACCUM_STEPS = 4
    layer = nn.Linear(d, d, bias=False).to(DEVICE)
    # Fixed inputs per step — we know exact expected gradient
    inps = [torch.randn(B, d, device=DEVICE) for _ in range(ACCUM_STEPS)]

    leaks = 0
    for p in range(1, passes+1):
        torch.cuda.empty_cache()
        layer.zero_grad()

        last_out = None
        for step, inp in enumerate(inps):
            out = layer(inp)
            loss = out.pow(2).mean() / ACCUM_STEPS
            loss.backward()
            if step == ACCUM_STEPS - 1:
                last_out = out.clone()
                last_out_shape = out.shape
            del loss, out
        torch.cuda.synchronize(DEVICE)
        layer.zero_grad()
        del last_out
        torch.cuda.synchronize(DEVICE)

        # After zero_grad + del last_out, pool has the last step's output buffer
        residue = torch.empty(last_out_shape, device=DEVICE)
        torch.cuda.synchronize(DEVICE)

        # Reference: what is layer(last_inp)?
        with torch.no_grad():
            ref = layer(inps[-1])
        cos = F.cosine_similarity(ref.view(-1).unsqueeze(0),
                                  residue.view(-1).unsqueeze(0)).item()
        mean_ok = abs(residue.mean() - ref.mean()) < 0.05 * ref.abs().mean() + 1e-4
        leak = abs(cos) > 0.9 or mean_ok
        if leak: leaks += 1
        print(f"  Pass {p}: cos={cos:.4f}  mean_ok={mean_ok}",
              "  [!!!] ACCUMULATED OUTPUT IN POOL" if leak else "  [~]")
        del residue, ref
        gc.collect()

    del layer, inps
    print(f"  Result: {leaks}/{passes} LEAKED")

# ══════════════════════════════════════════════════════════════
# Test F: Full model — scan pool after backward, find gradient buffers
# 3-layer MLP, backward, then scan 12 pool blocks to find which
# layer's buffer shows up (cosine similarity with known reference)
# ══════════════════════════════════════════════════════════════
def test_f(passes=3):
    separator("Test F: Full 3-layer MLP — pool scan after backward")
    d = 512; B = 16

    class MLP3(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(d, d * 2, bias=False)   # out: [B, d*2]
            self.fc2 = nn.Linear(d * 2, d, bias=False)   # out: [B, d]
            self.fc3 = nn.Linear(d, d // 2, bias=False)  # out: [B, d//2]
        def forward(self, x):
            h1 = F.gelu(self.fc1(x))    # [B, d*2]
            h2 = F.relu(self.fc2(h1))   # [B, d]
            return self.fc3(h2)          # [B, d//2]

    mdl = MLP3().to(DEVICE)
    inp = torch.randn(B, d, device=DEVICE)

    # Reference buffers
    with torch.no_grad():
        h1_ref = F.gelu(mdl.fc1(inp))
        h2_ref = F.relu(mdl.fc2(h1_ref))
        out_ref = mdl.fc3(h2_ref)

    for p in range(1, passes+1):
        torch.cuda.empty_cache()
        out  = mdl(inp)
        loss = out.pow(2).mean()
        loss.backward()
        del loss, out
        mdl.zero_grad()
        torch.cuda.synchronize(DEVICE)

        # Scan pool: allocate 12 blocks of each possible shape
        found = {}
        for shape, ref, name in [
            ((B, d*2), h1_ref, "h1[B,d*2]"),
            ((B, d),   h2_ref, "h2[B,d]"),
            ((B, d//2), out_ref, "out[B,d//2]"),
        ]:
            best_cos = 0.0
            for _ in range(6):
                block = torch.empty(shape, device=DEVICE)
                cos = F.cosine_similarity(ref.view(-1).unsqueeze(0),
                                          block.view(-1).unsqueeze(0)).item()
                if abs(cos) > abs(best_cos): best_cos = cos
                del block
            found[name] = best_cos

        print(f"  Pass {p}:")
        for name, cos in found.items():
            print(f"    {name}: cos={cos:.4f}",
                  "  [!!!] FOUND" if abs(cos) > 0.9 else "  [~]")
        gc.collect()

    del mdl, inp, h1_ref, h2_ref, out_ref

if __name__ == "__main__":
    print("=== Experiment 41: Autograd Backward Pass Residue ===")
    print(f"    PyTorch {torch.__version__}  GPU: cuda:1")
    print(f"    Verification: cosine_sim, mean/std, elem%, exact gradient match")
    print()
    test_a()
    test_b()
    test_c()
    test_d()
    test_e()
    test_f()
    print("\n=== Done ===")
