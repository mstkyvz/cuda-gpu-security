"""
Experiment 35: LLM Token Reconstruction via Activation Buffer Residue (v2)

Key insight from v1: GPT-2 forward creates 13 hidden_states of same shape
[1,S,768] — pool is LIFO, so torch.empty gets the LAST freed (final layer),
not the embedding. Fix: isolate each step individually.

Tests:
  A. Embedding step isolation:
     wte(input_ids) alone → del → torch.empty → nearest-neighbour reconstruct
  B. Logit step isolation:
     lm_head(random_hidden) alone → del → torch.empty → compare values
  C. Full forward scan:
     Run full forward → del → allocate ALL expected shapes → scan each for
     embedding-like vectors → find which block has SECRET tokens
  D. Sequential contamination:
     empty_cache() → Request A full forward (del all) → immediately torch.empty
     every shape that forward created → which ones have SECRET patterns?
  E. Prompt length detection:
     Run prompts of varying lengths, then torch.empty(1, S, 768) for each S —
     does residue cos-sim with wte prove which S was used?

All runs: GPU 1, GPT-2 small, SECRET="my password is 12345"
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gc

DEVICE  = torch.device("cuda:1")
SECRET  = "my password is 12345"
TOL     = 0.01

def load_model():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    mdl = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()
    return tok, mdl

def encode(tok, text, device=DEVICE):
    return tok(text, return_tensors="pt").to(device)

def separator(title):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print('='*62)

def nearest_token(vec: torch.Tensor, wte: torch.Tensor) -> int:
    """Cosine nearest-neighbour lookup: vec [H] → token_id"""
    sims = F.normalize(vec.unsqueeze(0), dim=-1) @ F.normalize(wte, dim=-1).T
    return sims.argmax().item()

def reconstruct_from_embedding(emb: torch.Tensor, wte: torch.Tensor, tok):
    """emb: [1, S, H] → list of token strings"""
    ids = [nearest_token(emb[0, i], wte) for i in range(emb.shape[1])]
    return tok.convert_ids_to_tokens(ids), ids

# ════════════════════════════════════════════════════════════════
# Test A: Embedding step isolation
# ONLY run wte(input_ids) — one tensor, one shape, one del → re-alloc
# ════════════════════════════════════════════════════════════════
def test_a(tok, mdl, passes=5):
    separator("Test A: Embedding step isolation → token reconstruction")
    wte    = mdl.transformer.wte.weight.detach()   # [50257, 768]
    ids    = encode(tok, SECRET)
    S      = ids["input_ids"].shape[1]
    true_t = tok.convert_ids_to_tokens(ids["input_ids"][0].tolist())
    print(f"  TRUE tokens: {true_t}")

    for p in range(1, passes+1):
        # ONE operation: embed input_ids
        with torch.no_grad():
            emb = mdl.transformer.wte(ids["input_ids"])   # [1, S, 768]
        torch.cuda.synchronize(DEVICE)
        emb_shape = emb.shape
        del emb
        torch.cuda.synchronize(DEVICE)

        # Immediately re-alloc same shape
        residue = torch.empty(emb_shape, device=DEVICE)
        torch.cuda.synchronize(DEVICE)

        recon, _ = reconstruct_from_embedding(residue, wte, tok)
        correct  = sum(a == b for a, b in zip(true_t, recon))
        pct      = 100 * correct / S
        # Also check raw value match
        hits = int(((residue - 0.0).abs() > 0.1).float().mean().item() * 100)

        print(f"  Pass {p}: shape={tuple(emb_shape)}  "
              f"reconstruct={correct}/{S} ({pct:.0f}%)  nonzero={hits}%")
        if pct > 60:
            print(f"    [!!!] RECON: {recon}")
        del residue
    del ids

# ════════════════════════════════════════════════════════════════
# Test B: Logit step isolation
# lm_head(zeros) → del → re-alloc → compare std and values
# Then lm_head(SECRET embedding) → del → re-alloc → compare
# ════════════════════════════════════════════════════════════════
def test_b(tok, mdl, passes=5):
    separator("Test B: Logit step isolation → logit buffer residue")
    ids = encode(tok, SECRET)
    S   = ids["input_ids"].shape[1]
    V   = mdl.config.vocab_size  # 50257

    # Get true embeddings to use as input to lm_head
    with torch.no_grad():
        hidden = mdl.transformer.wte(ids["input_ids"])  # [1, S, 768]
        hidden = hidden.unsqueeze(0) if hidden.dim() == 2 else hidden

    print(f"  Input: \"{SECRET}\"  S={S}  logit_shape=[1,{S},{V}]")

    for p in range(1, passes+1):
        with torch.no_grad():
            # Run lm_head on the secret hidden state → logits [1, S, 50257]
            logits_true = mdl.lm_head(hidden)
        true_preds = logits_true[0].argmax(dim=-1).tolist()
        true_std   = logits_true.std().item()
        logit_shape = logits_true.shape
        del logits_true
        torch.cuda.synchronize(DEVICE)

        residue = torch.empty(logit_shape, device=DEVICE)
        torch.cuda.synchronize(DEVICE)

        resid_std   = residue.std().item()
        recon_preds = residue[0].argmax(dim=-1).tolist()
        correct     = sum(a == b for a, b in zip(true_preds, recon_preds))
        pct         = 100 * correct / S

        print(f"  Pass {p}: true_std={true_std:.3f}  resid_std={resid_std:.3f}  "
              f"pred_match={correct}/{S} ({pct:.0f}%)",
              "  [!!!] LOGIT LEAK" if pct == 100 else
              "  [!!!] PARTIAL"    if pct > 60  else "  [~]")
        del residue
    del ids, hidden

# ════════════════════════════════════════════════════════════════
# Test C: Full forward pass — drain pool, scan all blocks
# After full forward + del: allocate ALL shapes the forward created.
# Check each block: does any contain embedding-like vectors for SECRET?
# ════════════════════════════════════════════════════════════════
def test_c(tok, mdl, passes=3):
    separator("Test C: Full forward → drain all pool blocks → scan for SECRET")
    wte   = mdl.transformer.wte.weight.detach()
    ids   = encode(tok, SECRET)
    S     = ids["input_ids"].shape[1]
    true_t = tok.convert_ids_to_tokens(ids["input_ids"][0].tolist())
    H, V  = mdl.config.n_embd, mdl.config.vocab_size
    nh    = mdl.config.n_head
    hd    = H // nh

    # Shapes created during GPT-2 forward pass (S=5, hidden=768):
    forward_shapes = {
        "hidden_[1,S,H]"  : (1, S, H),        # 13 of these (embed + 12 layers)
        "mlp_[1,S,4H]"    : (1, S, 4 * H),    # 12 MLP intermediates
        "attn_qkv_[1,S,H]": (1, S, H),        # (already counted above)
        "logits_[1,S,V]"  : (1, S, V),        # 1 logit tensor
    }

    for p in range(1, passes+1):
        torch.cuda.empty_cache()   # start clean

        # Run full forward
        with torch.no_grad():
            out = mdl(**ids, output_hidden_states=True)
            logit_std_true = out.logits.std().item()
            del out
        torch.cuda.synchronize(DEVICE)

        # Drain: allocate 20 blocks of shape [1, S, H] (covers all hidden + more)
        candidates = []
        for _ in range(20):
            candidates.append(torch.empty(1, S, H, device=DEVICE))
        torch.cuda.synchronize(DEVICE)

        # Check each candidate for SECRET embedding content
        best_correct = 0
        best_idx     = -1
        best_recon   = []
        for i, c in enumerate(candidates):
            recon, _ = reconstruct_from_embedding(c, wte, tok)
            correct  = sum(a == b for a, b in zip(true_t, recon))
            if correct > best_correct:
                best_correct = correct
                best_idx     = i
                best_recon   = recon

        pct = 100 * best_correct / S
        print(f"  Pass {p}: best block idx={best_idx}  "
              f"reconstruct={best_correct}/{S} ({pct:.0f}%)")
        if best_correct > 0:
            print(f"    TRUE : {true_t}")
            print(f"    RECON: {best_recon}")
            if pct > 60:
                print(f"    [!!!] SECRET TOKENS FOUND IN POOL BLOCK {best_idx}")
        for c in candidates:
            del c
        gc.collect()
    del ids

# ════════════════════════════════════════════════════════════════
# Test D: Embedding residue — direct value comparison
# Fill embedding manually with KNOWN tensor, del, re-alloc, compare exactly
# Baseline sanity check that the mechanism works for this exact operation
# ════════════════════════════════════════════════════════════════
def test_d(tok, mdl, passes=5):
    separator("Test D: Manual embedding residue — exact value verification")
    wte  = mdl.transformer.wte.weight.detach()
    ids  = encode(tok, SECRET)
    S    = ids["input_ids"].shape[1]
    H    = mdl.config.n_embd
    true_t = tok.convert_ids_to_tokens(ids["input_ids"][0].tolist())

    # Get true embedding values once
    with torch.no_grad():
        true_emb = mdl.transformer.wte(ids["input_ids"]).clone()  # [1, S, H]

    for p in range(1, passes+1):
        # Allocate tensor, fill with TRUE embedding values, delete
        x = torch.empty(1, S, H, device=DEVICE)
        x.copy_(true_emb)
        torch.cuda.synchronize(DEVICE)
        del x
        torch.cuda.synchronize(DEVICE)

        # Re-alloc: should get the same block back
        residue = torch.empty(1, S, H, device=DEVICE)
        torch.cuda.synchronize(DEVICE)

        # Exact value match
        exact_match = torch.allclose(residue, true_emb, atol=1e-4)
        recon, _    = reconstruct_from_embedding(residue, wte, tok)
        correct     = sum(a == b for a, b in zip(true_t, recon))
        pct         = 100 * correct / S

        print(f"  Pass {p}: exact_allclose={exact_match}  "
              f"token_reconstruct={correct}/{S} ({pct:.0f}%)",
              "  [!!!] PERFECT LEAK" if exact_match else
              "  [!!!] TOKEN LEAK"   if pct > 60   else "  [~]")
        if exact_match or pct > 60:
            print(f"    TRUE : {true_t}")
            print(f"    RECON: {recon}")
        del residue

    del ids, true_emb

# ════════════════════════════════════════════════════════════════
# Test E: Sequential request — does Request B's torch.empty
# return Request A's embedding or logit buffer?
# ════════════════════════════════════════════════════════════════
def test_e(tok, mdl, passes=5):
    separator("Test E: Sequential requests A→B — B observes A's buffers")
    wte   = mdl.transformer.wte.weight.detach()
    H, V  = mdl.config.n_embd, mdl.config.vocab_size

    prompts_a = [
        "my password is 12345",
        "the api key is secret",
        "account number 99887766",
        "pin code is 4321 ok",
        "credit card 5500 0000",
    ]
    prompt_b = "hello world today is"   # benign request, same length

    for p in range(1, passes+1):
        pa = prompts_a[p - 1]
        ids_a = encode(tok, pa)
        ids_b = encode(tok, prompt_b)
        Sa = ids_a["input_ids"].shape[1]
        Sb = ids_b["input_ids"].shape[1]
        true_t_a = tok.convert_ids_to_tokens(ids_a["input_ids"][0].tolist())

        torch.cuda.empty_cache()

        # Request A: embed ONLY → del
        with torch.no_grad():
            emb_a = mdl.transformer.wte(ids_a["input_ids"])  # [1, Sa, H]
        torch.cuda.synchronize(DEVICE)
        del emb_a
        torch.cuda.synchronize(DEVICE)

        # Request B: embed → this should reuse A's block if same Sa==Sb
        with torch.no_grad():
            emb_b = mdl.transformer.wte(ids_b["input_ids"])  # [1, Sb, H]
        torch.cuda.synchronize(DEVICE)

        # Allocate one MORE tensor of the embedding shape
        # If B's wte() consumed A's pool block, this gets B's block
        # If B got a fresh block (different Sa/Sb), A's block is still in pool
        residue = torch.empty(1, Sa, H, device=DEVICE)
        torch.cuda.synchronize(DEVICE)

        recon, _ = reconstruct_from_embedding(residue, wte, tok)
        correct  = sum(a == b for a, b in zip(true_t_a, recon))
        pct      = 100 * correct / Sa

        print(f"  Pass {p}: A=\"{pa[:20]}\" Sa={Sa} Sb={Sb}  "
              f"reconstruct={correct}/{Sa} ({pct:.0f}%)",
              "  [!!!] A's PROMPT LEAKED to B's residue" if pct > 60 else "  [~]")
        del emb_b, residue, ids_a, ids_b

# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== Experiment 35: LLM Token Reconstruction via Activation Residue (v2) ===")
    print("Loading GPT-2 small ...")
    tok, mdl = load_model()
    cfg = mdl.config
    print(f"  GPT-2: vocab={cfg.vocab_size}  hidden={cfg.n_embd}  "
          f"layers={cfg.n_layer}  heads={cfg.n_head}")
    print(f'  SECRET = "{SECRET}"\n')

    test_a(tok, mdl)
    test_b(tok, mdl)
    test_c(tok, mdl)
    test_d(tok, mdl)
    test_e(tok, mdl)

    print("\n=== Summary ===")
    print("A: wte(input_ids) alone → del → re-alloc → reconstruct")
    print("B: lm_head(hidden) alone → del → re-alloc → logit match")
    print("C: full forward → drain 20 blocks → scan best match")
    print("D: manual copy → del → re-alloc → exact value verification")
    print("E: sequential A→B — B's allocation exposes A's embedding")
    print("\n[Done]")
