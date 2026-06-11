"""
Experiment 6: Persistent GPU Pool Scanner (PyTorch CUDACachingAllocator)

Simulates a malicious co-tenant on a shared GPU inference server.
In each round, a "victim user" fills a tensor with data and deletes it.
An "attacker user" immediately allocates a tensor of the same size and
reads whatever is in it — all victim data visible in every round.

Key finding:
  PyTorch's CUDACachingAllocator does NOT zero memory between requests.
  torch.cuda.synchronize() (which every server does between requests) does
  NOT help. The attacker always gets the victim's full tensor data.

  This is different from raw cudaMallocAsync, which zeroes memory when a
  cudaStreamSynchronize() is called between free and realloc. PyTorch
  uses its own block-based free list that bypasses this driver behavior.
"""

import torch
import sys


def run():
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(1)

    print("=" * 62)
    print("Persistent Pool Scanner — PyTorch CUDACachingAllocator")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 62)

    N      = 512 * 1024   # 2 MB (fp32)
    ROUNDS = 10
    SECRETS = [2.71828, 3.14159, 1.41421, 1.61803, 0.57721,
               2.30259, 1.73205, 0.36788, 4.66920, 6.02214]

    print(f"\nBuffer size per round : {N * 4 // 1024} KB")
    print(f"Rounds                : {ROUNDS}")
    print(f"Sync between rounds   : YES (torch.cuda.synchronize)")
    print()

    total_match = 0
    total_elems = 0

    print(f"{'Round':<6}  {'Secret':<10}  {'Ptr-reuse':<10}  {'Leaked/Total':<16}  {'Match%'}")
    print("-" * 58)

    for r in range(ROUNDS):
        secret = SECRETS[r % len(SECRETS)]

        # VICTIM: allocate, fill with secret pattern, delete
        victim = torch.empty(N, dtype=torch.float32, device="cuda")
        victim.fill_(secret)                         # all elements = secret
        victim[0] = secret * 1.1                    # unique marker at index 0
        victim[-1] = secret * 9.9                   # unique marker at index -1
        v_ptr = victim.data_ptr()
        del victim
        torch.cuda.synchronize()                     # server waits for GPU

        # ATTACKER: torch.empty — no initialization
        leaked = torch.empty(N, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        a_ptr = leaked.data_ptr()

        leaked_cpu = leaked.cpu()
        matches = (leaked_cpu - secret).abs() < 1e-4
        match_n = matches.sum().item()
        total_match += match_n
        total_elems += N

        reuse = "YES" if a_ptr == v_ptr else "no"
        print(f"{r+1:<6}  {secret:<10.5f}  {reuse:<10}  {match_n:<7}/{N:<8}  {100.0*match_n/N:.1f}%")

        if r == 0:
            vals = leaked_cpu[:4].tolist()
            print(f"         Leaked [0:4]: {[f'{v:.5f}' for v in vals]}")
            print(f"         Expected   : {secret:.5f} (all elements should be {secret:.5f})")
            if a_ptr == v_ptr:
                marker0 = leaked_cpu[0].item()
                marker_end = leaked_cpu[-1].item()
                print(f"         Marker[0]  : {marker0:.5f}  (expected {secret*1.1:.5f})")
                print(f"         Marker[-1] : {marker_end:.5f}  (expected {secret*9.9:.5f})")

        del leaked

    print()
    pct = 100.0 * total_match / total_elems
    print(f"[Summary] Total leaked: {total_match}/{total_elems} ({pct:.1f}%)")

    if pct > 80:
        print()
        print("[!] CONFIRMED LEAK — PyTorch CUDACachingAllocator exposes")
        print("    victim data in every round, even with synchronize().")
        print()
        print("    Real-world impact:")
        print("    - GPU server handles Request A → frees KV-cache to pool")
        print("    - Server syncs GPU (waits for result)")
        print("    - GPU server handles Request B → torch.empty gets same block")
        print("    - Request B's tensor contains Request A's KV-cache / activations")
    elif pct > 0:
        print("[~] Partial exposure — some data leaked")
    else:
        print("[=] No exposure — different size class or fresh allocation")

    # Mitigation check
    print()
    print("[Mitigation] torch.zeros:")
    secret2 = 3.14159
    victim2 = torch.empty(N, dtype=torch.float32, device="cuda")
    victim2.fill_(secret2)
    del victim2
    torch.cuda.synchronize()

    safe = torch.zeros(N, dtype=torch.float32, device="cuda")
    nonzero = (safe.cpu().abs() > 1e-6).sum().item()
    print(f"    Non-zero elements: {nonzero}/{N} "
          f"({'SAFE' if nonzero == 0 else 'UNSAFE'})")
    del safe
    print()
    print("[Done]")


if __name__ == "__main__":
    run()
