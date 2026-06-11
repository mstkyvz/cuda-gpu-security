"""
Experiment 13: Temporal Persistence Window

How many "intervening" allocations can occur between victim free
and attacker allocation before the data is gone from the pool?

This is critical for real-world attack timing:
  - Attacker submits request immediately after victim: certain hit
  - 1 request gap: does data survive?
  - N requests gap: what is the maximum window?

Method:
  Victim fills SECRET into N floats → frees to pool
  K same-sized "noise" allocations + fills with 0.0 → freed
  Attacker allocates N floats → counts how many match SECRET

Also tests with different-sized noise (different size classes)
to show that size-class matching is NOT required for the attack.
"""
import torch
import sys


def run():
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(1)

    N      = 512 * 1024  # 2 MB fp32 — typical activation size
    SECRET = 3.14159

    print("=" * 62)
    print("Temporal Persistence Window Experiment")
    print(f"Device : {torch.cuda.get_device_name(0)}")
    print(f"Buffer : {N*4//1024} KB fp32  |  Secret = {SECRET}")
    print("=" * 62)

    # -------------------------------------------------------
    # Part A: same-size noise (hardest case for attacker)
    # If noise fills the same pool block, data is overwritten
    # -------------------------------------------------------
    print("\n[Part A] Same-size noise allocations between victim and attacker")
    print(f"{'Gap (allocs)':>14}  {'V-ptr':>14}  {'A-ptr':>14}  {'Reuse':>6}  {'Survived%':>10}")
    print("-" * 62)

    for k in [0, 1, 2, 4, 8, 16, 32]:
        victim = torch.empty(N, dtype=torch.float32, device="cuda")
        victim.fill_(SECRET)
        v_ptr = victim.data_ptr()
        del victim
        torch.cuda.synchronize()

        # K same-size noise allocations, each overwritten with 0
        noise = []
        for _ in range(k):
            n = torch.empty(N, dtype=torch.float32, device="cuda")
            n.fill_(0.0)
            noise.append(n)
        del noise
        torch.cuda.synchronize()

        attacker = torch.empty(N, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        a_ptr = attacker.data_ptr()
        a_cpu = attacker.cpu()
        survived = ((a_cpu - SECRET).abs() < 1e-4).sum().item()
        del attacker

        print(f"{k:>14}  0x{v_ptr:012x}  0x{a_ptr:012x}  "
              f"{'YES' if a_ptr == v_ptr else 'no':>6}  "
              f"{100.0*survived/N:>9.1f}%")

    # -------------------------------------------------------
    # Part B: different-size noise (realistic: other users
    # submit requests of different sizes between Alice & Bob)
    # -------------------------------------------------------
    print("\n[Part B] Different-size noise (realistic server traffic)")
    print(f"{'Gap (allocs)':>14}  {'Reuse':>6}  {'Survived%':>10}  {'Note'}")
    print("-" * 58)

    for k, factor in [(0, None), (1, 0.5), (1, 2.0), (4, 0.5), (8, 2.0)]:
        victim = torch.empty(N, dtype=torch.float32, device="cuda")
        victim.fill_(SECRET)
        v_ptr = victim.data_ptr()
        del victim
        torch.cuda.synchronize()

        if k > 0 and factor is not None:
            noise = []
            for _ in range(k):
                noise_n = int(N * factor)
                nn = torch.empty(noise_n, dtype=torch.float32, device="cuda")
                nn.fill_(0.0)
                noise.append(nn)
            del noise
            torch.cuda.synchronize()

        attacker = torch.empty(N, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        a_ptr = attacker.data_ptr()
        a_cpu = attacker.cpu()
        survived = ((a_cpu - SECRET).abs() < 1e-4).sum().item()
        del attacker

        note = "baseline" if k == 0 else f"{k}x {factor:.0%}-size noise"
        print(f"{k:>14}  {'YES' if a_ptr == v_ptr else 'no':>6}  "
              f"{100.0*survived/N:>9.1f}%  {note}")

    # -------------------------------------------------------
    # Part C: time-based — how long does data persist in pool?
    # -------------------------------------------------------
    import time
    print("\n[Part C] Time-based persistence (no noise, just wait)")
    print(f"{'Wait (ms)':>12}  {'Reuse':>6}  {'Survived%':>10}")
    print("-" * 34)

    for wait_ms in [0, 10, 100, 500, 1000, 5000]:
        victim = torch.empty(N, dtype=torch.float32, device="cuda")
        victim.fill_(SECRET)
        v_ptr = victim.data_ptr()
        del victim
        torch.cuda.synchronize()

        if wait_ms > 0:
            time.sleep(wait_ms / 1000.0)

        attacker = torch.empty(N, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        a_ptr = attacker.data_ptr()
        a_cpu = attacker.cpu()
        survived = ((a_cpu - SECRET).abs() < 1e-4).sum().item()
        del attacker

        print(f"{wait_ms:>12}  {'YES' if a_ptr == v_ptr else 'no':>6}  "
              f"{100.0*survived/N:>9.1f}%")

    print()
    print("[Summary] Pool uses free-list (first-fit by size class).")
    print("  - Same-size gap=0: attacker gets victim block (100%)")
    print("  - Same-size gap=1: noise takes the block, attacker gets NEXT free")
    print("  - Different-size noise: victim block stays free → attacker gets it")
    print("  - Time: pool never GCs unless empty_cache() called")
    print("  => Attacker who knows victim's tensor shape can reliably target block")
    print("     even with intervening requests of different sizes.")


if __name__ == "__main__":
    run()
