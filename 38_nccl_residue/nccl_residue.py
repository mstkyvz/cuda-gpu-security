"""
Experiment 38 v2: NCCL Collective Buffer Residue
Fix: torch.cuda.set_device(rank) must be FIRST before init_process_group
"""

import os, gc, torch, torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

def run_test(rank, world_size):
    # Set device FIRST — before anything else including init_process_group
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29522'
    os.environ['NCCL_DEBUG'] = 'WARN'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}")

    def verify3(residue, true_out, label):
        t = true_out.float().view(-1)
        r = residue.float().view(-1)
        mean_t, std_t = t.mean().item(), t.std().item()
        mean_r, std_r = r.mean().item(), r.std().item()
        mean_ok = abs(mean_r - mean_t) < 0.05 * abs(mean_t) + 1e-3
        std_ok  = abs(std_r  - std_t)  < 0.05 * abs(std_t)  + 1e-3
        cos     = F.cosine_similarity(t.unsqueeze(0), r.unsqueeze(0)).item()
        secret_mean = t.mean().item()
        elem_hits = int(((r - secret_mean).abs() < 0.05 * abs(secret_mean) + 1e-2).sum().item())
        elem_pct  = 100.0 * elem_hits / r.numel()
        leak = (mean_ok and std_ok) or abs(cos) > 0.9 or elem_pct > 80
        return mean_ok, std_ok, cos, elem_pct, leak, mean_t, mean_r

    # ── Test A: allreduce ─────────────────────────────────────────
    if rank == 0:
        print("\n=== Test A: allreduce buffer residue (5 passes) ===")
    a_leaks = 0
    N = 256 * 1024  # 1M floats — not too large
    SECRET_A = 3.14159
    for p in range(5):
        torch.cuda.empty_cache()
        x = torch.full((N,), SECRET_A + rank * 0.001, dtype=torch.float32, device=device)
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        dist.barrier()
        true_out = x.clone()
        del x
        torch.cuda.synchronize(device)

        residue = torch.empty(N, dtype=torch.float32, device=device)
        torch.cuda.synchronize(device)

        mo, so, cos, ep, leak, mt, mr = verify3(residue, true_out, f"A.p{p+1}")
        if rank == 0:
            print(f"  Pass {p+1}: mean_match={mo}  std_match={so}  cos={cos:.4f}  "
                  f"elem%={ep:.1f}  mean_true={mt:.4f}  mean_res={mr:.4f}",
                  "  [!!!] LEAK" if leak else "  [~]")
        if leak: a_leaks += 1
        del residue, true_out
        gc.collect()

    # ── Test B: allgather ─────────────────────────────────────────
    if rank == 0:
        print("\n=== Test B: allgather output buffer residue (5 passes) ===")
    b_leaks = 0
    N2 = 128 * 1024
    SECRET_B = 2.71828
    for p in range(5):
        torch.cuda.empty_cache()
        local = torch.full((N2,), SECRET_B + rank, dtype=torch.float32, device=device)
        gathered = [torch.empty(N2, dtype=torch.float32, device=device)
                    for _ in range(world_size)]
        dist.all_gather(gathered, local)
        dist.barrier()
        true_out = torch.cat(gathered).clone()
        del gathered, local
        torch.cuda.synchronize(device)

        residue = torch.empty(N2 * world_size, dtype=torch.float32, device=device)
        torch.cuda.synchronize(device)

        mo, so, cos, ep, leak, mt, mr = verify3(residue, true_out, f"B.p{p+1}")
        if rank == 0:
            print(f"  Pass {p+1}: cos={cos:.4f}  elem%={ep:.1f}  "
                  f"mean_match={mo}  mean_true={mt:.4f}  mean_res={mr:.4f}",
                  "  [!!!] LEAK" if leak else "  [~]")
        if leak: b_leaks += 1
        del residue, true_out
        gc.collect()

    # ── Test C: broadcast ─────────────────────────────────────────
    if rank == 0:
        print("\n=== Test C: broadcast buffer residue (5 passes) ===")
    c_leaks = 0
    N3 = 128 * 1024
    SECRET_C = 1.41421
    for p in range(5):
        torch.cuda.empty_cache()
        if rank == 0:
            x = torch.full((N3,), SECRET_C, dtype=torch.float32, device=device)
        else:
            x = torch.empty(N3, dtype=torch.float32, device=device)
        dist.broadcast(x, src=0)
        dist.barrier()
        true_out = x.clone()
        del x
        torch.cuda.synchronize(device)

        residue = torch.empty(N3, dtype=torch.float32, device=device)
        torch.cuda.synchronize(device)

        mo, so, cos, ep, leak, mt, mr = verify3(residue, true_out, f"C.p{p+1}")
        if rank == 0:
            print(f"  Pass {p+1}: cos={cos:.4f}  elem%={ep:.1f}  mean_true={mt:.4f}  mean_res={mr:.4f}",
                  "  [!!!] LEAK" if leak else "  [~]")
        if leak: c_leaks += 1
        del residue, true_out
        gc.collect()

    # ── Test D: different SECRET values ──────────────────────────
    if rank == 0:
        print("\n=== Test D: allreduce — 3 SECRET values × 4 passes each ===")
    d_results = {}
    N4 = 64 * 1024
    for secret in [3.14159, 100.0, -7.77]:
        hits = 0
        for p in range(4):
            torch.cuda.empty_cache()
            x = torch.full((N4,), secret + rank * 0.001, dtype=torch.float32, device=device)
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            dist.barrier()
            true_out = x.clone()
            del x
            torch.cuda.synchronize(device)
            residue = torch.empty(N4, dtype=torch.float32, device=device)
            torch.cuda.synchronize(device)
            _, _, cos, ep, leak, mt, mr = verify3(residue, true_out, "D")
            if leak: hits += 1
            del residue, true_out
            gc.collect()
        d_results[secret] = hits
        if rank == 0:
            print(f"  SECRET={secret:8.5f}: {hits}/4 passes LEAKED  (allreduce sum → expected {secret*world_size:.5f})")

    # ── Test E: different tensor shapes ───────────────────────────
    if rank == 0:
        print("\n=== Test E: allreduce — 4 shapes × 3 passes each ===")
    shapes = [(256, 256), (1024, 64), (32, 32, 32), (65536,)]
    for shape in shapes:
        hits = 0
        for p in range(3):
            torch.cuda.empty_cache()
            x = torch.full(shape, 3.14159 + rank * 0.001, dtype=torch.float32, device=device)
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            dist.barrier()
            true_out = x.clone()
            del x
            torch.cuda.synchronize(device)
            residue = torch.empty(shape, dtype=torch.float32, device=device)
            torch.cuda.synchronize(device)
            _, _, cos, ep, leak, mt, mr = verify3(residue, true_out, "E")
            if leak: hits += 1
            del residue, true_out
            gc.collect()
        if rank == 0:
            print(f"  shape={str(shape):20s}: {hits}/3 LEAKED")

    dist.barrier()
    if rank == 0:
        print(f"\n=== NCCL Residue Summary ===")
        print(f"  Test A (allreduce):   {a_leaks}/5")
        print(f"  Test B (allgather):   {b_leaks}/5")
        print(f"  Test C (broadcast):   {c_leaks}/5")
        print(f"  Test D (val sweep):   {d_results}")

    dist.destroy_process_group()

if __name__ == "__main__":
    print("=== Experiment 38: NCCL Collective Buffer Residue ===")
    print("  2x H100, NCCL backend, 5 verification passes per test")
    mp.spawn(run_test, args=(2,), nprocs=2, join=True)
