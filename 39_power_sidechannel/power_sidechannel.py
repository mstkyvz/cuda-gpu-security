"""
Experiment 39 v2: GPU Power Side-Channel via NVML
Uses GPU 0 (both GPUs now free), smaller tensors to avoid OOM.
4 rounds per test for multi-method verification.
"""

import time, gc, statistics
import torch, torch.nn as nn, torch.nn.functional as F

import pynvml
pynvml.nvmlInit()
GPU_IDX = 0
nvml_h0 = pynvml.nvmlDeviceGetHandleByIndex(0)
nvml_h1 = pynvml.nvmlDeviceGetHandleByIndex(1)

def get_power(gpu=0):
    h = nvml_h0 if gpu == 0 else nvml_h1
    return pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0  # Watts

DEVICE = torch.device("cuda:0")

def sample_power(duration_s=1.0, interval_s=0.05, gpu=0):
    readings = []
    t0 = time.time()
    while time.time() - t0 < duration_s:
        readings.append(get_power(gpu))
        time.sleep(interval_s)
    return readings

def stats(readings):
    m = statistics.mean(readings)
    s = statistics.stdev(readings) if len(readings) > 1 else 0.0
    return m, s

def separator(t):
    print(f"\n{'='*60}\n  {t}\n{'='*60}")

# ══════════════════════════════════════════════════════════════
# Test A: Idle vs heavy matmul — can we see the power jump?
# Verification: 4 rounds, measure Δ and SNR each round
# ══════════════════════════════════════════════════════════════
def test_a(rounds=4):
    separator("Test A: Idle vs Heavy MatMul (4 rounds, SNR verification)")
    d = 4096  # 4096×4096 = 64M params, fits easily in 80GB
    A = torch.randn(d, d, device=DEVICE)
    B_ = torch.randn(d, d, device=DEVICE)

    deltas, snrs = [], []
    for r in range(1, rounds+1):
        torch.cuda.synchronize(DEVICE)
        idle = sample_power(1.0, 0.05)

        bufs = []
        t0 = time.time()
        while time.time() - t0 < 1.2:
            with torch.no_grad():
                bufs.append(torch.mm(A, B_))
            if len(bufs) > 20: del bufs[:-5]
        busy = sample_power(1.0, 0.05)
        del bufs; gc.collect(); torch.cuda.synchronize(DEVICE)

        im, is_ = stats(idle)
        bm, bs  = stats(busy)
        delta = bm - im
        snr   = delta / (is_ + 1e-3)
        deltas.append(delta); snrs.append(snr)
        print(f"  Round {r}: idle={im:.1f}W±{is_:.1f}  matmul={bm:.1f}W±{bs:.1f}  "
              f"Δ={delta:.1f}W  SNR={snr:.1f}",
              "  [!!!] DETECTABLE" if delta > 10 and snr > 3 else "  [~]")

    print(f"\n  Avg Δ={statistics.mean(deltas):.1f}W  Avg SNR={statistics.mean(snrs):.1f}  "
          f"(4-round verified)")
    del A, B_

# ══════════════════════════════════════════════════════════════
# Test B: Compute type fingerprinting
# matmul (compute) vs attention (mixed) vs memset (memory-bound)
# 4 rounds, compare average power per workload
# ══════════════════════════════════════════════════════════════
def test_b(rounds=4):
    separator("Test B: Workload fingerprinting — matmul / attention / memset (4 rounds)")
    d = 2048
    N_MEM = 64 * 1024 * 1024  # 64M floats for memory sweep

    power_by = {'matmul': [], 'attention': [], 'memset': []}

    for r in range(1, rounds+1):
        row = {}

        # matmul
        A = torch.randn(d, d, device=DEVICE)
        B_ = torch.randn(d, d, device=DEVICE)
        bufs = []
        t0 = time.time()
        while time.time() - t0 < 0.6:
            with torch.no_grad(): bufs.append(torch.mm(A, B_))
            if len(bufs) > 10: del bufs[:-3]
        mm_r = sample_power(0.8, 0.04)
        del bufs, A, B_; gc.collect(); torch.cuda.synchronize(DEVICE)
        row['matmul'] = statistics.mean(mm_r)

        # attention
        bufs = []
        t0 = time.time()
        while time.time() - t0 < 0.6:
            q = torch.randn(4, 16, 128, 64, device=DEVICE, dtype=torch.float16)
            k = torch.randn(4, 16, 128, 64, device=DEVICE, dtype=torch.float16)
            v = torch.randn(4, 16, 128, 64, device=DEVICE, dtype=torch.float16)
            with torch.no_grad(): bufs.append(F.scaled_dot_product_attention(q, k, v))
            if len(bufs) > 10: del bufs[:-3]
        attn_r = sample_power(0.8, 0.04)
        del bufs, q, k, v; gc.collect(); torch.cuda.synchronize(DEVICE)
        row['attention'] = statistics.mean(attn_r)

        # memset
        bufs = []
        t0 = time.time()
        while time.time() - t0 < 0.6:
            bufs.append(torch.zeros(N_MEM, device=DEVICE))
            if len(bufs) > 5: del bufs[:-2]
        mem_r = sample_power(0.8, 0.04)
        del bufs; gc.collect(); torch.cuda.synchronize(DEVICE)
        row['memset'] = statistics.mean(mem_r)

        for k_, v_ in row.items():
            power_by[k_].append(v_)

        print(f"  Round {r}: matmul={row['matmul']:.1f}W  "
              f"attn={row['attention']:.1f}W  "
              f"memset={row['memset']:.1f}W  "
              f"Δ(mm-mem)={row['matmul']-row['memset']:.1f}W")

    avgs = {k: statistics.mean(v) for k, v in power_by.items()}
    spread = max(avgs.values()) - min(avgs.values())
    print(f"\n  4-round avgs: {' '.join(f'{k}={v:.1f}W' for k,v in avgs.items())}")
    print(f"  Max spread: {spread:.1f}W",
          "  [!!!] FINGERPRINTING FEASIBLE" if spread > 15 else "  [~] spread too small")

# ══════════════════════════════════════════════════════════════
# Test C: Batch size inference
# B=4, 16, 64, 256 — does power scale with batch?
# 4 rounds, measure linearity of power vs B
# ══════════════════════════════════════════════════════════════
def test_c(rounds=4):
    separator("Test C: Batch size oracle — B=4/16/64/256 (4 rounds)")
    hidden = 2048
    layer = nn.Linear(hidden, hidden * 4, bias=False).to(DEVICE).eval()
    batch_sizes = [4, 16, 64, 256]
    power_by_b = {b: [] for b in batch_sizes}

    for r in range(1, rounds+1):
        row = {}
        for b in batch_sizes:
            inp = torch.randn(b, hidden, device=DEVICE)
            bufs = []
            t0 = time.time()
            while time.time() - t0 < 0.5:
                with torch.no_grad(): bufs.append(layer(inp))
                if len(bufs) > 20: del bufs[:-5]
            rd = sample_power(0.7, 0.05)
            del bufs, inp; gc.collect(); torch.cuda.synchronize(DEVICE)
            m, _ = stats(rd)
            row[b] = m
            power_by_b[b].append(m)
        row_str = "  ".join(f"B={b}:{row[b]:.1f}W" for b in batch_sizes)
        print(f"  Round {r}: {row_str}")

    avgs = {b: statistics.mean(power_by_b[b]) for b in batch_sizes}
    spread = max(avgs.values()) - min(avgs.values())
    print(f"\n  Avgs: {' '.join(f'B={b}:{avgs[b]:.1f}W' for b in batch_sizes)}")
    print(f"  Spread: {spread:.1f}W",
          "  [!!!] BATCH SIZE INFERABLE" if spread > 10 else "  [~]")
    del layer

# ══════════════════════════════════════════════════════════════
# Test D: Temporal detection — catch compute start/stop
# Observer samples power every 50ms. Victim starts compute mid-window.
# Can observer detect the event? 4 rounds.
# ══════════════════════════════════════════════════════════════
def test_d(rounds=4):
    separator("Test D: Temporal detection — victim compute start/stop (4 rounds)")
    d = 4096
    A = torch.randn(d, d, device=DEVICE)
    B_ = torch.randn(d, d, device=DEVICE)

    detected = 0
    for r in range(1, rounds+1):
        # Phase 1: idle
        ph1 = sample_power(0.6, 0.05)

        # Phase 2: victim compute
        bufs = []
        t0 = time.time()
        while time.time() - t0 < 1.0:
            with torch.no_grad(): bufs.append(torch.mm(A, B_))
            if len(bufs) > 10: del bufs[:-3]
        ph2 = sample_power(0.8, 0.05)
        del bufs; gc.collect(); torch.cuda.synchronize(DEVICE)

        # Phase 3: idle again
        ph3 = sample_power(0.6, 0.05)

        m1, s1 = stats(ph1)
        m2, s2 = stats(ph2)
        m3, s3 = stats(ph3)
        det = (m2 - m1 > 10) and (m2 - m3 > 10)
        if det: detected += 1
        print(f"  Round {r}: idle1={m1:.1f}W → compute={m2:.1f}W → idle2={m3:.1f}W  "
              f"rise={m2-m1:.1f}W fall={m2-m3:.1f}W",
              "  [!!!] DETECTED" if det else "  [~]")

    print(f"\n  Detection rate: {detected}/{rounds} rounds")
    del A, B_

# ══════════════════════════════════════════════════════════════
# Test E: Dense (GELU) vs Sparse (ReLU) activation power delta
# Verification: 4 rounds, measure power difference
# ══════════════════════════════════════════════════════════════
def test_e(rounds=4):
    separator("Test E: Dense (GELU) vs Sparse (ReLU) activation power (4 rounds)")
    d = 4096
    B = 32

    dense = nn.Sequential(
        nn.Linear(d, d * 2, bias=False), nn.GELU(), nn.Linear(d * 2, d, bias=False)
    ).to(DEVICE).eval()
    sparse = nn.Sequential(
        nn.Linear(d, d * 2, bias=False), nn.ReLU(), nn.Linear(d * 2, d, bias=False)
    ).to(DEVICE).eval()

    inp = torch.randn(B, d, device=DEVICE)
    deltas = []

    for r in range(1, rounds+1):
        # Dense (GELU)
        bufs = []
        t0 = time.time()
        while time.time() - t0 < 0.7:
            with torch.no_grad(): bufs.append(dense(inp))
            if len(bufs) > 10: del bufs[:-3]
        dr = sample_power(0.8, 0.04)
        del bufs; gc.collect(); torch.cuda.synchronize(DEVICE)

        # Sparse (ReLU)
        bufs = []
        t0 = time.time()
        while time.time() - t0 < 0.7:
            with torch.no_grad(): bufs.append(sparse(inp))
            if len(bufs) > 10: del bufs[:-3]
        sr = sample_power(0.8, 0.04)
        del bufs; gc.collect(); torch.cuda.synchronize(DEVICE)

        dm, ds_ = stats(dr)
        sm, ss  = stats(sr)
        delta = dm - sm
        deltas.append(delta)
        print(f"  Round {r}: gelu={dm:.1f}W±{ds_:.1f}  relu={sm:.1f}W±{ss:.1f}  "
              f"Δ={delta:.1f}W",
              "  [!!!] DISTINGUISHABLE" if abs(delta) > 5 else "  [~]")

    avg_delta = statistics.mean(deltas)
    print(f"\n  Avg Δ={avg_delta:.1f}W across {rounds} rounds",
          "  [!!!] ACTIVATION TYPE FINGERPRINTABLE" if abs(avg_delta) > 5 else "  [~]")
    del dense, sparse, inp

if __name__ == "__main__":
    print("=== Experiment 39: GPU Power Side-Channel (NVML) ===")
    print(f"    Device: cuda:0  Baseline power: {get_power(0):.1f}W")

    test_a()
    test_b()
    test_c()
    test_d()
    test_e()
    print("\n=== Done ===")
