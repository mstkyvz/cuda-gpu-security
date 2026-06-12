"""
Experiment 40: cuFFT Intermediate Buffer Residue

torch.fft functions (rfft, fft, fftn, irfft) use cuFFT internally.
cuFFT allocates a workspace/scratch buffer from PyTorch's CachingAllocator.
After the FFT completes, output and internal buffers are freed to the pool.

Attack scenario: speech-to-text server processes SECRET audio via FFT.
Next caller's torch.empty gets the frequency-domain representation.
From FFT spectrum → inverse FFT → recover original audio signal.

Tests (5 passes, 3+ verification methods each):
  A. rfft residue      — real FFT of SECRET audio signal
  B. fft residue       — complex FFT, larger output shape
  C. fftn residue      — 2D FFT (image spectrogram / mel-filterbank)
  D. irfft residue     — inverse FFT output (back to time domain)
  E. End-to-end audio  — mel spectrogram pipeline residue
  F. Reconstruction    — from FFT residue, recover original signal via IFFT
"""

import torch, torch.nn.functional as F, gc, math

DEVICE = torch.device("cuda:1")
SECRET = 3.14159

def separator(t):
    print(f"\n{'='*60}\n  {t}\n{'='*60}")

def verify(true_t, residue, label=""):
    """3 verification methods"""
    t = true_t.float().reshape(-1)
    r = residue.float().reshape(-1)
    # Method 1: cosine similarity
    cos = F.cosine_similarity(t.unsqueeze(0), r.unsqueeze(0)).item()
    # Method 2: mean/std relative match
    mean_ok = abs(r.mean() - t.mean()) < 0.05 * t.abs().mean() + 1e-4
    std_ok  = abs(r.std()  - t.std())  < 0.05 * t.std()  + 1e-4
    # Method 3: top-k magnitude match (for FFT: dominant freq components)
    if t.numel() > 10:
        top_k = min(32, t.numel() // 4)
        t_topk_idx = t.abs().topk(top_k).indices
        r_topk_idx = r.abs().topk(top_k).indices
        overlap = len(set(t_topk_idx.tolist()) & set(r_topk_idx.tolist()))
        topk_match = overlap / top_k
    else:
        topk_match = 0.0
    leak = abs(cos) > 0.9 or (mean_ok and std_ok) or topk_match > 0.7
    return cos, mean_ok, std_ok, topk_match, leak

# ══════════════════════════════════════════════════════════════
# Test A: torch.fft.rfft residue (real FFT — audio)
# SECRET audio: single-frequency sine wave at known frequency
# rfft output shape: [N//2 + 1] complex
# ══════════════════════════════════════════════════════════════
def test_a(passes=5):
    separator("Test A: torch.fft.rfft (real FFT) residue — 5 passes, 3 methods")
    N = 65536  # 64K samples — typical audio buffer
    # SECRET signal: 440 Hz tone (known frequency)
    t_axis = torch.arange(N, dtype=torch.float32, device=DEVICE) / 16000.0
    signal = torch.sin(2 * math.pi * 440 * t_axis) * SECRET

    leaks = 0
    for p in range(1, passes+1):
        torch.cuda.empty_cache()
        fft_out = torch.fft.rfft(signal)  # [N//2+1] complex
        true_mag = fft_out.abs().clone()   # magnitude spectrum
        out_shape = fft_out.shape
        del fft_out
        torch.cuda.synchronize(DEVICE)

        residue = torch.empty(out_shape, dtype=torch.complex64, device=DEVICE)
        torch.cuda.synchronize(DEVICE)

        cos, mo, so, topk, leak = verify(true_mag, residue.abs(), f"A.p{p}")
        if leak: leaks += 1
        print(f"  Pass {p}: cos={cos:.4f}  mean_ok={mo}  std_ok={so}  "
              f"topk_match={topk:.2f}",
              "  [!!!] LEAK" if leak else "  [~]")
        del residue
        gc.collect()

    del signal, t_axis
    print(f"  Result: {leaks}/{passes} passes LEAKED")

# ══════════════════════════════════════════════════════════════
# Test B: torch.fft.fft (complex FFT) residue
# Complex FFT output is [N] complex — twice the size of rfft
# ══════════════════════════════════════════════════════════════
def test_b(passes=5):
    separator("Test B: torch.fft.fft (complex FFT) residue — 5 passes")
    N = 32768
    t_axis = torch.arange(N, dtype=torch.float32, device=DEVICE) / 16000.0
    signal = (torch.sin(2 * math.pi * 440 * t_axis) +
              0.5 * torch.sin(2 * math.pi * 880 * t_axis)) * SECRET
    signal_c = signal.to(torch.complex64)

    leaks = 0
    for p in range(1, passes+1):
        torch.cuda.empty_cache()
        fft_out = torch.fft.fft(signal_c)  # [N] complex
        true_mag = fft_out.abs().clone()
        out_shape = fft_out.shape
        del fft_out
        torch.cuda.synchronize(DEVICE)

        residue = torch.empty(out_shape, dtype=torch.complex64, device=DEVICE)
        torch.cuda.synchronize(DEVICE)

        cos, mo, so, topk, leak = verify(true_mag, residue.abs(), f"B.p{p}")
        if leak: leaks += 1
        print(f"  Pass {p}: cos={cos:.4f}  mean_ok={mo}  std_ok={so}  "
              f"topk_match={topk:.2f}",
              "  [!!!] LEAK" if leak else "  [~]")
        del residue
        gc.collect()

    del signal, signal_c, t_axis
    print(f"  Result: {leaks}/{passes} passes LEAKED")

# ══════════════════════════════════════════════════════════════
# Test C: torch.fft.fftn residue (2D FFT — image / spectrogram)
# STFT-like: image spectrogram shape [frames, freqs]
# ══════════════════════════════════════════════════════════════
def test_c(passes=5):
    separator("Test C: torch.fft.fftn (2D FFT) residue — spectrogram shape")
    H, W = 256, 256  # spectrogram: 256 time frames × 256 freq bins
    # SECRET image: known pattern (horizontal gradient × SECRET)
    img = torch.zeros(H, W, device=DEVICE)
    for i in range(H):
        img[i] = SECRET * (i / H)  # known gradient pattern

    leaks = 0
    for p in range(1, passes+1):
        torch.cuda.empty_cache()
        fft_out = torch.fft.fftn(img)  # [H, W] complex
        true_mag = fft_out.abs().clone()
        out_shape = fft_out.shape
        del fft_out
        torch.cuda.synchronize(DEVICE)

        residue = torch.empty(out_shape, dtype=torch.complex64, device=DEVICE)
        torch.cuda.synchronize(DEVICE)

        cos, mo, so, topk, leak = verify(true_mag, residue.abs(), f"C.p{p}")
        if leak: leaks += 1
        print(f"  Pass {p}: cos={cos:.4f}  mean_ok={mo}  std_ok={so}  "
              f"topk_match={topk:.2f}",
              "  [!!!] LEAK" if leak else "  [~]")
        del residue
        gc.collect()

    del img
    print(f"  Result: {leaks}/{passes} passes LEAKED")

# ══════════════════════════════════════════════════════════════
# Test D: torch.fft.irfft residue (inverse FFT output)
# After irfft, output is real audio signal [N] — if in pool,
# attacker gets reconstructed time-domain signal directly
# ══════════════════════════════════════════════════════════════
def test_d(passes=5):
    separator("Test D: torch.fft.irfft (inverse FFT) output residue")
    N = 65536
    t_axis = torch.arange(N, dtype=torch.float32, device=DEVICE) / 16000.0
    signal = torch.sin(2 * math.pi * 440 * t_axis) * SECRET
    fft_rep = torch.fft.rfft(signal)  # go to freq domain first

    leaks = 0
    for p in range(1, passes+1):
        torch.cuda.empty_cache()
        # irfft: freq domain → time domain (output = original signal)
        recovered = torch.fft.irfft(fft_rep, n=N)  # [N] real
        true_out = recovered.clone()
        out_shape = recovered.shape
        del recovered
        torch.cuda.synchronize(DEVICE)

        residue = torch.empty(out_shape, dtype=torch.float32, device=DEVICE)
        torch.cuda.synchronize(DEVICE)

        cos, mo, so, topk, leak = verify(true_out, residue, f"D.p{p}")
        if leak: leaks += 1
        print(f"  Pass {p}: cos={cos:.4f}  mean_ok={mo}  std_ok={so}  "
              f"topk_match={topk:.2f}",
              "  [!!!] LEAK" if leak else "  [~]")
        del residue
        gc.collect()

    del signal, t_axis, fft_rep
    print(f"  Result: {leaks}/{passes} passes LEAKED")

# ══════════════════════════════════════════════════════════════
# Test E: End-to-end mel spectrogram pipeline
# Common in speech models: signal → stft → mel filterbank → log
# Each step creates intermediate tensors in pool
# ══════════════════════════════════════════════════════════════
def test_e(passes=5):
    separator("Test E: Mel spectrogram pipeline residue")
    SR = 16000; N = SR  # 1 second of audio
    WIN = 400; HOP = 160; N_FFT = 512; N_MELS = 80

    t_axis = torch.arange(N, dtype=torch.float32, device=DEVICE) / SR
    signal = torch.sin(2 * math.pi * 440 * t_axis) * SECRET

    # Precompute mel filterbank
    mel_fb = torch.zeros(N_MELS, N_FFT // 2 + 1, device=DEVICE)
    for m in range(N_MELS):
        lo = int(m * (N_FFT // 2 + 1) / N_MELS)
        hi = int((m + 2) * (N_FFT // 2 + 1) / N_MELS)
        hi = min(hi, N_FFT // 2 + 1)
        mel_fb[m, lo:hi] = 1.0 / max(hi - lo, 1)

    leaks = 0
    for p in range(1, passes+1):
        torch.cuda.empty_cache()
        # STFT frames
        frames = signal.unfold(0, WIN, HOP)  # [T, WIN]
        window = torch.hann_window(WIN, device=DEVICE)
        frames_w = frames * window
        fft_frames = torch.fft.rfft(frames_w, n=N_FFT)  # [T, N_FFT//2+1]
        mag = fft_frames.abs()  # [T, N_FFT//2+1]
        mel = (mel_fb @ mag.T).T  # [T, N_MELS]
        log_mel = torch.log(mel + 1e-6)  # [T, N_MELS]

        true_logmel = log_mel.clone()
        out_shape = log_mel.shape
        del log_mel, mel, mag, fft_frames, frames_w, frames, window
        torch.cuda.synchronize(DEVICE)

        residue = torch.empty(out_shape, dtype=torch.float32, device=DEVICE)
        torch.cuda.synchronize(DEVICE)

        cos, mo, so, topk, leak = verify(true_logmel, residue, f"E.p{p}")
        if leak: leaks += 1
        print(f"  Pass {p}: log_mel shape={out_shape}  cos={cos:.4f}  "
              f"mean_ok={mo}  topk={topk:.2f}",
              "  [!!!] LEAK" if leak else "  [~]")
        del residue, true_logmel
        gc.collect()

    del signal, t_axis, mel_fb
    print(f"  Result: {leaks}/{passes} passes LEAKED")

# ══════════════════════════════════════════════════════════════
# Test F: Signal reconstruction — FFT residue → IFFT → audio
# Verify that signal reconstructed from pool residue
# matches the original SECRET audio within tolerance
# ══════════════════════════════════════════════════════════════
def test_f(passes=5):
    separator("Test F: Signal reconstruction from FFT residue via IFFT")
    N = 65536
    t_axis = torch.arange(N, dtype=torch.float32, device=DEVICE) / 16000.0
    SECRET_SIGNAL = torch.sin(2 * math.pi * 440 * t_axis) * SECRET

    recons = 0
    for p in range(1, passes+1):
        torch.cuda.empty_cache()
        fft_out = torch.fft.rfft(SECRET_SIGNAL)  # [N//2+1] complex
        true_fft = fft_out.clone()
        out_shape = fft_out.shape
        del fft_out
        torch.cuda.synchronize(DEVICE)

        # Attacker gets the FFT output from pool
        residue = torch.empty(out_shape, dtype=torch.complex64, device=DEVICE)
        torch.cuda.synchronize(DEVICE)

        # Reconstruct: IFFT of residue
        reconstructed = torch.fft.irfft(residue, n=N)

        # Quality: cosine similarity with original signal
        cos_signal = F.cosine_similarity(
            SECRET_SIGNAL.unsqueeze(0), reconstructed.unsqueeze(0)).item()
        # Also check peak frequency match
        fft_resid_true = torch.fft.rfft(reconstructed).abs()
        fft_orig_true  = torch.fft.rfft(SECRET_SIGNAL).abs()
        peak_orig  = fft_orig_true.argmax().item()  # should be 440 Hz bin
        peak_resid = fft_resid_true.argmax().item()
        freq_match = (peak_orig == peak_resid)

        recon_ok = abs(cos_signal) > 0.8 or freq_match
        if recon_ok: recons += 1
        print(f"  Pass {p}: signal_cosine={cos_signal:.4f}  "
              f"peak_orig={peak_orig}  peak_resid={peak_resid}  "
              f"freq_match={freq_match}",
              "  [!!!] AUDIO RECONSTRUCTED" if recon_ok else "  [~]")
        del residue, reconstructed, fft_resid_true, fft_orig_true
        gc.collect()

    del SECRET_SIGNAL, t_axis, true_fft
    print(f"  Result: {recons}/{passes} passes audio reconstructed from FFT residue")

if __name__ == "__main__":
    print("=== Experiment 40: cuFFT Intermediate Buffer Residue ===")
    print(f"    PyTorch {torch.__version__}  GPU: cuda:1")
    print(f"    Verification: cosine_sim, mean/std match, top-k magnitude overlap")
    print()
    test_a()
    test_b()
    test_c()
    test_d()
    test_e()
    test_f()
    print("\n=== Done ===")
