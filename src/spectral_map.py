import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Signal generation
# -------------------------------------------------
n_long = np.arange(N * 16)
L = len(n_long)
half = L // 2

fin2 = min(3 * fin, 0.45 * Fs)

x_analog_new = np.zeros(L, dtype=np.float32)
x_analog_new[:half] = amp * np.sin(2 * np.pi * fin  * n_long[:half] / Fs)
x_analog_new[half:] = 0.6 * amp * np.sin(2 * np.pi * fin2 * n_long[half:] / Fs)

x_long = adc_quantize_q1_11(x_analog_new)

assert len(x_long) >= N, "Signal must be at least one FFT frame long"

# -------------------------------------------------
# STFT settings
# -------------------------------------------------
fft_len = 256                 # explicit 256-point FFT
hop = fft_len // 4            # 75% overlap
n_freqs = fft_len // 2 + 1
n_frames = 1 + (len(x_long) - fft_len) // hop

win = np.hanning(fft_len).astype(np.float32)

spec_fixed = np.zeros((n_frames, n_freqs), dtype=np.float64)
spec_npfft = np.zeros((n_frames, n_freqs), dtype=np.float64)

# -------------------------------------------------
# Frame processing
# -------------------------------------------------
for m in range(n_frames):
    start = m * hop

    frame_fp = x_long[start:start + fft_len].astype(np.float32)
    frame_np = x_analog_new[start:start + fft_len].astype(np.float32)

    frame_fp_win = frame_fp * win
    frame_np_win = frame_np * win

    # ---- Fixed-point FFT path ----
    X_fixed_full = dif_fft_radix2_fixedpoint(frame_fp_win)
    X_fixed_pos = X_fixed_full[:n_freqs]
    spec_fixed[m, :] = np.abs(X_fixed_pos) / (np.sum(win) + 1e-12)

    # ---- NumPy FFT path (256-point, FP32 input) ----
    X_np = np.fft.rfft(frame_np_win.astype(np.float32), n=fft_len)
    spec_npfft[m, :] = np.abs(X_np) / (np.sum(win) + 1e-12)

# -------------------------------------------------
# dB conversion
# -------------------------------------------------
eps = 1e-12
spec_fixed_db = 20 * np.log10(spec_fixed + eps)
spec_npfft_db = 20 * np.log10(spec_npfft + eps)

# Normalizing each plot to its own peak
spec_fixed_db -= np.max(spec_fixed_db)
spec_npfft_db -= np.max(spec_npfft_db)

freq_axis = (Fs / fft_len) * np.arange(n_freqs)
time_axis = (hop / Fs) * np.arange(n_frames)

# -------------------------------------------------
# Plot both spectrograms
# -------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

pcm0 = ax[0].pcolormesh(time_axis, freq_axis, spec_fixed_db.T,
                        shading='auto', cmap='viridis')
ax[0].set_title("Fixed-Point FFT Spectrogram")
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Frequency (Hz)")
ax[0].set_ylim(0, Fs / 2)
fig.colorbar(pcm0, ax=ax[0], label="Magnitude (dB, normalized)")

pcm1 = ax[1].pcolormesh(time_axis, freq_axis, spec_npfft_db.T,
                        shading='auto', cmap='viridis')
ax[1].set_title("NumPy FFT Spectrogram (256-pt, FP32)")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylim(0, Fs / 2)
fig.colorbar(pcm1, ax=ax[1], label="Magnitude (dB, normalized)")

plt.tight_layout()
plt.show()