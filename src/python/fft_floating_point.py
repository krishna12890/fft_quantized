import math
import os

import matplotlib.pyplot as plt
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
runs_dir = os.path.join(PROJECT_DIR, "runs")
os.makedirs(runs_dir, exist_ok=True)


def _bit_reverse_indices(N: int) -> np.ndarray:
    """Returns bit-reversed index permutation for length N=2^m."""
    m = int(math.log2(N))
    rev = np.zeros(N, dtype=np.uint32)
    for i in range(N):
        x = i
        r = 0
        for _ in range(m):
            r = (r << 1) | (x & 1)
            x >>= 1
        rev[i] = r
    return rev


def fft16_dif_real_f32(x, N, natural_order=True) -> np.ndarray:
    """
    N-point radix-2 DIF FFT for real-valued input (float32), in-place per stage.
    Returns complex64 spectrum; if natural_order is True, applies final bit-reversal.
    """

    x = np.asarray(x, dtype=np.float32)
    if x.size != N:
        raise ValueError("Input length must be 16")
    # Work buffer as complex64
    a = x.astype(np.complex64, copy=True)

    # DIF stages (Gentleman–Sande butterflies)
    m = N
    while m > 1:
        half = m // 2
        for k in range(half):
            # Twiddle for this stage/butterfly
            W = np.complex64(np.exp(-2j * np.pi * (k / m)))
            for r in range(0, N, m):
                i = r + k
                j = i + half
                u = a[i]
                v = a[j]
                a[i] = u + v
                a[j] = (u - v) * W
        m = half

    if natural_order:
        # DIF yields bit-reversed order; permute to linear
        rev = _bit_reverse_indices(N)
        a = a[rev]
    return a.astype(np.complex64)


if __name__ == "__main__":
    # Test signal (float32): sum of two cosines exactly on DFT bins (k=3 and k=5)
    N = 256
    Fs = 256
    tone_bin = 11
    fin = tone_bin * Fs / N
    amp = 0.9
    n = np.arange(N, dtype=np.float32)

    x = amp * np.sin(2 * np.pi * fin * n / Fs)

    # Our DIF FFT
    X_dif = fft16_dif_real_f32(x, N, natural_order=True)

    # Reference NumPy
    X_np = np.fft.fft(x.astype(np.float32))

    # Verify numerical agreement (float32 tolerance)
    ok = np.allclose(X_dif, X_np, rtol=1e-5, atol=1e-5)
    print("Matches NumPy:", ok)

    # Plot time-domain signal
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
    axs[0].plot(n, x, marker="o")
    axs[0].set_title("Time Domain (x[n])")
    axs[0].set_xlabel("n")
    axs[0].set_ylabel("Amplitude")

    # Plot magnitude spectrum |X[k]| centered at 0 Hz
    freq_axis = np.fft.fftshift(np.fft.fftfreq(N, d=1 / Fs))
    axs[1].plot(freq_axis, np.abs(np.fft.fftshift(X_dif)))
    axs[1].set_title("Frequency Domain (|X[k]|)")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Magnitude")
    axs[1].set_xlim(-Fs / 2, Fs / 2)
    plt.savefig(
        os.path.join(runs_dir, "fft_floating_point_plot.png"),
        dpi=300,
        bbox_inches="tight",
    )
    # plt.show()
