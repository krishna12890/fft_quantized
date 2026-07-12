import math
import os

import matplotlib.pyplot as plt
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
# The runs/python directory holds the generated Python artifacts (plots).
runs_dir = os.path.join(PROJECT_DIR, "runs", "python")
os.makedirs(runs_dir, exist_ok=True)


def _bit_reverse_indices(N: int) -> np.ndarray:
    """Returns the bit-reversed index permutation for length N=2^m."""
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
    Computes an N-point radix-2 DIF FFT for real-valued float32 input,
    in-place per stage. Returns the complex64 spectrum; when natural_order
    is True, applies a final bit-reversal.
    """

    x = np.asarray(x, dtype=np.float32)
    if x.size != N:
        raise ValueError("Input length must be N")
    # Holds the working buffer as complex64.
    a = x.astype(np.complex64, copy=True)

    # Iterates over the DIF stages.
    m = N
    while m > 1:
        half = m // 2
        for k in range(half):
            # Computes the twiddle factor for this stage/butterfly.
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
        # Bit-reverses the output to natural order.
        rev = _bit_reverse_indices(N)
        a = a[rev]
    return a.astype(np.complex64)


if __name__ == "__main__":
    N = 256
    Fs = 256
    tone_bin = 13
    fin = tone_bin * Fs / N
    amp = 0.9
    n = np.arange(N, dtype=np.float32)

    x = amp * np.sin(2 * np.pi * fin * n / Fs)

    # Runs the DIF FFT.
    X_dif = fft16_dif_real_f32(x, N, natural_order=True)

    # Computes the reference NumPy FFT.
    X_np = np.fft.fft(x.astype(np.float32))

    # Checks the result against NumPy within float32 tolerance.
    ok = np.allclose(X_dif, X_np, rtol=1e-5, atol=1e-5)
    print("Matches NumPy:", ok)

    # Plots the time-domain signal.
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
    axs[0].plot(n, x, marker="o")
    axs[0].set_title("Time Domain (x[n])")
    axs[0].set_xlabel("n")
    axs[0].set_ylabel("Amplitude")

    # Plots the frequency magnitude spectrum |X[k]| centered at 0 Hz.
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
