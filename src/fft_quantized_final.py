import csv
import math

import matplotlib.pyplot as plt
import numpy as np

###############################################################################
# Fixed-point format helpers
###############################################################################


def fxp_specs(int_bits, frac_bits, total_bits=16):
    assert int_bits + frac_bits == total_bits
    return {
        "int_bits": int_bits,
        "frac_bits": frac_bits,
        "total_bits": total_bits,
        "min_code": -(2 ** (total_bits - 1)),
        "max_code": (2 ** (total_bits - 1) - 1),
        "scale": 2**frac_bits,
    }


def float_to_fxp(val, fmt):
    scale = fmt["scale"]
    lo = fmt["min_code"]
    hi = fmt["max_code"]

    def q_scalar(z):
        if np.iscomplexobj(z):
            r = np.trunc(z.real * scale).astype(np.int64)
            i = np.trunc(z.imag * scale).astype(np.int64)
            r = np.clip(r, lo, hi).astype(np.int32)
            i = np.clip(i, lo, hi).astype(np.int32)
            return r + 1j * i

        zz = np.trunc(z * scale).astype(np.int64)
        zz = np.clip(zz, lo, hi).astype(np.int32)
        return zz

    if isinstance(val, np.ndarray):
        out_dtype = complex if np.iscomplexobj(val) else np.int32
        out = np.zeros(val.shape, dtype=out_dtype)
        it = np.nditer(
            val,
            flags=["multi_index", "refs_ok"],
            op_flags=["readonly"],
        )
        while not it.finished:
            out[it.multi_index] = q_scalar(it[0].item())
            it.iternext()
        return out

    return q_scalar(val)


def fxp_to_float(val, fmt):
    scale = fmt["scale"]
    if np.iscomplexobj(val):
        return (val.real / scale) + 1j * (val.imag / scale)
    return val / scale


def fxp_add(a, b, fmt):
    lo = fmt["min_code"]
    hi = fmt["max_code"]

    if np.iscomplexobj(a) or np.iscomplexobj(b):
        ar = a.real if np.iscomplexobj(a) else a
        ai = a.imag if np.iscomplexobj(a) else 0
        br = b.real if np.iscomplexobj(b) else b
        bi = b.imag if np.iscomplexobj(b) else 0

        rr = ar.astype(np.int64) + br.astype(np.int64)
        ii = ai.astype(np.int64) + bi.astype(np.int64)
        rr = np.clip(rr, lo, hi).astype(np.int32)
        ii = np.clip(ii, lo, hi).astype(np.int32)
        return rr + 1j * ii

    c = a.astype(np.int64) + b.astype(np.int64)
    c = np.clip(c, lo, hi).astype(np.int32)
    return c


def fxp_sub(a, b, fmt):
    lo = fmt["min_code"]
    hi = fmt["max_code"]

    if np.iscomplexobj(a) or np.iscomplexobj(b):
        ar = a.real if np.iscomplexobj(a) else a
        ai = a.imag if np.iscomplexobj(a) else 0
        br = b.real if np.iscomplexobj(b) else b
        bi = b.imag if np.iscomplexobj(b) else 0

        rr = ar.astype(np.int64) - br.astype(np.int64)
        ii = ai.astype(np.int64) - bi.astype(np.int64)
        rr = np.clip(rr, lo, hi).astype(np.int32)
        ii = np.clip(ii, lo, hi).astype(np.int32)
        return rr + 1j * ii

    c = a.astype(np.int64) - b.astype(np.int64)
    c = np.clip(c, lo, hi).astype(np.int32)
    return c


def fxp_mul(a, b, fmt_a, fmt_b, fmt_out):
    fa = fmt_a["frac_bits"]
    fb = fmt_b["frac_bits"]
    fo = fmt_out["frac_bits"]
    shift = fa + fb - fo

    lo = fmt_out["min_code"]
    hi = fmt_out["max_code"]

    def round_shift(v):
        if shift > 0:
            bias = 1 << (shift - 1)
            v_pos = v >= 0
            v_rounded = np.where(v_pos, v + bias, v - bias)
            return v_rounded >> shift
        if shift < 0:
            return v << (-shift)
        return v

    if np.iscomplexobj(a) or np.iscomplexobj(b):
        ar = a.real if np.iscomplexobj(a) else a
        ai = a.imag if np.iscomplexobj(a) else 0
        br = b.real if np.iscomplexobj(b) else b
        bi = b.imag if np.iscomplexobj(b) else 0

        pr = ar.astype(np.int64) * br.astype(np.int64) - ai.astype(
            np.int64
        ) * bi.astype(np.int64)
        pi = ar.astype(np.int64) * bi.astype(np.int64) + ai.astype(
            np.int64
        ) * br.astype(np.int64)

        pr_s = round_shift(pr)
        pi_s = round_shift(pi)

        pr_s = np.clip(pr_s, lo, hi).astype(np.int32)
        pi_s = np.clip(pi_s, lo, hi).astype(np.int32)
        return pr_s + 1j * pi_s

    prod = a.astype(np.int64) * b.astype(np.int64)
    prod_s = round_shift(prod)
    prod_s = np.clip(prod_s, lo, hi).astype(np.int32)
    return prod_s


def twiddle_quantized(k, block, fmt_tw):
    angle = -2j * np.pi * k / block
    W = np.exp(angle)
    return float_to_fxp(W, fmt_tw), fmt_tw


###############################################################################
# CSV writer helper
###############################################################################


def write_stage_csv(filename, X):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["real", "imag"])
        for x in X:
            writer.writerow([int(np.real(x)), int(np.imag(x))])


def write_inp_csv(filename, X):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample"])
        for x in X:
            writer.writerow([x])


###############################################################################
# Fixed-point DIF FFT core
###############################################################################


def dif_fft_radix2_fixedpoint(x_adc, base_filename="stage"):
    N = x_adc.shape[0]
    assert N == 256
    nstages = int(math.log2(N))
    assert nstages == 8

    stage_formats = [
        fxp_specs(2, 14, 16),
        fxp_specs(3, 13, 16),
        fxp_specs(4, 12, 16),
        fxp_specs(5, 11, 16),
        fxp_specs(6, 10, 16),
        fxp_specs(7, 9, 16),
        fxp_specs(8, 8, 16),
        fxp_specs(9, 7, 16),
    ]

    fmt_stage0 = fxp_specs(1, 11, 12)
    X = float_to_fxp(x_adc.astype(np.complex128), fmt_stage0)
    write_stage_csv(f"{base_filename}input_16b.csv", X)

    X_float = fxp_to_float(X, fmt_stage0)
    X = float_to_fxp(X_float, stage_formats[0])

    step = N // 2
    stage_idx = 0
    fmt_tw = fxp_specs(1, 15, 16)

    while step >= 1:
        fmt_stage = stage_formats[stage_idx]
        half_step = step
        block = 2 * step

        for start in range(0, N, block):
            for k in range(half_step):
                Wq, _ = twiddle_quantized(k, block, fmt_tw)

                i_top = start + k
                i_bot = i_top + half_step

                top = X[i_top]
                bot = X[i_bot]

                add_val = fxp_add(top, bot, fmt_stage)
                sub_val = fxp_sub(top, bot, fmt_stage)
                bot_new = fxp_mul(
                    sub_val,
                    Wq,
                    fmt_stage,
                    fmt_tw,
                    fmt_stage,
                )

                X[i_top] = add_val
                X[i_bot] = bot_new

        step //= 2
        stage_idx += 1

        if stage_idx < nstages:
            fmt_next = stage_formats[stage_idx]
            X_float = fxp_to_float(X, fmt_stage)
            X = float_to_fxp(X_float, fmt_next)
            write_stage_csv(
                f"{base_filename}{stage_idx}_output_16b.csv",
                X,
            )

    def bit_reverse_indices(n):
        bits = int(math.log2(n))
        out = np.zeros(n, dtype=int)
        for i in range(n):
            b = f"{i:0{bits}b}"
            out[i] = int(b[::-1], 2)
        return out

    rev = bit_reverse_indices(N)
    X_br = X[rev]
    write_stage_csv("final_fft_output.csv", X_br)

    X_fx_full = fxp_to_float(X_br, stage_formats[-1])
    return X_fx_full


###############################################################################
# ADC quantization model: 12-bit Q1.11
###############################################################################


def adc_quantize_q1_11(x):
    scale = 2048.0
    code = np.round(x * scale)
    code = np.clip(code, -2048, 2047)
    return code / scale


###############################################################################
# SNR calculation
###############################################################################


def compute_snr_from_fft_bins(X_bins, fs_hz):
    del fs_hz

    mag = np.abs(X_bins)
    bin_rms = mag
    bin_power = bin_rms**2

    fund_bin = np.argmax(bin_power[1:]) + 1
    p_sig = bin_power[fund_bin]

    exclude = [0, fund_bin]
    noise_bins = [i for i in range(len(bin_power)) if i not in exclude]
    p_noise = np.sum(bin_power[noise_bins])

    snr_db = 10 * np.log10(p_sig / (p_noise + 1e-30))
    return snr_db, fund_bin, p_sig, p_noise


def snr_numpy_path(x_q, Fs):
    N = len(x_q)
    X_np = np.fft.rfft(x_q, n=N)
    return compute_snr_from_fft_bins(X_np, Fs)


def snr_fixedpoint_path(x_q, Fs):
    X_full = dif_fft_radix2_fixedpoint(x_q)
    X_pos = X_full[: len(x_q) // 2 + 1]
    return compute_snr_from_fft_bins(X_pos, Fs)


###############################################################################
# Main test + MSE
###############################################################################


if __name__ == "__main__":
    Fs = 256
    N = 256
    tone_bin = 11
    fin = tone_bin * Fs / N

    amp = 0.9
    n = np.arange(N)
    x_analog = amp * np.sin(2 * np.pi * fin * n / Fs)

    x_q = adc_quantize_q1_11(x_analog)
    write_inp_csv("sine.csv", x_q)

    snr_np, _, _, _ = snr_numpy_path(x_q, Fs)
    snr_fx, _, _, _ = snr_fixedpoint_path(x_q, Fs)

    X_np_full = np.fft.fft(x_q, n=N)
    X_fx_full = dif_fft_radix2_fixedpoint(x_q)

    mse_all = np.mean(np.abs(X_fx_full - X_np_full) ** 2)

    rmse_all = np.sqrt(mse_all)
    mag_np = np.abs(X_np_full)
    nrmse = rmse_all / (np.mean(mag_np) + 1e-12)

    print("===== Results =====")
    print(f"RMSE: {rmse_all:.2f}")
    print(f"Mean magnitude (NumPy): {np.mean(mag_np):.2f}")
    print(f"Normalized RMSE: {nrmse * 100:.2f}%")
    print(
        f"SNR comparison: NumPy={snr_np:.2f} dB, "
        f"Fixed={snr_fx:.2f} dB, "
        f"Delta={snr_np - snr_fx:.2f} dB"
    )

    n_long = np.arange(N * 16)
    L = len(n_long)
    half = L // 2

    fin2 = min(3 * fin, 0.45 * Fs)

    x_analog_new = np.zeros(L, dtype=np.float32)
    x_analog_new[:half] = amp * np.sin(2 * np.pi * fin * n_long[:half] / Fs)
    x_analog_new[half:] = 0.6 * amp * np.sin(2 * np.pi * fin2 * n_long[half:] / Fs)

    x_long = adc_quantize_q1_11(x_analog_new)

    assert len(x_long) >= N, "Signal must be at least one FFT frame long"

    fft_len = 256
    hop = fft_len
    n_frames = 1 + (len(x_long) - fft_len) // hop

    win = np.hanning(fft_len).astype(np.float32)

    spec_fixed = np.zeros((n_frames, fft_len), dtype=np.float32)
    spec_npfft = np.zeros((n_frames, fft_len), dtype=np.float32)

    for m in range(n_frames):
        start = m * hop

        frame_fp = x_long[start : start + fft_len].astype(np.float32)
        frame_np = x_analog_new[start : start + fft_len].astype(np.float32)

        frame_fp_win = frame_fp * win
        frame_np_win = frame_np * win

        X_fixed_full = dif_fft_radix2_fixedpoint(frame_fp_win)
        spec_fixed[m, :] = np.abs(X_fixed_full) / (np.sum(win) + 1e-12)

        X_np = np.fft.fft(frame_np_win.astype(np.float32), n=fft_len)
        spec_npfft[m, :] = np.abs(X_np) / (np.sum(win) + 1e-12)

    eps = 1e-12
    spec_fixed_db = 20 * np.log10(spec_fixed + eps)
    spec_npfft_db = 20 * np.log10(spec_npfft + eps)

    spec_fixed_db -= np.max(spec_fixed_db)
    spec_npfft_db -= np.max(spec_npfft_db)

    spec_fixed_db = np.fft.fftshift(spec_fixed_db, axes=1)
    spec_npfft_db = np.fft.fftshift(spec_npfft_db, axes=1)

    freq_axis_full = np.fft.fftshift(np.fft.fftfreq(N, d=1 / Fs))
    X_fx_shifted = np.fft.fftshift(X_fx_full)
    X_np_shifted = np.fft.fftshift(X_np_full)

    plt.figure(figsize=(10, 4))
    plt.plot(
        freq_axis_full,
        np.abs(X_fx_shifted),
        label="Fixed-Point FFT",
        marker="o",
    )
    plt.plot(
        freq_axis_full,
        np.abs(X_np_shifted),
        label="NumPy FFT (256-pt, FP32)",
        marker="x",
    )
    plt.title("FFT Magnitude Comparison")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (normalized)")
    plt.xlim(-(Fs / 2), Fs / 2)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("fft_plot.png", dpi=300, bbox_inches="tight")
    plt.show()
