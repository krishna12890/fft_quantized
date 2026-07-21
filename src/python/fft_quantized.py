import csv
import math
import os

import matplotlib.pyplot as plt
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
# The runs/python directory holds the generated Python artifacts (FFT output + plots).
runs_dir = os.path.join(PROJECT_DIR, "runs", "python")
os.makedirs(runs_dir, exist_ok=True)

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

    # datapath drops fractional bits with an arithmetic right shift (>>>).
    def q_scalar(z):
        if np.iscomplexobj(z):
            r = np.floor(z.real * scale).astype(np.int64)
            i = np.floor(z.imag * scale).astype(np.int64)
            r = np.clip(r, lo, hi).astype(np.int32)
            i = np.clip(i, lo, hi).astype(np.int32)
            return r + 1j * i

        zz = np.floor(z * scale).astype(np.int64)
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

    def trunc_shift(v):
        if shift > 0:
            return v >> shift
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

        pr_s = trunc_shift(pr)
        pi_s = trunc_shift(pi)

        pr_s = np.clip(pr_s, lo, hi).astype(np.int32)
        pi_s = np.clip(pi_s, lo, hi).astype(np.int32)
        return pr_s + 1j * pi_s

    prod = a.astype(np.int64) * b.astype(np.int64)
    prod_s = trunc_shift(prod)
    prod_s = np.clip(prod_s, lo, hi).astype(np.int32)
    return prod_s


def twiddle_quantized(k, block, fmt_tw):
    angle = -2j * np.pi * k / block
    W = np.exp(angle)
    re_q = np.int32(np.round(W.real * (fmt_tw["scale"] - 1)))
    im_q = np.int32(np.round(W.imag * (fmt_tw["scale"] - 1)))
    return re_q + 1j * im_q, fmt_tw


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

                if step == 1:
                    bot_new = sub_val
                else:
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

    def bit_reverse_indices(n):
        bits = int(math.log2(n))
        out = np.zeros(n, dtype=int)
        for i in range(n):
            b = f"{i:0{bits}b}"
            out[i] = int(b[::-1], 2)
        return out

    rev = bit_reverse_indices(N)
    X_br = X[rev]

    X_fx_full = fxp_to_float(X_br, stage_formats[-1])
    return X_fx_full


###############################################################################
# ADC quantization model: 12-bit Q1.11
###############################################################################


def adc_quantize_q1_11(x):
    scale = 2048.0
    code = np.trunc(x * scale)
    code = np.clip(code, -2048, 2047)
    return code / scale


###############################################################################
# Main test + MSE
###############################################################################


if __name__ == "__main__":
    Fs = 256
    N = 256
    tone_bin = 13
    fin = tone_bin * Fs / N

    amp = 0.9
    n = np.arange(N)
    x_analog = amp * np.sin(2 * np.pi * fin * n / Fs)

    x_q = adc_quantize_q1_11(x_analog)

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

    # Writes the output file used for comparison with the RTL.
    #
    # The RTL upscales its Q1.11 input by 3 bits (Q2.14) and ends at Q9.7, so
    # its output codes equal this model's final-stage codes: float * 128. The
    # testbench packs each bin as {out_im[15:0], out_re[15:0]} (32-bit), and
    # each field is written here as 16-bit two's complement to match.
    RTL_OUTPUT_SCALE = 128
    output_file = os.path.join(runs_dir, "fft_py_out.txt")
    with open(output_file, "w") as f:
        for i in range(N):
            real_val = int(np.real(X_fx_full[i]) * RTL_OUTPUT_SCALE) & 0xFFFF
            imag_val = int(np.imag(X_fx_full[i]) * RTL_OUTPUT_SCALE) & 0xFFFF

            # Packs the two fields as: real_16bit | (imag_16bit << 16)
            combined = real_val | (imag_val << 16)
            f.write(f"{i} {combined}\n")

    print(f"\nFFT output written to: {output_file}")

    # -------------------------------------------------------------------------
    # Plots the frequency-domain output: fixed-point FFT vs NumPy FP64 reference
    # -------------------------------------------------------------------------
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / Fs))
    mag_fx = np.fft.fftshift(np.abs(X_fx_full))
    mag_ref = np.fft.fftshift(mag_np)

    plt.figure(figsize=(9, 4))
    plt.plot(freqs, mag_ref, label="NumPy FFT (FP64)", linewidth=1.2)
    plt.plot(
        freqs,
        mag_fx,
        label="Fixed-Point FFT (16-bit)",
        linestyle="--",
        marker="o",
        markersize=3,
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("FFT Magnitude Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_file = os.path.join(runs_dir, "fft_magnitude.png")
    plt.savefig(plot_file, dpi=150)
    print(f"Frequency-domain plot written to: {plot_file}")
