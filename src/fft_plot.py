import numpy as np
import math
import matplotlib.pyplot as plt
import csv

###############################################################################
# Fixed-point format helpers
###############################################################################

def fxp_specs(int_bits, frac_bits, total_bits=16):
    assert int_bits + frac_bits == total_bits
    return {
        "int_bits": int_bits,
        "frac_bits": frac_bits,
        "total_bits": total_bits,
        "min_code": -(2**(total_bits-1)),
        "max_code":  (2**(total_bits-1) - 1),
        "scale":    2**frac_bits,
    }

def float_to_fxp(val, fmt):
    scale = fmt["scale"]
    lo = fmt["min_code"]
    hi = fmt["max_code"]

    def q_scalar(z):
        if np.iscomplexobj(z):
            #r = np.round(z.real * scale)
            r = np.trunc(z.real*scale)
            #if r == 1098:
             #   print("z=,z.scal=, r=",z.real,z.real*scale,r)
            #i = np.round(z.imag * scale)
            i = np.trunc(z.imag*scale)
            r = np.clip(r, lo, hi)
            i = np.clip(i, lo, hi)
            return r.astype(np.int32) + 1j * i.astype(np.int32)
        else:
            #zz = np.round(z * scale)
            zz = np.trunc(z*scale)
            if zz == 1098:
                print("z=, zz=",z*scale,zz)
            zz = np.clip(zz, lo, hi)
            return zz.astype(np.int32)

    if isinstance(val, np.ndarray):
        out = np.zeros(val.shape, dtype=complex if np.iscomplexobj(val) else np.int32)
        it = np.nditer(val, flags=['multi_index','refs_ok'], op_flags=['readonly'])
        while not it.finished:
            out[it.multi_index] = q_scalar(it[0].item())
            it.iternext()
        return out
    else:
        return q_scalar(val)

def fxp_to_float(val, fmt):
    scale = fmt["scale"]
    if np.iscomplexobj(val):
        return (val.real / scale) + 1j * (val.imag / scale)
    else:
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
    else:
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
    else:
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
            bias = (1 << (shift-1))
            v_pos = v >= 0
            v_rounded = np.where(v_pos, v + bias, v - bias)
            v_s = v_rounded >> shift
        elif shift < 0:
            v_s = v << (-shift)
        else:
            v_s = v
        return v_s

    if np.iscomplexobj(a) or np.iscomplexobj(b):
        ar = a.real if np.iscomplexobj(a) else a
        ai = a.imag if np.iscomplexobj(a) else 0
        br = b.real if np.iscomplexobj(b) else b
        bi = b.imag if np.iscomplexobj(b) else 0

        pr = ar.astype(np.int64)*br.astype(np.int64) - ai.astype(np.int64)*bi.astype(np.int64)
        pi = ar.astype(np.int64)*bi.astype(np.int64) + ai.astype(np.int64)*br.astype(np.int64)

        pr_s = round_shift(pr)
        pi_s = round_shift(pi)

        pr_s = np.clip(pr_s, lo, hi).astype(np.int32)
        pi_s = np.clip(pi_s, lo, hi).astype(np.int32)

        return pr_s + 1j * pi_s
    else:
        prod = a.astype(np.int64)*b.astype(np.int64)
        prod_s = round_shift(prod)
        prod_s = np.clip(prod_s, lo, hi).astype(np.int32)
        return prod_s

def twiddle_quantized(k, block):
    angle = -2j * np.pi * k / block
    W = np.exp(angle)
    fmt_tw = fxp_specs(1,11,12)  # Q1.15
    return float_to_fxp(W, fmt_tw), fmt_tw

###############################################################################
# CSV writer helper
###############################################################################
def write_stage_csv(filename, X):
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["real", "imag"])
        for x in X:
            w.writerow([int(np.real(x)), int(np.imag(x))])

def write_inp_csv(filename, X):
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        # optional header
        w.writerow(["sample"])
        for x in X:
            w.writerow([x])

###############################################################################
# Fixed-point DIF FFT core (your version)
###############################################################################

def dif_fft_radix2_fixedpoint(x_adc,base_filename="stage"):
    N = x_adc.shape[0]
    assert N == 256
    nstages = int(math.log2(N))
    assert nstages == 8

    # NOTE: your stage_formats here are *not* all 16-bit.
    # You mixed total_bits = 17..23. I'll leave as-is to match your post.
    stage_formats = [
         fxp_specs(2,16,18),
         fxp_specs(3,15,18),
         fxp_specs(4,14,18),
         fxp_specs(5,13,18),
         fxp_specs(6,12,18),
         fxp_specs(7,11,18),
         fxp_specs(8,10,18),
         fxp_specs(9,9,18),
    ]


    #fmt_stage0 = fxp_specs(1,11,12)
    #fmt_stage0 = stage_formats[0]
    fmt_stage0 = fxp_specs(1,11,12)
    X = float_to_fxp(x_adc.astype(np.complex128), fmt_stage0)

    write_stage_csv(f"{base_filename}input_18b.csv", X)

    step = N // 2
    stage_idx = 0
    while step >= 1:
        fmt_stage = stage_formats[stage_idx]
        half_step = step
        block = 2 * step

        for start in range(0, N, block):
            for k in range(half_step):
                Wq, fmt_tw = twiddle_quantized(k, block)

                i_top = start + k
                i_bot = i_top + half_step

                top = X[i_top]
                bot = X[i_bot]

                add_val = fxp_add(top, bot, fmt_stage)
                sub_val = fxp_sub(top, bot, fmt_stage)
                bot_new = fxp_mul(sub_val, Wq, fmt_stage, fmt_tw, fmt_stage)

                X[i_top] = add_val
                X[i_bot] = bot_new

                if i_bot == 192 or i_top==192:
                    print("bot_new=",bot_new);


        step //= 2
        stage_idx += 1

        if stage_idx < nstages:
            fmt_next = stage_formats[stage_idx]
            X_float = fxp_to_float(X, fmt_stage)
            X = float_to_fxp(X_float, fmt_next)
            #if start == 0:
                #write_stage_csv(f"{base_filename}twiddle_18b.csv", Wq)
            write_stage_csv(f"{base_filename}{stage_idx}_output_18b.csv", X)
            #print("X[192]=",X[192])

    # bit-reverse
    def bit_reverse_indices(n):
        bits = int(math.log2(n))
        out = np.zeros(n, dtype=int)
        for i in range(n):
            b = f"{i:0{bits}b}"
            out[i] = int(b[::-1], 2)
        return out

    #final_fmt = stage_formats[-1]
    #X_float_final = fxp_to_float(X, final_fmt)
    #X_float_final = X_float_final[bit_reverse_indices(N)]
    #return X_float_final
    rev = bit_reverse_indices(N)
    X_br = X[rev]
    write_stage_csv("final_fft_output.csv", X_br)

    return X_br
###############################################################################
# ADC quantization model: 12-bit Q1.11
###############################################################################

def adc_quantize_q1_11(x):
    scale = 2048.0
    code = np.round(x * scale)
    code = np.clip(code, -2048, 2047)
    return code / scale

###############################################################################
# SNR calculation (tone bin vs rest)
###############################################################################

def compute_snr_from_fft_bins(X_bins, fs_hz):
    N2 = len(X_bins)
    N  = (N2 - 1) * 2

    mag = np.abs(X_bins)
    #bin_rms = (mag * (2.0 / N)) / np.sqrt(2.0)
    bin_rms = mag
    bin_power = bin_rms**2

    fund_bin = np.argmax(bin_power[1:]) + 1

    p_sig = bin_power[fund_bin]

    exclude = [0, fund_bin]
    noise_bins = [i for i in range(len(bin_power)) if i not in exclude]
    p_noise = np.sum(bin_power[noise_bins])

    snr_db = 10*np.log10(p_sig / (p_noise + 1e-30))
    return snr_db, fund_bin, p_sig, p_noise

def snr_numpy_path(x_q, Fs):
    N = len(x_q)
    X_np = np.fft.rfft(x_q, n=N)
    return compute_snr_from_fft_bins(X_np, Fs)

def snr_fixedpoint_path(x_q, Fs):
    X_full = dif_fft_radix2_fixedpoint(x_q)
    X_pos = X_full[: len(x_q)//2 + 1]
    return compute_snr_from_fft_bins(X_pos, Fs)

###############################################################################
# Main test + MSE
###############################################################################

if __name__ == "__main__":
    Fs = 100e6
    N  = 256
    tone_bin = 13
    fin = tone_bin * Fs / N

    amp = 0.9
    n = np.arange(N)
    x_analog = amp * np.sin(2*np.pi*fin*n/Fs)

    x_q = adc_quantize_q1_11(x_analog)
    write_inp_csv("sine.csv", x_q)
    # NumPy reference SNR (1-sided)
    snr_np, fund_np, p_sig_np, p_noise_np = snr_numpy_path(x_q, Fs)

    # Fixed-point SNR (1-sided)
    snr_fx, fund_fx, p_sig_fx, p_noise_fx = snr_fixedpoint_path(x_q, Fs)

    # === MSE PART ===========================================================
    # full NumPy FFT (complex, 256 bins)
    X_np_full = np.fft.fft(x_q, n=N)
    # full fixed-point FFT (complex, 256 bins, already bit-reversed -> natural)
    X_fx_full = dif_fft_radix2_fixedpoint(x_q)

    # MSE over all 256 bins
    mse_all = np.mean(np.abs(X_fx_full - X_np_full)**2)

    # MSE over 0..N/2 (to match SNR style)
    mse_pos = np.mean(np.abs(X_fx_full[:N//2+1] - X_np_full[:N//2+1])**2)
    # ========================================================================

    print("===== Results =====")
    print("NumPy FFT:")
    print(f"  fund bin      = {fund_np}")
    print(f"  signal power  = {p_sig_np:.3e}")
    print(f"  noise power   = {p_noise_np:.3e}")
    print(f"  SNR (dB)      = {snr_np:.2f} dB\n")

    print("Fixed-point FFT (your staged widths):")
    print(f"  fund bin      = {fund_fx}")
    print(f"  signal power  = {p_sig_fx:.3e}")
    print(f"  noise power   = {p_noise_fx:.3e}") 
    print(f"  SNR (dB)      = {snr_fx:.2f} dB\n")

    print("MSE vs NumPy:")
    print(f"  MSE (all 256 bins): {mse_all:.6e}")
    print(f"  MSE (0..N/2 bins):  {mse_pos:.6e}")

    f = np.arange(N)
    Omega = 2*np.pi*fin/N
    fig, axs = plt.subplots(2, 1, figsize=(16,10), constrained_layout = True)

    X_ref = np.fft.fft(x_q)

    # Plotting FFT signal
    X_f = dif_fft_radix2_fixedpoint(x_q)

    # Calculate the absolute error in the frequency domain
    error_mag = np.abs(X_ref - X_f)

    # Calculate PSD
    psd_ref = (np.abs(X_ref)**2) / N
    psd_fix = (np.abs(X_f)**2) / N

    # Convert to dBFS (Relative to a peak of 1.0)
    db_ref = 10 * np.log10(psd_ref + 1e-12)
    db_fix = 10 * np.log10(psd_fix + 1e-12)

    # Frequency axis
    freqs = np.fft.fftfreq(N, d=1/Fs)

    # Plotting PSD 
    # plt.figure(figsize=(10, 6))
    # plt.plot(freqs[:N//2] / 1e6, db_ref[:N//2], label='Ideal Float PSD', alpha=0.8)
    # plt.plot(freqs[:N//2] / 1e6, db_fix[:N//2], '--', label='16-bit Fixed PSD', alpha=0.8)
    # plt.title(f"PSD Comparison: Quantization Noise Floor (N={N})")
    # plt.xlabel("Frequency (MHz)")
    # plt.ylabel("Power/Frequency (dB/Hz)")
    # plt.grid(True, which='both', linestyle='--', alpha=0.5)
    # plt.legend()
    # plt.ylim(bottom=-160) # Focused on the noise floor
    # plt.show()

    fig, ax = plt.subplots(figsize=(23, 7))
    ax.plot(f, 20*np.log10(error_mag + 1e-12), label='Quantization Error', alpha=0.5)

    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Quantization Effect: Ideal vs Fixed-Point FFT")
    ax.legend()
    plt.grid(True)
    plt.show()