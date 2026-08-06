# FFT256 — Fixed-Point 256-Point FFT (DIF) with RTL Implementation

A quantized model and RTL implementation of a **256-point radix-2 DIF FFT**, replicating the in-place FFT architecture of the reference paper (see [`reference_paper/`](reference_paper/)). The project covers the full path from algorithm to silicon-ready hardware:

1. **Floating-point Python model** — algorithm reference, verified against NumPy
2. **Fixed-point Python model** — bit-accurate golden reference for the RTL
3. **SystemVerilog RTL** — in-place, memory-based processing element (PE) + controller with AXI-Stream–style interfaces
4. **FPGA deployment** — running on an RFSoC4x1 (Zynq UltraScale+ RFSoC) at 200 MHz under PYNQ

The design simulates input from a **12-bit ADC** (Q1.11) and produces a 16-bit fixed-point spectrum (Q9.7), using a **block-wise (per-stage) scaling strategy** to manage FFT bit growth across the 8 stages.

## Architecture

| Property | Value |
|---|---|
| Transform | 256-point, radix-2, decimation-in-frequency (DIF), 8 stages |
| Input | Real-only, 12-bit signed **Q1.11** (simulated ADC codes) |
| Internal datapath | 16-bit, per-stage formats **Q2.14 → Q9.7** (one integer bit added per stage) |
| Twiddle factors | 16-bit **Q1.15** ROM (`twiddle_factors.sv`) |
| Rounding | Truncation (arithmetic right shift), matched bit-exactly between Python and RTL |
| Output | 16-bit signed **Q9.7** per bin, streamed in bit-reversed index order with `out_idx` |
| Memory | In-place, BRAM-based ping-pong memories inside the PE |
| Butterfly | 4-stage pipelined multiply/accumulate |
| Interfaces | AXI-Stream–style handshake on input and output (`valid/ready/last`) |
| Clock | 200 MHz on RFSoC4x1 (timing closed, WNS +1.99 ns) |
| Latency | 1,560 cycles ≈ **7.8 µs** per frame at 200 MHz (from cycle-accurate simulation) |

## Repository Layout

```
├── src/
│   ├── python/
│   │   ├── fft_floating_point.py   # float32 DIF FFT reference (vs NumPy)
│   │   └── fft_quantized.py        # bit-accurate fixed-point golden model
│   └── verilog/
│       ├── design.sv               # fft256_pe_controller_top (controller + I/O)
│       ├── fft_pe.sv               # fft_pe256_mem_trunc (in-place PE, BRAM + pipelined butterfly)
│       ├── twiddle_factors.sv      # Q1.15 twiddle ROM
│       ├── tb.sv                   # main testbench (two-tone signal, file output)
│       ├── tb_axis.sv              # AXI-Stream handshake corner cases
│       ├── tb_frames.sv            # back-to-back multi-frame test
│       └── tb_stress.sv            # long-run stress test (up to 1M frames)
├── runs/                           # generated outputs (plots, FFT dumps, sim results)
├── reference_paper/                # paper this architecture replicates
├── requirements.txt
└── LICENSE                         # Apache 2.0
```

## Getting Started (Python Models)

```bash
pip install -r requirements.txt
```

Both models generate the same **two-tone test signal**:

```
x[n] = 0.9·sin(2π·13·n/256) + 0.5·cos(2π·21·n/256)
```

Tone amplitudes/bins exercise multiple frequency bins, both sine and cosine phase, and input saturation (peak |x| ≈ 1.37 clips against the Q1.11 range).

**Floating-point reference:**

```bash
python src/python/fft_floating_point.py
```

Prints `Matches NumPy: True/False` and saves a time/frequency plot to `runs/python/fft_floating_point_plot.png`.

**Fixed-point golden model:**

```bash
python src/python/fft_quantized.py
```

Prints RMSE, mean magnitude, and normalized RMSE against NumPy FP64, then writes:

- `runs/python/fft_py_out.txt` — one line per bin: `index real imag` (signed Q9.7 codes), directly diff-able against the RTL output
- `runs/python/fft_magnitude.png` — fixed-point vs FP64 magnitude spectrum

## RTL Simulation (Icarus Verilog)

`design.sv` \`include\`s `fft_pe.sv`, which \`include\`s `twiddle_factors.sv`, so only the top file and a testbench are passed to the compiler. From `src/verilog/`:

```bash
# Main two-tone testbench — writes fft_out.txt ("index real imag")
iverilog -g2012 -o tb design.sv tb.sv
vvp tb

# AXI-Stream corner cases: input stalls, output backpressure,
# TVALID/TDATA stability, TLAST placement — vs a golden no-stall run
iverilog -g2012 -o tb_axis design.sv tb_axis.sv
vvp tb_axis

# Back-to-back frames through the same core
iverilog -g2012 -o tb_frames design.sv tb_frames.sv
vvp tb_frames

# Stress test: cycles input tones across 125 bins, checks peak bin,
# bin count, and TLAST per frame. Plusargs: +frames=N +progress=N +tone0=N
iverilog -g2012 -o tb_stress design.sv tb_stress.sv
vvp tb_stress +frames=10000 +progress=1000
```

Compare RTL against the golden model:

```bash
diff src/verilog/fft_out.txt runs/python/fft_py_out.txt
```

## Validation Results

| Comparison | Result |
|---|---|
| Float Python vs NumPy | Matches within float32 tolerance |
| Fixed-point Python vs RTL sim | 256/256 bins within ±3 LSB (input quantizers differ: `np.round` vs `$rtoi` truncation) |
| Fixed-point Python vs **FPGA** | **256/256 bins bit-exact** (identical `np.round` input quantization) |

Low-magnitude bins account for nearly all LSB-level sim differences; there is no systematic bias.

## FPGA Deployment (RFSoC4x1 + PYNQ)

The core is packaged as an IP with AXI-Stream input/output and driven from a PYNQ Jupyter notebook via AXI DMA:

- Input: 256 real samples as Q1.11 codes (`np.round(x * 2048)`, clipped to [-2048, 2047])
- Output: 256 × 32-bit words (packed Q9.7 real/imag), bit-reversed bin order
- PL clock: 200 MHz

Measured performance:

| Layer | Latency |
|---|---|
| FFT kernel (hardware) | **7.8 µs** (1,560 cycles @ 200 MHz, cycle-accurate sim) |
| Hardware + DMA, on-board bound | < 386 µs (limited by MMIO polling granularity — hardware finishes before the first status read completes) |
| End-to-end via Python/Jupyter | ~2.4 ms (>99 % software/DMA driver overhead) |

## Reference

This design replicates the in-place FFT architecture described in the paper included at `reference_paper/fft_architectures_seizure_prediction.pdf`.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
