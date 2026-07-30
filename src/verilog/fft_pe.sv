`include "twiddle_factors.sv"
// ---------------------------------------------------------------------------
// fft_pe256_mem_trunc -- PIPELINED processing element
//
// Same bit-exact arithmetic as the original single-cycle PE, restructured for
// 200 MHz timing closure:
//
//   * mem0/mem1 are true-dual-port synchronous-read RAMs (BRAM-inferable,
//     1-cycle read latency, read-first). This removes the huge asynchronous
//     256:1 read muxes and the 8000+ fanout on the address lines.
//   * The butterfly is a 4-stage pipeline (paper: 2 regs on adds, 3 on mults):
//       issue : controller drives cmd/addrs (registered in the controller)
//       v1    : RAM read data + twiddle ROM output registered
//       v2    : add/sub results registered      (y0, t = a-b)
//       v3    : DSP partial products registered (p1..p4)
//       retire: post-add + >>>16 + trunc, results written to the RAM
//     => a butterfly issued at edge T has its writes committed at edge T+4.
//   * One butterfly can be issued EVERY cycle (throughput unchanged). Within
//     a stage reads and writes go to opposite memories, so in-flight ops
//     never collide. The controller must drain the pipeline (>=3 idle
//     cycles) before switching stages / starting readout.
//   * New input cmd_valid: ops enter the pipeline only when it is high, so
//     idle controller cycles no longer issue phantom READ commands.
//
// Latency contract used by the controller:
//   LOAD  : write commits at the edge after issue (unchanged)
//   READ  : read_re16/read_im16 valid 1 cycle after issue (unchanged)
//   BFLY_*: memory write commits 4 edges after issue
// ---------------------------------------------------------------------------
module fft_pe256_mem_trunc (
    input  wire        clk,

    // Command (qualified by cmd_valid)
    input  wire        cmd_valid,
    input  wire [1:0]  cmd,          // 00=LOAD, 01=READ, 10=BFLY_ADD, 11=BFLY_MUL

    // Unused here, for future stage-based control if needed
    input  wire [2:0]  stage_sel,

    // For butterflies:
    //   mem_sel = 0: read mem0, write mem1
    //   mem_sel = 1: read mem1, write mem0
    input  wire        mem_sel,

    // LOAD interface (12-bit external input)
    input  wire        load_mem_sel,   // 0->mem0, 1->mem1
    input  wire [7:0]  load_addr,
    input  wire signed [11:0] load_re12,
    input  wire signed [11:0] load_im12,
    input  wire        load_real_only, // 1 => imag forced to 0

    // READ interface (16-bit output)
    input  wire        read_mem_sel,   // 0->mem0, 1->mem1
    input  wire [7:0]  read_addr,
    output wire signed [15:0] read_re16,
    output wire signed [15:0] read_im16,
    output reg         read_valid,

    // Butterfly addresses
    input  wire [7:0]  addr_a,
    input  wire [7:0]  addr_b,
    input  wire [7:0]  addr_out_a,
    input  wire [7:0]  addr_out_b,

    // Twiddle index (0..255, Q1.15 in ROM)
    input  wire [7:0]  twiddle_idx,

    // Kept for port compatibility (had no effect in the original datapath)
    input  wire        real_compute,

    // Operation retire pulse
    output reg         done
);

    // --------------------------------------------------------
    // Command decode
    // --------------------------------------------------------
    wire is_load = cmd_valid && (cmd == 2'b00);
    wire is_read = cmd_valid && (cmd == 2'b01);
    wire is_bfly = cmd_valid &&  cmd[1];          // 10 or 11
    wire is_add  = cmd_valid && (cmd == 2'b10);

    // Bit-reversed write addresses (used by BFLY_ADD, final stage)
    wire [7:0] rev_addr_out_a = { addr_out_a[0], addr_out_a[1], addr_out_a[2], addr_out_a[3],
                                  addr_out_a[4], addr_out_a[5], addr_out_a[6], addr_out_a[7] };
    wire [7:0] rev_addr_out_b = { addr_out_b[0], addr_out_b[1], addr_out_b[2], addr_out_b[3],
                                  addr_out_b[4], addr_out_b[5], addr_out_b[6], addr_out_b[7] };

    // --------------------------------------------------------
    // Twiddle ROM (combinational LUT ROM, output registered in v1)
    // --------------------------------------------------------
    wire signed [15:0] W_re15, W_im15;
    twiddle_rom_256_16 U_TW_1 (
        .addr(twiddle_idx),
        .re(W_re15),
        .im(W_im15)
    );

    // --------------------------------------------------------
    // Helpers: sign-extension & truncation (same as original)
    // --------------------------------------------------------
    function signed [15:0] sext12to16;
        input signed [11:0] x;
        begin
            sext12to16 = { x[11], x, 3'b000 };
        end
    endfunction

    function signed [15:0] sx16_add_trunc;   // trunc16(sx16(a)+sx16(b))
        input signed [15:0] a, b;
        reg signed [16:0] s;
        begin
            s = a + b;
            sx16_add_trunc = s[15:0];
        end
    endfunction

    function signed [15:0] sx16_sub_trunc;   // trunc16(sx16(a)-sx16(b))
        input signed [15:0] a, b;
        reg signed [16:0] s;
        begin
            s = a - b;
            sx16_sub_trunc = s[15:0];
        end
    endfunction

    // --------------------------------------------------------
    // True-dual-port synchronous-read memories
    //   port A: butterfly operand a / LOAD write / READ / retire write y0
    //   port B: butterfly operand b / retire write y1
    // --------------------------------------------------------
    (* ram_style = "block" *) reg signed [15:0] mem0_re [0:255];
    (* ram_style = "block" *) reg signed [15:0] mem0_im [0:255];
    (* ram_style = "block" *) reg signed [15:0] mem1_re [0:255];
    (* ram_style = "block" *) reg signed [15:0] mem1_im [0:255];

    // Port control (combinational muxes, see below)
    reg        m0a_we, m0b_we, m1a_we, m1b_we;
    reg [7:0]  m0a_addr, m0b_addr, m1a_addr, m1b_addr;
    reg signed [15:0] m0a_wre, m0a_wim, m0b_wre, m0b_wim;
    reg signed [15:0] m1a_wre, m1a_wim, m1b_wre, m1b_wim;

    // Registered read data (BRAM output registers)
    reg signed [15:0] m0a_rre, m0a_rim, m0b_rre, m0b_rim;
    reg signed [15:0] m1a_rre, m1a_rim, m1b_rre, m1b_rim;

    // mem0 port A
    always @(posedge clk) begin
        if (m0a_we) begin
            mem0_re[m0a_addr] <= m0a_wre;
            mem0_im[m0a_addr] <= m0a_wim;
        end
        m0a_rre <= mem0_re[m0a_addr];
        m0a_rim <= mem0_im[m0a_addr];
    end
    // mem0 port B
    always @(posedge clk) begin
        if (m0b_we) begin
            mem0_re[m0b_addr] <= m0b_wre;
            mem0_im[m0b_addr] <= m0b_wim;
        end
        m0b_rre <= mem0_re[m0b_addr];
        m0b_rim <= mem0_im[m0b_addr];
    end
    // mem1 port A
    always @(posedge clk) begin
        if (m1a_we) begin
            mem1_re[m1a_addr] <= m1a_wre;
            mem1_im[m1a_addr] <= m1a_wim;
        end
        m1a_rre <= mem1_re[m1a_addr];
        m1a_rim <= mem1_im[m1a_addr];
    end
    // mem1 port B
    always @(posedge clk) begin
        if (m1b_we) begin
            mem1_re[m1b_addr] <= m1b_wre;
            mem1_im[m1b_addr] <= m1b_wim;
        end
        m1b_rre <= mem1_re[m1b_addr];
        m1b_rim <= mem1_im[m1b_addr];
    end

    // --------------------------------------------------------
    // Pipeline registers
    // --------------------------------------------------------
    // v1: read data arriving, write-back info piped
    reg        v1_valid, v1_is_add, v1_mem_sel;
    reg [7:0]  v1_waddr_a, v1_waddr_b;
    reg signed [15:0] v1_W_re, v1_W_im;

    // v2: butterfly add/sub done
    reg        v2_valid, v2_is_add, v2_mem_sel;
    reg [7:0]  v2_waddr_a, v2_waddr_b;
    reg signed [15:0] v2_W_re, v2_W_im;
    reg signed [15:0] v2_y0_re, v2_y0_im;
    reg signed [15:0] v2_t_re,  v2_t_im;   // MUL: t = a-b ; ADD: y1 = a-b

    // v3: DSP partial products
    reg        v3_valid, v3_is_add, v3_mem_sel;
    reg [7:0]  v3_waddr_a, v3_waddr_b;
    reg signed [15:0] v3_y0_re, v3_y0_im;
    reg signed [15:0] v3_y1add_re, v3_y1add_im;
    reg signed [31:0] p1, p2, p3, p4;

    // ---- v1 capture ----
    always @(posedge clk) begin
        v1_valid   <= is_bfly;
        v1_is_add  <= is_add;
        v1_mem_sel <= mem_sel;
        v1_waddr_a <= is_add ? rev_addr_out_a : addr_out_a;
        v1_waddr_b <= is_add ? rev_addr_out_b : addr_out_b;
        v1_W_re    <= W_re15;
        v1_W_im    <= W_im15;
    end

    // Operand select (RAM output registers of the memory being read)
    wire signed [15:0] a_re = v1_mem_sel ? m1a_rre : m0a_rre;
    wire signed [15:0] a_im = v1_mem_sel ? m1a_rim : m0a_rim;
    wire signed [15:0] b_re = v1_mem_sel ? m1b_rre : m0b_rre;
    wire signed [15:0] b_im = v1_mem_sel ? m1b_rim : m0b_rim;

    wire signed [15:0] sum_re = sx16_add_trunc(a_re, b_re);
    wire signed [15:0] sum_im = sx16_add_trunc(a_im, b_im);
    wire signed [15:0] dif_re = sx16_sub_trunc(a_re, b_re);
    wire signed [15:0] dif_im = sx16_sub_trunc(a_im, b_im);

    // ---- v2 capture ----
    always @(posedge clk) begin
        v2_valid   <= v1_valid;
        v2_is_add  <= v1_is_add;
        v2_mem_sel <= v1_mem_sel;
        v2_waddr_a <= v1_waddr_a;
        v2_waddr_b <= v1_waddr_b;
        v2_W_re    <= v1_W_re;
        v2_W_im    <= v1_W_im;
        if (v1_is_add) begin
            // Final stage: y0 = a+b, y1 = a-b (no scaling)
            v2_y0_re <= sum_re;
            v2_y0_im <= sum_im;
        end else begin
            // BFLY_MUL: y0 = trunc16(a+b) >>> 1 (per-stage scale)
            v2_y0_re <= sum_re >>> 1;
            v2_y0_im <= sum_im >>> 1;
        end
        v2_t_re <= dif_re;
        v2_t_im <= dif_im;
    end

    // ---- v3 capture (multipliers -> DSP48 with input+output registers) ----
    always @(posedge clk) begin
        v3_valid    <= v2_valid;
        v3_is_add   <= v2_is_add;
        v3_mem_sel  <= v2_mem_sel;
        v3_waddr_a  <= v2_waddr_a;
        v3_waddr_b  <= v2_waddr_b;
        v3_y0_re    <= v2_y0_re;
        v3_y0_im    <= v2_y0_im;
        v3_y1add_re <= v2_t_re;
        v3_y1add_im <= v2_t_im;
        p1 <= v2_t_re * v2_W_re;
        p2 <= v2_t_im * v2_W_im;
        p3 <= v2_t_re * v2_W_im;
        p4 <= v2_t_im * v2_W_re;
    end

    // ---- retire: post-add, Q1.15 scale (>>>16 = twiddle + per-stage /2) ----
    wire signed [35:0] mult_re36 = $signed({{4{p1[31]}}, p1}) - $signed({{4{p2[31]}}, p2});
    wire signed [35:0] mult_im36 = $signed({{4{p3[31]}}, p3}) + $signed({{4{p4[31]}}, p4});
    wire signed [35:0] t_re36 = mult_re36 >>> 16;
    wire signed [35:0] t_im36 = mult_im36 >>> 16;

    wire signed [15:0] wr_y1_re = v3_is_add ? v3_y1add_re : t_re36[15:0];
    wire signed [15:0] wr_y1_im = v3_is_add ? v3_y1add_im : t_im36[15:0];

    wire retire_wr  = v3_valid;
    wire retire_mem = ~v3_mem_sel;    // butterflies write the opposite memory

    // --------------------------------------------------------
    // Port muxing.
    // Retiring butterfly writes have priority; the controller guarantees
    // (via cmd_valid gating + inter-stage drain) that no read/load targets
    // the same memory in the same cycle.
    // --------------------------------------------------------
    always @* begin
        // defaults: no write, address = whatever read might need
        m0a_we = 1'b0; m0b_we = 1'b0; m1a_we = 1'b0; m1b_we = 1'b0;
        m0a_addr = 8'd0; m0b_addr = 8'd0; m1a_addr = 8'd0; m1b_addr = 8'd0;
        m0a_wre = 16'sd0; m0a_wim = 16'sd0; m0b_wre = 16'sd0; m0b_wim = 16'sd0;
        m1a_wre = 16'sd0; m1a_wim = 16'sd0; m1b_wre = 16'sd0; m1b_wim = 16'sd0;

        // ---- mem0 ----
        if (retire_wr && retire_mem == 1'b0) begin
            m0a_we   = 1'b1;
            m0a_addr = v3_waddr_a;
            m0a_wre  = v3_y0_re;  m0a_wim = v3_y0_im;
            m0b_we   = 1'b1;
            m0b_addr = v3_waddr_b;
            m0b_wre  = wr_y1_re;  m0b_wim = wr_y1_im;
        end
        else begin
            if (is_load && load_mem_sel == 1'b0) begin
                m0a_we   = 1'b1;
                m0a_addr = load_addr;
                m0a_wre  = sext12to16(load_re12);
                m0a_wim  = load_real_only ? 16'sd0 : sext12to16(load_im12);
            end
            else if (is_bfly && mem_sel == 1'b0) begin
                m0a_addr = addr_a;
                m0b_addr = addr_b;
            end
            else if (is_read && read_mem_sel == 1'b0) begin
                m0a_addr = read_addr;
            end
        end

        // ---- mem1 ----
        if (retire_wr && retire_mem == 1'b1) begin
            m1a_we   = 1'b1;
            m1a_addr = v3_waddr_a;
            m1a_wre  = v3_y0_re;  m1a_wim = v3_y0_im;
            m1b_we   = 1'b1;
            m1b_addr = v3_waddr_b;
            m1b_wre  = wr_y1_re;  m1b_wim = wr_y1_im;
        end
        else begin
            if (is_load && load_mem_sel == 1'b1) begin
                m1a_we   = 1'b1;
                m1a_addr = load_addr;
                m1a_wre  = sext12to16(load_re12);
                m1a_wim  = load_real_only ? 16'sd0 : sext12to16(load_im12);
            end
            else if (is_bfly && mem_sel == 1'b1) begin
                m1a_addr = addr_a;
                m1b_addr = addr_b;
            end
            else if (is_read && read_mem_sel == 1'b1) begin
                m1a_addr = read_addr;
            end
        end
    end

    // --------------------------------------------------------
    // READ interface: 1-cycle latency (same contract as before)
    // --------------------------------------------------------
    reg rd_msel_q;
    always @(posedge clk) begin
        if (is_read)
            rd_msel_q <= read_mem_sel;
        read_valid <= is_read;
        done       <= is_load || is_read || v3_valid;   // retire pulse
    end

    assign read_re16 = rd_msel_q ? m1a_rre : m0a_rre;
    assign read_im16 = rd_msel_q ? m1a_rim : m0a_rim;

endmodule
