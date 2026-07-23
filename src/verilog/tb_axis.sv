`timescale 1ns/1ps

// AXI4-Stream corner-case verification for fft256_pe_controller_top.
// Maps: in_re/in_valid/in_ready/in_last  = slave  (S_AXIS: TDATA/TVALID/TREADY/TLAST)
//       out_*/out_valid/out_ready/out_last = master (M_AXIS)
// Runs three scenarios against a golden (no-stall) reference and checks
// data integrity plus AXI-Stream handshake rules.

module tb_axis;

    // ---------------- clock / reset ----------------
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst;

    // ---------------- DUT I/O ----------------
    reg               start;
    wire              busy, done;
    reg  signed [11:0] in_re;
    reg               in_valid;
    wire              in_ready;
    reg               in_last;
    wire              out_valid;
    wire [8:0]        out_idx;
    wire signed [15:0] out_re, out_im;
    reg               out_ready;
    wire              out_last;

    fft256_pe_controller_top dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .in_re(in_re), .in_valid(in_valid), .in_ready(in_ready), .in_last(in_last),
        .out_ready(out_ready), .out_last(out_last), .out_valid(out_valid),
        .out_idx(out_idx), .out_re(out_re), .out_im(out_im)
    );

    // ---------------- stimulus samples ----------------
    localparam integer N = 256, KBIN = 13;
    reg signed [11:0] sine [0:N-1];
    integer si; real pi, amp, th; integer qq;
    initial begin
        pi = 3.141592653589793; amp = 0.9;
        for (si = 0; si < N; si = si + 1) begin
            th = 2.0*pi*KBIN*si/N;
            qq = $rtoi(amp*$sin(th)*2048.0);
            if (qq >  2047) qq =  2047;
            if (qq < -2048) qq = -2048;
            sine[si] = qq[11:0];
        end
    end

    // ---------------- scenario control ----------------
    reg     stall_in;      // 1 => AXI-compliant stalling master on the input
    reg     bp_out;        // 1 => random backpressure on the output
    reg     collecting;
    reg     feeding;
    integer in_idx;
    integer stall_cnt;
    reg [15:0] in_lfsr, out_lfsr;

    // ---------------- scoreboard ----------------
    reg signed [15:0] gold_re [0:N-1];
    reg signed [15:0] gold_im [0:N-1];
    reg signed [15:0] cap_re  [0:N-1];
    reg signed [15:0] cap_im  [0:N-1];
    reg               seen    [0:N-1];
    integer recv_count;
    integer valid_drop_viol;   // TVALID deasserted before handshake
    integer data_change_viol;  // TDATA/idx changed while stalled
    integer dup_viol;          // same bin delivered twice
    reg               tlast_on_last;  // was out_last high on the bin-255 beat?
    integer tlast_stray;       // out_last high while out_valid low

    integer k;

    // ================= INPUT master BFM =================
    always @(posedge clk) begin
        if (rst) begin
            in_valid  <= 1'b0;
            in_re     <= 12'sd0;
            in_last   <= 1'b0;
            in_idx    <= 0;
            feeding   <= 1'b0;
            stall_cnt <= 0;
            in_lfsr   <= 16'hACE1;
        end else begin
            in_lfsr <= {in_lfsr[14:0],
                        in_lfsr[15]^in_lfsr[13]^in_lfsr[12]^in_lfsr[10]};

            if (!feeding && start && busy)
                feeding <= 1'b1;

            if (feeding && in_idx < N) begin
                if (!stall_in) begin
                    // Legacy master: present a new sample every ready cycle.
                    if (in_ready) begin
                        in_valid <= 1'b1;
                        in_re    <= sine[in_idx];
                        in_last  <= (in_idx == N-1);
                        in_idx   <= in_idx + 1;
                    end else begin
                        in_valid <= 1'b0;
                    end
                end else begin
                    // AXI-compliant master: hold TVALID+TDATA until handshake,
                    // insert random idle (TVALID low) gaps between beats.
                    if (stall_cnt > 0) begin
                        in_valid  <= 1'b0;
                        stall_cnt <= stall_cnt - 1;
                    end else begin
                        in_valid <= 1'b1;
                        in_re    <= sine[in_idx];
                        in_last  <= (in_idx == N-1);
                        if (in_valid && in_ready) begin
                            in_idx    <= in_idx + 1;
                            in_valid  <= 1'b0;
                            stall_cnt <= in_lfsr[1:0];   // 0..3 idle cycles
                        end
                    end
                end
            end else begin
                in_valid <= 1'b0;
            end
        end
    end

    // ================= OUTPUT slave BFM + protocol monitor =================
    reg               p_valid, p_ready;
    reg signed [15:0] p_re, p_im;
    reg [8:0]         p_idx;

    always @(posedge clk) begin
        if (rst) begin
            out_ready <= 1'b0;
            out_lfsr  <= 16'hBEEF;
            p_valid <= 1'b0; p_ready <= 1'b0;
        end else begin
            out_lfsr <= {out_lfsr[14:0],
                         out_lfsr[15]^out_lfsr[13]^out_lfsr[12]^out_lfsr[10]};

            // Drive TREADY: full-rate or random backpressure.
            if (collecting) out_ready <= bp_out ? out_lfsr[0] : 1'b1;
            else            out_ready <= 1'b0;

            // Capture on a real handshake (TVALID && TREADY).
            if (out_valid && out_ready) begin
                if (seen[out_idx]) dup_viol = dup_viol + 1;
                seen[out_idx]   <= 1'b1;
                cap_re[out_idx] <= out_re;
                cap_im[out_idx] <= out_im;
                recv_count      <= recv_count + 1;
                if (out_idx == (N-1)) tlast_on_last <= out_last;
            end

            // AXI-S rule: once TVALID is high it must stay high, with stable
            // payload, until TREADY completes the beat.
            if (collecting && p_valid && !p_ready) begin
                if (!out_valid)
                    valid_drop_viol = valid_drop_viol + 1;
                else if (out_re !== p_re || out_im !== p_im || out_idx !== p_idx)
                    data_change_viol = data_change_viol + 1;
            end

            // TLAST must never be asserted without TVALID.
            if (out_last && !out_valid)
                tlast_stray = tlast_stray + 1;

            p_valid <= out_valid; p_ready <= out_ready;
            p_re <= out_re; p_im <= out_im; p_idx <= out_idx;
        end
    end

    // ================= scenario runner =================
    task run_scenario(input do_stall_in, input do_bp_out);
        begin
            stall_in = do_stall_in;
            bp_out   = do_bp_out;
            collecting = 1'b0;
            for (k = 0; k < N; k = k + 1) seen[k] = 1'b0;
            recv_count       = 0;
            valid_drop_viol  = 0;
            data_change_viol = 0;
            dup_viol         = 0;
            tlast_stray      = 0;
            tlast_on_last    = 1'b0;

            rst = 1'b1; start = 1'b0;
            repeat (5) @(posedge clk);
            rst = 1'b0;
            @(posedge clk);
            collecting = 1'b1;      // TREADY driver + monitor active for the whole run
            start = 1'b1;

            wait (done);
            repeat (60) @(posedge clk);
            start = 1'b0;
            wait (!busy);
            repeat (5) @(posedge clk);
            collecting = 1'b0;
        end
    endtask

    function integer mismatches_vs_gold;
        integer c, m;
        begin
            m = 0;
            for (c = 0; c < N; c = c + 1) begin
                if (!seen[c])
                    m = m + 1;
                else if (cap_re[c] !== gold_re[c] || cap_im[c] !== gold_im[c])
                    m = m + 1;
            end
            mismatches_vs_gold = m;
        end
    endfunction

    integer mB, mC;
    initial begin
        $dumpfile("axis.vcd");
        $dumpvars(0, tb_axis);

        // ---- A: golden reference (no stalls, no backpressure) ----
        run_scenario(1'b0, 1'b0);
        for (k = 0; k < N; k = k + 1) begin
            gold_re[k] = cap_re[k];
            gold_im[k] = cap_im[k];
        end
        $display("AXIS| A golden        : recv=%0d/256  dup=%0d", recv_count, dup_viol);
        $display("AXIS| A TLAST-on-bin255=%0b  stray-TLAST=%0d", tlast_on_last, tlast_stray);

        // ---- B: output backpressure (random TREADY) ----
        run_scenario(1'b0, 1'b1);
        mB = mismatches_vs_gold();
        $display("AXIS| B out-backpress : recv=%0d/256  dataMismatch=%0d  dup=%0d",
                 recv_count, mB, dup_viol);
        $display("AXIS|   validDrops=%0d  dataChangesWhileStalled=%0d",
                 valid_drop_viol, data_change_viol);

        // ---- C: input stalls (AXI-compliant stalling master) ----
        run_scenario(1'b1, 1'b0);
        mC = mismatches_vs_gold();
        $display("AXIS| C in-stall      : recv=%0d/256  dataMismatch=%0d", recv_count, mC);

        $display("AXIS| ================ VERDICT ================");
        $display("AXIS| output backpressure : %s", (mB==0 && valid_drop_viol==0 && data_change_viol==0) ? "PASS" : "FAIL");
        $display("AXIS| input stalling      : %s", (mC==0) ? "PASS" : "FAIL");
        $display("AXIS| TLAST on last beat  : %s", (tlast_on_last==1'b1 && tlast_stray==0) ? "PASS" : "FAIL");
        $finish;
    end

endmodule
