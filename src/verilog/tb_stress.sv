`timescale 1ns/1ps

module tb_stress;
    localparam integer N = 256;

    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst;

    reg                start;
    wire               busy, done;
    reg  signed [11:0] in_re;
    reg                in_valid;
    wire               in_ready;
    reg                in_last;
    wire               out_valid;
    wire [8:0]         out_idx;
    wire signed [15:0] out_re, out_im;
    reg                out_ready;
    wire               out_last;

    fft256_pe_controller_top dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .in_re(in_re), .in_valid(in_valid), .in_ready(in_ready), .in_last(in_last),
        .out_ready(out_ready), .out_last(out_last), .out_valid(out_valid),
        .out_idx(out_idx), .out_re(out_re), .out_im(out_im)
    );

    reg signed [11:0] sine   [0:N-1];
    reg signed [15:0] cap_re [0:N-1];
    reg signed [15:0] cap_im [0:N-1];
    reg               seen   [0:N-1];

    reg               collecting;
    integer           recv, last_cnt, stray_last;

    integer i, k, j;
    real    pi, th;
    integer qq;
    integer nframes, f, argmax, miss, ferr, fails, progress_every, tone0, tval;
    reg [63:0] p, pmax;

    always @(posedge clk) begin
        if (rst) begin
            out_ready  <= 1'b0;
            recv       <= 0;
            last_cnt   <= 0;
            stray_last <= 0;
        end else begin
            out_ready <= collecting ? 1'b1 : 1'b0;
            if (collecting && out_valid && out_ready) begin
                cap_re[out_idx] <= out_re;
                cap_im[out_idx] <= out_im;
                seen[out_idx]   <= 1'b1;
                recv            <= recv + 1;
                if (out_last) last_cnt <= last_cnt + 1;
            end
            if (out_last && !out_valid) stray_last <= stray_last + 1;
        end
    end

    task build_sine(input integer kbin);
        begin
            pi = 3.141592653589793;
            for (i = 0; i < N; i = i + 1) begin
                th = 2.0*pi*kbin*i/N;
                qq = $rtoi(0.9*$sin(th)*2048.0);
                if (qq >  2047) qq =  2047;
                if (qq < -2048) qq = -2048;
                sine[i] = qq[11:0];
            end
        end
    endtask

    task run_frame(input integer kbin);
        begin
            build_sine(kbin);
            for (k = 0; k < N; k = k + 1) seen[k] = 1'b0;
            recv = 0; last_cnt = 0; stray_last = 0;

            @(posedge clk);
            collecting = 1'b1;
            start      = 1'b1;
            @(posedge clk);
            wait (busy);

            @(negedge clk);
            while (!in_ready) @(negedge clk);
            for (j = 0; j < N; j = j + 1) begin
                in_valid = 1'b1;
                in_re    = sine[j];
                in_last  = (j == N-1);
                @(negedge clk);
            end
            in_valid = 1'b0;
            in_last  = 1'b0;

            wait (done);
            repeat (5) @(posedge clk);
            start = 1'b0;
            wait (!busy);
            repeat (3) @(posedge clk);
            collecting = 1'b0;
            @(posedge clk);
        end
    endtask

    task check_frame(input integer kbin, input integer fnum);
        begin
            pmax = 0; argmax = 0; miss = 0;
            for (k = 1; k < N; k = k + 1) begin
                p = $signed(cap_re[k])*$signed(cap_re[k])
                  + $signed(cap_im[k])*$signed(cap_im[k]);
                if (p > pmax) begin pmax = p; argmax = k; end
            end
            for (k = 0; k < N; k = k + 1) if (!seen[k]) miss = miss + 1;
            ferr = 0;
            if (recv !== N)                              ferr = ferr + 1;
            if (miss !== 0)                              ferr = ferr + 1;
            if (last_cnt !== 1)                          ferr = ferr + 1;
            if (stray_last !== 0)                        ferr = ferr + 1;
            if (argmax !== kbin && argmax !== (N-kbin))  ferr = ferr + 1;
            if (ferr !== 0) begin
                fails = fails + 1;
                $display("STRESS| FAIL frame=%0d tone=%0d recv=%0d miss=%0d peak=%0d(exp %0d/%0d) tlast=%0d stray=%0d",
                         fnum, kbin, recv, miss, argmax, kbin, N-kbin, last_cnt, stray_last);
            end
        end
    endtask

    initial begin
        if (!$value$plusargs("frames=%d", nframes)) nframes = 1000000;
        if (!$value$plusargs("progress=%d", progress_every)) progress_every = 1000;
        if (!$value$plusargs("tone0=%d", tone0)) tone0 = 1;

        start = 1'b0; in_valid = 1'b0; in_re = 12'sd0; in_last = 1'b0;
        collecting = 1'b0;
        rst = 1'b1;
        repeat (5) @(posedge clk);
        rst = 1'b0;
        @(posedge clk);

        fails = 0;
        $display("STRESS| streaming %0d frames ...", nframes);
        for (f = 0; f < nframes; f = f + 1) begin
            tval = tone0 + (f % 125);
            run_frame(tval);
            check_frame(tval, f);
            if (((f + 1) % progress_every) == 0)
                $display("STRESS| progress %0d/%0d frames, fails=%0d", f+1, nframes, fails);
        end

        $display("STRESS| ================ VERDICT ================");
        $display("STRESS| %0d frames, fails=%0d -> %s",
                 nframes, fails, (fails==0) ? "PASS" : "FAIL");
        $finish;
    end

endmodule
