// Code your testbench here
// or browse Examples
`timescale 1ns/1ps

module tb_fft256;

    // ----------------------------------------------------
    // Clock & reset
    // ----------------------------------------------------
    reg clk;
    reg rst;

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;   // 100 MHz clock
    end

    // ----------------------------------------------------
    // DUT I/O
    // ----------------------------------------------------
    reg         start;
    wire        busy;
    wire        done;

    reg  signed [11:0] in_re;
    reg         in_valid;
    wire        in_ready;

    wire        out_valid;
    wire [8:0]  out_idx;
    wire signed [15:0] out_re;
    wire signed [15:0] out_im;
    reg in_last;
    reg out_ready;
    wire out_last;

    // ----------------------------------------------------
    // Instantiate top-level FFT controller + PE
    // ----------------------------------------------------
    fft256_pe_controller_top dut (
        .clk      (clk),
        .rst      (rst),
        .start    (start),
        .busy     (busy),
        .done     (done),
        .in_re    (in_re),
        .in_valid (in_valid),
        .in_ready (in_ready),
        .out_valid(out_valid),
        .out_idx  (out_idx),
        .out_re   (out_re),
      .out_im   (out_im),
      .in_last (in_last),
      .out_ready(out_ready),
      .out_last(out_last)
    );

    // ----------------------------------------------------
    // Stimulus: 256-point sine wave at bin kbin
    // Quantization: Q1.11 (12-bit signed)
    // x[n] = 0.9 * sin(2π*kbin*n/N), N = 256
    // ----------------------------------------------------
    localparam integer N    = 256;
    localparam integer KBIN = 13;

    reg signed [11:0] sine_mem [0:N-1];
    integer i;
    real    pi;
    real    amp;
    real    theta;
    real    val;
    integer q;

    initial 
      begin
        pi  = 3.141592653589793;
        amp = 0.9;  // 90% of full-scale

        for (i = 0; i < N; i = i + 1) begin
            theta = 2.0 * pi * KBIN * i / N;
            val   = amp * $sin(theta);    // in [-0.9, 0.9]
          
          
           // $display("inside sine generation loop", $time);

            // Q1.11 scaling: 1 LSB = 1/2048
            q = $rtoi(val * 2048.0);      // ideally in [-1843, 1843]

            // Saturate to 12-bit signed range [-2048, 2047]
            if (q >  2047) q =  2047;
            if (q < -2048) q = -2048;
         // $display("n=%0d : q=%0d ", i, q);
            //$display("inside sine generation loop", $time);

            sine_mem[i] = q[11:0];
          //$display("n=%0d : mem=%0d ", i, sine_mem[i]);
        end
    end

    // ----------------------------------------------------
    // Input driver: send 256 samples when in_ready=1
    // ----------------------------------------------------
    integer in_idx;
    reg     feeding;

    initial begin
        in_idx   = 0;
        in_re    = 12'sd0;
        in_valid = 1'b0;
    end

    always @(posedge clk) begin
        if (rst) begin
            in_idx   <= 0;
            feeding  <= 0;
            in_valid <= 1'b0;
            in_re    <= 12'sd0;
            in_last <= 1'b0;
            out_ready <= 1'b0;
        end
        else begin
            if (!feeding && start && busy) begin
                // Just started LOAD phase
                feeding <= 1'b1;
            end

            if (feeding && in_idx < N) begin
                if (in_ready) begin
                    // Present next sample when controller is ready
                  	in_last <= (in_idx == (N-1));
                    in_valid <= 1'b1;
                    in_re    <= sine_mem[in_idx];
                  if(in_idx==0) $display("entered the loop of writing sine");
            //      $display("n=%0d : inidx=%0d ", in_idx, sine_mem[in_idx]);
                    in_idx   <= in_idx + 1;
                end
                else 
                  begin
                    in_valid <= 1'b0;  // wait for in_ready
                  end
            end
            else begin
                // Finished sending all samples
                in_valid <= 1'b0;
              if(in_idx >= N && !in_valid)
                	out_ready <= 1'b1;
            end 
        end
    end

    // ----------------------------------------------------
    // Output capture: write FFT bins to a file
    // Format per line: index  {out_im[15:0], out_re[15:0]}  (32-bit packed)
    // ----------------------------------------------------
    integer fh;
    initial begin
        fh = $fopen("fft_out.txt", "w");
        if (fh == 0)
          begin
            $display("ERROR: Could not open fft_out.txt for writing");
            $finish;
        end
    end

  always @(posedge clk)
      begin
        if (!rst && out_valid)
          begin
            $fwrite(fh, "%d %d\n", out_idx,{out_im,out_re});
          end
      end

    // ----------------------------------------------------
    // Main control / reset / start sequence
    // ----------------------------------------------------
    initial begin
        // VCD dump
        $dumpfile("fft256.vcd");
        $dumpvars(0, tb_fft256);

        // Reset
        rst   = 1'b1;
        start = 1'b0;
        #50;
        rst = 1'b0;
        #25;

        // Start FFT
        start = 1'b1;

        // Wait until done (controller asserts done in S_DONE)
        wait(done);
        $display("FFT computation completed at time %0t", $time);

        // Allow a few cycles for last outputs to flush
      #2000;

        $fclose(fh);
        $display("FFT results written to fft_out.txt");
        $finish;
    end

endmodule
