`include "twiddle_factors.sv"
module fft_pe256_mem_trunc (
    input  wire        clk,

    // Command
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
    output reg  signed [15:0] read_re18,
    output reg  signed [15:0] read_im18,
    output reg         read_valid,

    // Butterfly addresses
    input  wire [7:0]  addr_a,
    input  wire [7:0]  addr_b,
    input  wire [7:0]  addr_out_a,
    input  wire [7:0]  addr_out_b,

    // Twiddle index (0..255, Q1.15 in ROM)
    input  wire [7:0]  twiddle_idx,

    // If 1, imag(b) is forced to 0 in the multiply math
    input  wire        real_compute,

    // Operation done (one-cycle pulse)
    output reg         done
);
    // --------------------------------------------------------
    // Internal dual memories: mem0 and mem1, each 256 x 16-bit complex
    // --------------------------------------------------------
    reg signed [15:0] mem0_re [0:255];
    reg signed [15:0] mem0_im [0:255];
    reg signed [15:0] mem1_re [0:255];
    reg signed [15:0] mem1_im [0:255];
 	reg [7:0] rev_addr_out_a;
  	reg [7:0] rev_addr_out_b;
    reg [4:0] stage_frac;
  
  assign rev_addr_out_a = { addr_out_a[0], addr_out_a[1], addr_out_a[2], addr_out_a[3],
                           addr_out_a[4], addr_out_a[5], addr_out_a[6], addr_out_a[7] };
  assign rev_addr_out_b = { addr_out_b[0], addr_out_b[1], addr_out_b[2], addr_out_b[3],
                           addr_out_b[4], addr_out_b[5], addr_out_b[6], addr_out_b[7] };
  
  

    // --------------------------------------------------------
    // Twiddle ROM: 16-bit signed Q1.15 (W = W_re + j W_im)
    // Replace this with your full ROM definition.
    // --------------------------------------------------------
    wire signed [15:0] W_re15, W_im15;
    twiddle_rom_256_16 U_TW_1 (
        .addr(twiddle_idx),
        .re(W_re15),
        .im(W_im15)
    );

    // --------------------------------------------------------
    // Helpers: sign-extension & truncation
    // --------------------------------------------------------

    // 12 -> 16 sign-extend
    function signed [15:0] sext12to18;
        input signed [11:0] x;
        begin
            sext12to18 = { {4{x[11]}}, x };
        end
    endfunction

    // 16 -> 36 sign-extend
    function signed [35:0] sx18;
        input signed [15:0] x;
        begin
            sx18 = { {20{x[15]}}, x };
        end
    endfunction

    // 36 -> 16: keep low 16 bits (TRUNCATION ONLY)
    function signed [15:0] trunc18;
        input signed [35:0] x;
        begin
            trunc18 = x[15:0];
        end
    endfunction

    // Arithmetic >> 15 (for Q1.15 twiddle scaling), truncation
    function signed [35:0] arshift15;
        input signed [35:0] x;
        begin
            arshift15 = x >>> stage_frac;
        end
    endfunction

    // --------------------------------------------------------
    // Internal regs for butterfly
    // --------------------------------------------------------
    reg signed [15:0] a_re, a_im, b_re, b_im;
    reg signed [15:0] b_im_eff;
    reg signed [15:0] y0_re, y0_im, y1_re, y1_im;


    reg signed [31:0] p1, p2, p3, p4;
    reg signed [35:0] mult_re36, mult_im36;
    reg signed [35:0] t_re36, t_im36;
    reg signed [15:0] t_re18, t_im18;
  
    always @*
      begin
        case(stage_sel)
          3'b000: stage_frac = 5'd16;
          3'b001: stage_frac = 5'd16;
          3'b010: stage_frac = 5'd16;
          3'b011: stage_frac = 5'd16;
          3'b100: stage_frac = 5'd16;
          3'b101: stage_frac = 5'd16;
          3'b110: stage_frac = 5'd16;
          3'b111: stage_frac = 5'd16;
        endcase
      end

    // --------------------------------------------------------
    // Single-cycle command execution
    // --------------------------------------------------------
    always @(posedge clk) 
      begin
        // default outputs each cycle
        done       <= 1'b0;
        read_valid <= 1'b0;
       

        case (cmd)
            // --------------------------------------------
            // 00: LOAD 12-bit external into mem0/mem1
            // --------------------------------------------
            2'b00: 
              begin
                if (load_mem_sel == 1'b0) 
                  begin
                    mem0_re[load_addr] <= sext12to18(load_re12);
                    mem0_im[load_addr] <= (load_real_only) ? 16'sd0
                                                           : sext12to18(load_im12);
                end 
              else 
                begin
                    mem1_re[load_addr] <= sext12to18(load_re12);
                    mem1_im[load_addr] <= (load_real_only) ? 16'sd0
                                                           : sext12to18(load_im12);
                end
                done <= 1'b1;
            end

            // --------------------------------------------
            // 01: READ 18-bit out from mem0/mem1
            // --------------------------------------------
            2'b01: 
              begin
                if (read_mem_sel == 1'b0) 
                  begin
                    read_re18 <= mem0_re[read_addr];
                    read_im18 <= mem0_im[read_addr];
                   end 
                else 
                  begin
                    read_re18 <= mem1_re[read_addr];
                    read_im18 <= mem1_im[read_addr];
                  end
                read_valid <= 1'b1;
                done       <= 1'b1;
            end

            // --------------------------------------------
            // 10: BFLY_ADD
            //      y0 = a + b
            //      y1 = a - b
            //      read from mem_sel, write to the other
            // --------------------------------------------
            2'b10: begin
                // Fetch operands
                if (mem_sel == 1'b0) 
                  begin
                    a_re = mem0_re[addr_a]; a_im = mem0_im[addr_a];
                    b_re = mem0_re[addr_b]; b_im = mem0_im[addr_b];
                  end 
                else 
                  begin
                    a_re = mem1_re[addr_a]; a_im = mem1_im[addr_a];
                    b_re = mem1_re[addr_b]; b_im = mem1_im[addr_b];
                  end

                // Butterfly add/sub with truncation
                y0_re = trunc18( sx18(a_re) + sx18(b_re) );
                y0_im = trunc18( sx18(a_im) + sx18(b_im) );
                y1_re = trunc18( sx18(a_re) - sx18(b_re) );
                y1_im = trunc18( sx18(a_im) - sx18(b_im) );
              
              
                // Write to the opposite memory
                if (mem_sel == 1'b0) begin
                  mem1_re[rev_addr_out_a] <= y0_re; mem1_im[rev_addr_out_a] <= y0_im;
                  mem1_re[rev_addr_out_b] <= y1_re; mem1_im[rev_addr_out_b] <= y1_im;
                end else begin
                  mem0_re[rev_addr_out_a] <= y0_re; mem0_im[rev_addr_out_a] <= y0_im;
                  mem0_re[rev_addr_out_b] <= y1_re; mem0_im[rev_addr_out_b] <= y1_im;
                end

                done <= 1'b1;
            end

            // --------------------------------------------
            // 11: BFLY_MUL with twiddle:
            //      t = b * W
            //      y0 = a + t
            //      y1 = a - t
            // --------------------------------------------
            2'b11: begin
                // Fetch operands
                if (mem_sel == 1'b0) begin
                    a_re = mem0_re[addr_a]; a_im = mem0_im[addr_a];
                    b_re = mem0_re[addr_b]; b_im = mem0_im[addr_b];
                end else begin
                    a_re = mem1_re[addr_a]; a_im = mem1_im[addr_a];
                    b_re = mem1_re[addr_b]; b_im = mem1_im[addr_b];
                end

                // Optionally ignore imag(b) in multiply
                b_im_eff = (real_compute) ? 16'sd0 : b_im;
              
              y0_re = trunc18( sx18(a_re) + sx18(b_re) ) >>> 1;
              y0_im = trunc18( sx18(a_im) + sx18(b_im) ) >>> 1;
              t_re18 = trunc18( sx18(a_re) - sx18(b_re) );
              t_im18 = trunc18( sx18(a_im) - sx18(b_im) );
              
              p1 = $signed(t_re18)     * $signed(W_re15);
              p2 = $signed(t_im18) * $signed(W_im15);
              p3 = $signed(t_re18)     * $signed(W_im15);
              p4 = $signed(t_im18) * $signed(W_re15);
              
               mult_re36 = $signed({{4{p1[31]}},p1}) -
                            $signed({{4{p2[31]}},p2});
                mult_im36 = $signed({{4{p3[31]}},p3}) +
                            $signed({{4{p4[31]}},p4});

                // Scale by 2^-15 (Q1.15) with truncation
                t_re36 = arshift15(mult_re36);
                t_im36 = arshift15(mult_im36);
        
              y1_re = trunc18(t_re36);
              y1_im = trunc18(t_im36);
              
//             $display("a_re = %d, b_re = %d, a_im = %d, b_im = %d,y0_re=%d, y0_im=%d, y1_re=%d, y1_im=%d , addr_out_a = %d, addr_out_b = %d",a_re, b_re, a_im, b_im, y0_re,y0_im,y1_re,y1_im,addr_out_a,addr_out_b);
// $display("y0_re=%d, y0_im=%d, y1_re=%d, y1_im=%d , addr_out_a = %d, addr_out_b = %d",mem1_re[addr_out_a],mem1_im[addr_out_a],mem1_re[addr_out_b],mem1_im[addr_out_b],addr_out_a,addr_out_b);

                // Write to opposite memory
                if (mem_sel == 1'b0) begin
                    mem1_re[addr_out_a] <= y0_re; mem1_im[addr_out_a] <= y0_im;
                    mem1_re[addr_out_b] <= y1_re; mem1_im[addr_out_b] <= y1_im;
                end else begin
                    mem0_re[addr_out_a] <= y0_re; mem0_im[addr_out_a] <= y0_im;
                    mem0_re[addr_out_b] <= y1_re; mem0_im[addr_out_b] <= y1_im;
                end
              
              //if(stage_sel == 3'b011)
                //begin
                  //if(mem_sel == 1'b0)
                  //  $display("Stage  = %0d\n , mem_sel = %0d, mem0_re = %p\n\n",stage_idx,mem_sel,mem0_re);
                 // else
                    // $display("Stage  = %0d\n , mem_sel = %0d, mem0_re = %p\n\n",stage_idx,mem_sel,mem0_re);
                // end

               done <= 1'b1;
              
               

            end

            default: begin
                // no-op
            end
        endcase
    end
endmodule
