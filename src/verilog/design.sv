`include "fft_pe.sv"
module fft256_pe_controller_top (
    input  wire        clk,
    input  wire        rst,         // synchronous active-high

    // Control
    input  wire        start,       // start FFT
    output reg         busy,
    output reg         done,

    // Input real samples (256 total), 12-bit
    input  wire signed [11:0] in_re,
    input  wire        in_valid,
    output reg         in_ready,
    input  wire 	   in_last,

    // FFT output stream (final result)
    input reg  	   	   out_ready,
    output reg 		   out_last,
    output reg         out_valid,
    output reg  [8:0]  out_idx,     // bin index 0..257
    output reg  signed [15:0] out_re,

    output reg  signed [15:0] out_im
);

    // -----------------------------------------
    // Instantiate PE
    // -----------------------------------------
    wire signed [15:0] pe_read_re16, pe_read_im16;
    wire               pe_read_valid;
    wire               pe_done_cmd;

    reg                pe_cmd_valid;   // ops enter the PE pipeline only when high
    reg  [1:0]         pe_cmd;
    reg  [2:0]         pe_stage_sel;
    reg                pe_mem_sel;

    reg                pe_load_mem_sel;
    reg  [7:0]         pe_load_addr;
    reg  signed [11:0] pe_load_re12, pe_load_im12;
    reg                pe_load_real_only;

    reg                pe_read_mem_sel;
    reg  [7:0]         pe_read_addr;

    reg  [7:0]         pe_addr_a, pe_addr_b, pe_addr_out_a, pe_addr_out_b;
    reg  [7:0]         pe_twiddle_idx;
    reg                pe_real_compute;

    fft_pe256_mem_trunc u_pe (
        .clk           (clk),
        .cmd_valid     (pe_cmd_valid),
        .cmd           (pe_cmd),
        .stage_sel     (pe_stage_sel),
        .mem_sel       (pe_mem_sel),
        .load_mem_sel  (pe_load_mem_sel),
        .load_addr     (pe_load_addr),
        .load_re12     (pe_load_re12),
        .load_im12     (pe_load_im12),
        .load_real_only(pe_load_real_only),
        .read_mem_sel  (pe_read_mem_sel),
        .read_addr     (pe_read_addr),
        .read_re16     (pe_read_re16),
        .read_im16     (pe_read_im16),
        .read_valid    (pe_read_valid),
        .addr_a        (pe_addr_a),
        .addr_b        (pe_addr_b),
        .addr_out_a    (pe_addr_out_a),
        .addr_out_b    (pe_addr_out_b),
        .twiddle_idx   (pe_twiddle_idx),
        .real_compute  (pe_real_compute),
        .done          (pe_done_cmd)
    );

    // -----------------------------------------
    // Controller FSM
    // -----------------------------------------
    localparam S_IDLE       = 3'd0;
    localparam S_LOAD       = 3'd1;
    localparam S_STAGE_INIT = 3'd2;
    localparam S_STAGE_RUN  = 3'd3;
    localparam S_READOUT    = 3'd4;
    localparam S_DONE       = 3'd5;

    reg [2:0] state, next_state;

    // Load counter (0..256)
    reg [8:0] load_cnt;

    // Stage, start index and k for butterflies
    reg  [2:0] stage;        // 0..7
    reg  [7:0] start_idx;    // outer loop: 0, blk, 2*blk, ...
    reg  [7:0] k_idx;        // 0..half-1

    // Derived quantities for current stage
    wire [7:0] half = (8'd128 >> stage);   // half = N/2^(stage+1)
    wire [8:0] blk  = half*2;         // blk = 2*half

    // Current butterfly indices
    wire [7:0] cur_addr_a = start_idx + k_idx;
    wire [7:0] cur_addr_b = cur_addr_a + half;

    // Twiddle index: k << stage
    wire [7:0] cur_twiddle = (k_idx << stage);

    // End-of-loop flags
    wire last_k     = (k_idx == (half - 1));
  wire last_start = (start_idx + blk >= 8'd255);
 //   reg last_start;

    // Ping-pong memory:
    //   stage even:  read mem0, write mem1 => mem_sel=0
    //   stage odd :  read mem1, write mem0 => mem_sel=1
    wire cur_mem_sel   = stage[0];
    wire final_mem_sel = ~cur_mem_sel; // where final FFT lives after last stage

    // Readout counter (0..256)
    reg [8:0] out_cnt;

    localparam RO_ISSUE   = 2'd0;
    localparam RO_WAIT    = 2'd1;
    localparam RO_PRESENT = 2'd2;
    reg [1:0] ro_st;
    reg [8:0] ro_addr;
    reg [1:0] ro_cnt;
    reg       readout_done;

    // Wait between stages: the PE butterfly writes memory 4 cycles after issue.
    // The controller stays in S_STAGE_INIT until the last pending butterfly
    // finishes, then either starts the next stage or begins readout.
    // This adds the pipeline drain delay between stages.
    localparam [2:0] DRAIN_CYCLES = 3'd6;
    reg [2:0] drain_cnt;
    reg       all_stages_done;   // set after the stage-7 butterflies are issued

    // -----------------------------------------
    // FSM next-state logic
    // -----------------------------------------
    always @(*) begin
        next_state = state;
        case (state)
            S_IDLE: begin
                if (start)
                    next_state = S_LOAD;
            end

            S_LOAD: begin
                if (load_cnt == 9'd256)
                    next_state = S_STAGE_INIT;
            end

            S_STAGE_INIT: begin
                // Hold here until the PE pipeline has drained, then either
                // run the next stage or (after stage 7) start the readout.
                if (drain_cnt == 3'd0)
                    next_state = all_stages_done ? S_READOUT : S_STAGE_RUN;
            end

            S_STAGE_RUN: begin
                if (last_k && last_start)
                    next_state = S_STAGE_INIT;   // always drain first
            end

            S_READOUT:
              begin
                if (readout_done)
                    next_state = S_DONE;
              end

            S_DONE: begin
                if (!start)
                    next_state = S_IDLE;
            end

            default: next_state = S_IDLE;
        endcase
    end

    // -----------------------------------------
    // Sequential part
    // -----------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            state       <= S_IDLE;
            load_cnt    <= 9'd0;
            stage       <= 3'd0;
            start_idx   <= 8'd0;
            k_idx       <= 8'd0;
            out_cnt     <= 9'd0;
            ro_st        <= RO_ISSUE;
            ro_addr      <= 9'd0;
            ro_cnt       <= 2'd0;
            readout_done <= 1'b0;
            drain_cnt       <= 3'd0;
            all_stages_done <= 1'b0;

            busy        <= 1'b0;
            done        <= 1'b0;
            in_ready    <= 1'b0;
            out_valid   <= 1'b0;
            out_idx     <= 8'd0;
            out_re      <= 16'sd0;
            out_im      <= 16'sd0;
            out_last    <= 1'b0;

            pe_cmd_valid      <= 1'b0;
            pe_cmd            <= 2'b00;
            pe_stage_sel      <= 3'd0;
            pe_mem_sel        <= 1'b0;
            pe_load_mem_sel   <= 1'b0;
            pe_load_addr      <= 8'd0;
            pe_load_re12      <= 12'sd0;
            pe_load_im12      <= 12'sd0;
            pe_load_real_only <= 1'b0;
            pe_read_mem_sel   <= 1'b0;
            pe_read_addr      <= 8'd0;
            pe_addr_a         <= 8'd0;
            pe_addr_b         <= 8'd0;
            pe_addr_out_a     <= 8'd0;
            pe_addr_out_b     <= 8'd0;
            pe_twiddle_idx    <= 8'd0;
            pe_real_compute   <= 1'b0;
        end
        else begin
            state <= next_state;

            // Defaults each cycle
            busy      <= (next_state != S_IDLE && next_state != S_DONE);
            done      <= 1'b0;
            in_ready  <= 1'b0;

            // default: no command enters the PE pipeline this cycle
            pe_cmd_valid      <= 1'b0;
            pe_cmd            <= 2'b01;
            pe_stage_sel      <= stage;
            pe_mem_sel        <= cur_mem_sel;
            pe_load_mem_sel   <= 1'b0;
            pe_load_addr      <= 8'd0;
            pe_load_re12      <= 12'sd0;
            pe_load_im12      <= 12'sd0;
            pe_load_real_only <= 1'b0;
            pe_read_mem_sel   <= 1'b0;
            pe_read_addr      <= 8'd0;
            pe_addr_a         <= 8'd0;
            pe_addr_b         <= 8'd0;
            pe_addr_out_a     <= 8'd0;
            pe_addr_out_b     <= 8'd0;
            pe_twiddle_idx    <= 8'd0;
            pe_real_compute   <= 1'b0;


            case (state)
                // ---------------------------------------------
                S_IDLE: begin
                    load_cnt  <= 9'd0;
                    stage     <= 3'd0;
                    start_idx <= 8'd0;
                    k_idx     <= 8'd0;
                    out_cnt   <= 9'd0;
                    ro_st        <= RO_ISSUE;
                    ro_addr      <= 9'd0;
                    ro_cnt       <= 2'd0;
                    readout_done <= 1'b0;
                    drain_cnt       <= 3'd0;
                    all_stages_done <= 1'b0;
                    if (next_state == S_LOAD)
                        busy <= 1'b1;
                end

                // ---------------------------------------------
                // LOAD 256 real samples into mem0
                // ---------------------------------------------
                S_LOAD:
                  begin
                  if (load_cnt < 9'd256)
                    begin
                      in_ready <= 1'b1;
                      if (in_valid && in_ready)
                          begin
                            pe_cmd_valid      <= 1'b1;
                            pe_cmd            <= 2'b00;    // LOAD
                            pe_load_mem_sel   <= 1'b0;     // mem0
                            pe_load_addr      <= load_cnt[7:0];
                            pe_load_re12      <= in_re;
                            pe_load_im12      <= 12'sd0;
                            pe_load_real_only <= 1'b1;     // real-only
                            load_cnt          <= load_cnt + 9'd1;
                        end
                    end
                    if (next_state == S_STAGE_INIT)
                      begin
                        stage     <= 3'd0;
                        start_idx <= 8'd0;
                        k_idx     <= 8'd0;
                        in_ready <= 1'b0;
                    end
                end

                // ---------------------------------------------
                // Init stage: reset start_idx, k_idx and let the PE
                // pipeline drain (no ops are issued while we sit here)
                // ---------------------------------------------
                S_STAGE_INIT: begin
                    start_idx <= 8'd0;
                    k_idx     <= 8'd0;
                    if (drain_cnt != 3'd0)
                        drain_cnt <= drain_cnt - 3'd1;
                end

                // ---------------------------------------------
                // Run butterflies for this stage
                // ---------------------------------------------
                S_STAGE_RUN:
                  begin
                    // Issue one butterfly into the PE pipeline every cycle
                    pe_cmd_valid   <= 1'b1;
                    pe_mem_sel     <= cur_mem_sel;
                    pe_stage_sel   <= stage;
                    pe_addr_a      <= cur_addr_a;
                    pe_addr_b      <= cur_addr_b;
                    pe_addr_out_a  <= cur_addr_a;
                    pe_addr_out_b  <= cur_addr_b;
                    pe_twiddle_idx <= cur_twiddle;

                    // Stage 0: real-only multiply
                    pe_real_compute <= (stage == 3'd0);

                 //   Last stage uses ADD-only (twiddle = 1)
                     if (stage == 3'd7)
                         pe_cmd <= 2'b10; // BFLY_ADD
                     else
                        pe_cmd <= 2'b11; // BFLY_MUL

                    // Update counters
                    if (last_k) begin
                        k_idx <= 8'd0;
                        if (last_start) begin
                            // Finished issuing this stage; drain the PE
                            // pipeline in S_STAGE_INIT before the next one.
                            drain_cnt <= DRAIN_CYCLES;
                            start_idx <= 8'd0;
                            if (stage == 3'd7)
                                all_stages_done <= 1'b1;  // -> readout after drain
                            else
                                stage <= stage + 3'd1;
                        end
                        else begin
                            start_idx <= start_idx + blk;
                        end
                    end
                    else begin
                        k_idx <= k_idx + 8'd1;
                    end
                  end

                // ---------------------------------------------
                // READOUT: stream 256 FFT bins out
                // Final results are in opposite memory of last read stage.
                // For stage=7, cur_mem_sel=1 => final_mem_sel=0 => mem0
                // ---------------------------------------------
                S_READOUT:
                  begin
                    case (ro_st)
                      RO_ISSUE:
                        begin
                          pe_cmd_valid    <= 1'b1;
                          pe_cmd          <= 2'b01;
                          pe_read_mem_sel <= final_mem_sel;
                          pe_read_addr    <= ro_addr[7:0];
                          ro_cnt          <= 2'd1;
                          ro_st           <= RO_WAIT;
                        end
                      RO_WAIT:
                        begin
                          pe_cmd_valid    <= 1'b1;
                          pe_cmd          <= 2'b01;
                          pe_read_mem_sel <= final_mem_sel;
                          pe_read_addr    <= ro_addr[7:0];
                          if (ro_cnt != 2'd0)
                            ro_cnt <= ro_cnt - 2'd1;
                          else
                            begin
                              out_idx   <= ro_addr;
                              out_re    <= pe_read_re16;
                              out_im    <= pe_read_im16;
                              out_valid <= 1'b1;
                              out_last  <= (ro_addr == 9'd255);
                              ro_st     <= RO_PRESENT;
                            end
                        end
                      RO_PRESENT:
                        begin
                          if (out_valid && out_ready)
                            begin
                              out_valid <= 1'b0;
                              out_last  <= 1'b0;
                              if (ro_addr == 9'd255)
                                readout_done <= 1'b1;
                              else
                                begin
                                  ro_addr <= ro_addr + 9'd1;
                                  ro_st   <= RO_ISSUE;
                                end
                            end
                        end
                      default: ro_st <= RO_ISSUE;
                    endcase
                  end

                // ---------------------------------------------
                S_DONE: begin
                    done <= 1'b1; // hold until start deasserts
                end

                default: ;
            endcase
        end
    end

endmodule
