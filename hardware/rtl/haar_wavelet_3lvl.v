// WaveBNN-ECG: 3-Level Haar Wavelet (add/sub only, zero multipliers)
// 187 samples -> cA3(24), cD3(24), cD2(47), cD1(94)

module haar_wavelet_3lvl (
    input  wire        clk,
    input  wire        rst_n,

    // ── Input: stream 187 samples ──
    input  wire        i_start,          // pulse: begin accepting samples
    input  wire signed [7:0]  i_sample,  // 8-bit signed ECG sample
    input  wire        i_sample_valid,   // sample strobe

    // ── Status ──
    output reg         o_busy,           // 1 while ingesting / computing
    output reg         o_done,           // 1-cycle pulse when sub-bands ready

    // ── Sub-band read port ──
    input  wire [6:0]  i_rd_addr,        // read address (0-93 for cD1, 0-46 for cD2, etc.)
    output wire signed [9:0]  o_cD1,     // 10-bit detail level 1
    output wire signed [10:0] o_cD2,     // 11-bit detail level 2
    output wire signed [11:0] o_cD3,     // 12-bit detail level 3
    output wire signed [11:0] o_cA3      // 12-bit approx level 3
);

    // Parameters
    localparam N_SAMPLES = 187;
    localparam N_PAD1    = 188;  // padded for level 1
    localparam N_CA1     = 94;   // cA1 / cD1 length
    localparam N_CA2     = 47;   // cA2 / cD2 length
    localparam N_PAD3    = 48;   // padded for level 3
    localparam N_CA3     = 24;   // cA3 / cD3 length

    // Input sample buffer (188 entries: 187 + 1 pad)
    reg signed [7:0] sample_buf [0:N_PAD1-1];
    reg [7:0] sample_cnt;

    // Level 1 output: cA1[94], cD1[94] — 10-bit signed
    reg signed [9:0] cA1 [0:N_CA1-1];
    reg signed [9:0] cD1 [0:N_CA1-1];

    // Level 2 output: cA2[48], cD2[47] — 11-bit signed
    // (cA2 has 48 entries: 47 + 1 pad for level 3)
    reg signed [10:0] cA2 [0:N_PAD3-1];
    reg signed [10:0] cD2 [0:N_CA2-1];

    // Level 3 output: cA3[24], cD3[24] — 12-bit signed
    reg signed [11:0] cA3_buf [0:N_CA3-1];
    reg signed [11:0] cD3_buf [0:N_CA3-1];

    // Read port (directly from sub-band memories)
    assign o_cD1 = cD1[i_rd_addr];
    assign o_cD2 = cD2[i_rd_addr];
    assign o_cD3 = cD3_buf[i_rd_addr];
    assign o_cA3 = cA3_buf[i_rd_addr];

    // FSM
    localparam S_IDLE    = 2'd0;
    localparam S_INGEST  = 2'd1;
    localparam S_COMPUTE = 2'd2;
    localparam S_DONE    = 2'd3;

    reg [1:0] state;
    reg [1:0] compute_stage;  // 0=level1, 1=level2, 2=level3

    integer k;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            compute_stage <= 2'd0;
            sample_cnt    <= 8'd0;
            o_busy        <= 1'b0;
            o_done        <= 1'b0;
        end else begin
            o_done <= 1'b0;  // default: de-assert

            case (state)
                // ─── IDLE ───
                S_IDLE: begin
                    if (i_start) begin
                        state      <= S_INGEST;
                        o_busy     <= 1'b1;
                        // Capture first sample if valid arrives with start
                        if (i_sample_valid) begin
                            sample_buf[0] <= i_sample;
                            sample_cnt    <= 8'd1;
                        end else begin
                            sample_cnt    <= 8'd0;
                        end
                    end
                end

                // ─── INGEST: accept 187 samples ───
                S_INGEST: begin
                    if (i_sample_valid) begin
                        sample_buf[sample_cnt] <= i_sample;

                        if (sample_cnt == N_SAMPLES - 1) begin
                            // Pad sample 187 = copy of sample 186
                            sample_buf[N_SAMPLES] <= i_sample;
                            state         <= S_COMPUTE;
                            compute_stage <= 2'd0;
                        end

                        sample_cnt <= sample_cnt + 8'd1;
                    end
                end

                // ─── COMPUTE: 3 pipeline stages (1 cycle each) ───
                S_COMPUTE: begin
                    case (compute_stage)
                        2'd0: begin
                            // Level 1: 188 samples → 94 pairs
                            for (k = 0; k < N_CA1; k = k + 1) begin
                                cA1[k] <= $signed({sample_buf[2*k][7], sample_buf[2*k]})
                                        + $signed({sample_buf[2*k+1][7], sample_buf[2*k+1]});
                                cD1[k] <= $signed({sample_buf[2*k][7], sample_buf[2*k]})
                                        - $signed({sample_buf[2*k+1][7], sample_buf[2*k+1]});
                            end
                            compute_stage <= 2'd1;
                        end

                        2'd1: begin
                            // Level 2: 94 cA1 values → 47 pairs
                            for (k = 0; k < N_CA2; k = k + 1) begin
                                cA2[k] <= {cA1[2*k][9], cA1[2*k]} + {cA1[2*k+1][9], cA1[2*k+1]};
                                cD2[k] <= {cA1[2*k][9], cA1[2*k]} - {cA1[2*k+1][9], cA1[2*k+1]};
                            end
                            // Pad: cA2[47] = cA2[46] (done next cycle after cA2[46] is computed)
                            // We'll handle padding in stage 2 by using cA1 directly
                            // Actually compute cA2[47] = cA2[46] here:
                            // cA2[46] = cA1[92] + cA1[93], so cA2[47] = cA2[46]
                            // But cA2[46] isn't ready until next clock. Let's compute it directly:
                            cA2[47] <= {cA1[92][9], cA1[92]} + {cA1[93][9], cA1[93]};
                            compute_stage <= 2'd2;
                        end

                        2'd2: begin
                            // Level 3: 48 cA2 values → 24 pairs
                            for (k = 0; k < N_CA3; k = k + 1) begin
                                cA3_buf[k] <= {cA2[2*k][10], cA2[2*k]} + {cA2[2*k+1][10], cA2[2*k+1]};
                                cD3_buf[k] <= {cA2[2*k][10], cA2[2*k]} - {cA2[2*k+1][10], cA2[2*k+1]};
                            end
                            state <= S_DONE;
                        end

                        default: state <= S_DONE;
                    endcase
                end

                // ─── DONE ───
                S_DONE: begin
                    o_done <= 1'b1;
                    o_busy <= 1'b0;
                    state  <= S_IDLE;
                end
            endcase
        end
    end

endmodule
