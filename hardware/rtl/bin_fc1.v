// WaveBNN-ECG: Binary FC (2048 -> 128), 4-stage pipeline
// XNOR -> popcount -> partial sums -> threshold

module bin_fc1 #(
    parameter IN_BITS   = 2048,
    parameter OUT_BITS  = 128,
    parameter WEIGHT_FILE = "fc1_weights.mem",
    parameter THRESH_FILE = "fc1_bn_threshold.mem"
)(
    input  wire               clk,
    input  wire               rst_n,

    input  wire               i_start,
    input  wire [IN_BITS-1:0] i_data,

    output reg                o_done,
    output reg  [OUT_BITS-1:0] o_data
);

    // ─── Weight memory: 128 rows × 2048 bits ───
    (* ram_style = "block" *)
    reg [IN_BITS-1:0] weights [0:OUT_BITS-1];
    initial $readmemb(WEIGHT_FILE, weights);

    // ─── Threshold memory: 128 × 16-bit ───
    (* rom_style = "distributed" *)
    reg [15:0] thresholds [0:OUT_BITS-1];
    initial $readmemh(THRESH_FILE, thresholds);

    // ─── Constants ───
    localparam CHUNK_W     = 128;
    localparam N_CHUNKS    = IN_BITS / CHUNK_W;  // 16
    localparam HALF_CHUNKS = N_CHUNKS / 2;       // 8
    localparam CHUNK_CNT_W = $clog2(CHUNK_W) + 1;   // 8 bits (0..128)
    localparam HALF_SUM_W  = $clog2(CHUNK_W * HALF_CHUNKS) + 1; // 11 bits (0..1024)
    localparam TOTAL_SUM_W = $clog2(IN_BITS) + 1;   // 12 bits (0..2048)
    localparam IDX_W       = $clog2(OUT_BITS);       // 7 bits

    // ─── Latched input ───
    reg [IN_BITS-1:0] input_reg;

    // ─── Neuron counter + pipeline delays ───
    reg [IDX_W:0] neuron_idx;     // Stage 1 feeder
    reg [IDX_W:0] nid_s1;         // After Stage 1 (XNOR)
    reg [IDX_W:0] nid_s2;         // After Stage 2 (POP)
    reg [IDX_W:0] nid_s3;         // After Stage 3 (SUM)

    // ─── Stage 1 output: registered XNOR ───
    reg [IN_BITS-1:0] xnor_reg;

    // ─── Stage 2: popcount (combinational from xnor_reg) ───
    wire [CHUNK_CNT_W-1:0] chunk_cnt_comb [0:N_CHUNKS-1];
    genvar c;
    generate
        for (c = 0; c < N_CHUNKS; c = c + 1) begin : gen_pop
            popcount #(.IN_WIDTH(CHUNK_W)) u_pop (
                .i_data (xnor_reg[c*CHUNK_W +: CHUNK_W]),
                .o_count(chunk_cnt_comb[c])
            );
        end
    endgenerate

    // ─── Stage 2 output: registered chunk counts ───
    reg [CHUNK_CNT_W-1:0] chunk_cnt_reg [0:N_CHUNKS-1];

    // ─── Stage 3: partial sums (combinational from chunk_cnt_reg) ───
    // Sum chunks 0..7 and chunks 8..15 separately
    reg [HALF_SUM_W-1:0] psum_lo_comb, psum_hi_comb;
    integer j;
    always @(*) begin
        psum_lo_comb = 0;
        psum_hi_comb = 0;
        for (j = 0; j < HALF_CHUNKS; j = j + 1) begin
            psum_lo_comb = psum_lo_comb + {{(HALF_SUM_W-CHUNK_CNT_W){1'b0}}, chunk_cnt_reg[j]};
            psum_hi_comb = psum_hi_comb + {{(HALF_SUM_W-CHUNK_CNT_W){1'b0}}, chunk_cnt_reg[j + HALF_CHUNKS]};
        end
    end

    // ─── Stage 3 output: registered partial sums ───
    reg [HALF_SUM_W-1:0] psum_lo_reg, psum_hi_reg;

    // ─── Stage 4: final sum + threshold (combinational from psum regs) ───
    wire [TOTAL_SUM_W-1:0] total_pop = {{(TOTAL_SUM_W-HALF_SUM_W){1'b0}}, psum_lo_reg}
                                      + {{(TOTAL_SUM_W-HALF_SUM_W){1'b0}}, psum_hi_reg};

    // ─── Pipeline valid flags ───
    reg s2_valid, s3_valid, s4_valid;

    // ─── FSM ───
    localparam S_IDLE = 3'd0;
    localparam S_RUN  = 3'd1;  // Steady-state pipeline
    localparam S_DONE = 3'd2;

    reg [2:0] state;
    reg [2:0] fill_cnt;  // Pipeline fill counter (0..3)
    reg       draining;  // No more neurons to feed

    integer k;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            neuron_idx <= 0;
            nid_s1     <= 0;
            nid_s2     <= 0;
            nid_s3     <= 0;
            s2_valid   <= 1'b0;
            s3_valid   <= 1'b0;
            s4_valid   <= 1'b0;
            draining   <= 1'b0;
            fill_cnt   <= 0;
            o_done     <= 1'b0;
            o_data     <= {OUT_BITS{1'b0}};
            input_reg  <= {IN_BITS{1'b0}};
            xnor_reg   <= {IN_BITS{1'b0}};
            psum_lo_reg <= 0;
            psum_hi_reg <= 0;
            for (k = 0; k < N_CHUNKS; k = k + 1)
                chunk_cnt_reg[k] <= 0;
        end else begin
            o_done <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        input_reg  <= i_data;
                        neuron_idx <= 0;
                        s2_valid   <= 1'b0;
                        s3_valid   <= 1'b0;
                        s4_valid   <= 1'b0;
                        draining   <= 1'b0;
                        fill_cnt   <= 0;
                        state      <= S_RUN;
                    end
                end

                S_RUN: begin
                    // === Stage 1: XNOR (feeds into xnor_reg) ===
                    if (!draining) begin
                        xnor_reg <= ~(weights[neuron_idx[IDX_W-1:0]] ^ input_reg);
                        nid_s1   <= neuron_idx;
                        if (neuron_idx == OUT_BITS - 1)
                            draining <= 1'b1;
                        else
                            neuron_idx <= neuron_idx + 1;
                    end

                    // === Stage 2: Popcount (reads xnor_reg, writes chunk_cnt_reg) ===
                    if (fill_cnt >= 1 || s2_valid) begin
                        for (k = 0; k < N_CHUNKS; k = k + 1)
                            chunk_cnt_reg[k] <= chunk_cnt_comb[k];
                        nid_s2   <= nid_s1;
                        s2_valid <= 1'b1;
                    end

                    // === Stage 3: Partial sums (reads chunk_cnt_reg, writes psum regs) ===
                    if (fill_cnt >= 2 || s3_valid) begin
                        psum_lo_reg <= psum_lo_comb;
                        psum_hi_reg <= psum_hi_comb;
                        nid_s3      <= nid_s2;
                        s3_valid    <= 1'b1;
                    end

                    // === Stage 4: Threshold compare (reads psum regs, writes o_data) ===
                    if (fill_cnt >= 3 || s4_valid) begin
                        o_data[nid_s3[IDX_W-1:0]] <=
                            (total_pop > {1'b0, thresholds[nid_s3[IDX_W-1:0]]}) ? 1'b1 : 1'b0;
                        s4_valid <= 1'b1;

                        // Check if this was the last neuron out of Stage 4
                        if (nid_s3[IDX_W-1:0] == OUT_BITS - 1) begin
                            state <= S_DONE;
                        end
                    end

                    // Pipeline fill counter
                    if (fill_cnt < 4)
                        fill_cnt <= fill_cnt + 1;
                end

                S_DONE: begin
                    o_done   <= 1'b1;
                    s2_valid <= 1'b0;
                    s3_valid <= 1'b0;
                    s4_valid <= 1'b0;
                    state    <= S_IDLE;
                end
            endcase
        end
    end

endmodule
