// WaveBNN-ECG: FC Output (128 -> 5) + 2-stage pipelined argmax

module fc_output #(
    parameter IN_BITS     = 128,
    parameter NUM_CLASSES = 5,
    parameter W_BITS      = 16,
    parameter ACC_BITS    = 24,
    parameter WEIGHT_FILE = "fc2_weights.mem",
    parameter BIAS_FILE   = "fc2_bias.mem"
)(
    input  wire                clk,
    input  wire                rst_n,

    input  wire                i_start,
    input  wire [IN_BITS-1:0]  i_data,

    output reg                 o_done,
    output reg  [2:0]          o_class,
    output reg  signed [ACC_BITS-1:0] o_score
);

    // ─── Weight memory ───
    (* rom_style = "distributed" *)
    reg [W_BITS-1:0] weights [0:NUM_CLASSES*IN_BITS-1];
    initial $readmemh(WEIGHT_FILE, weights);

    // ─── Bias ───
    (* rom_style = "distributed" *)
    reg [W_BITS-1:0] bias [0:NUM_CLASSES-1];
    initial $readmemh(BIAS_FILE, bias);

    // ─── Input register ───
    reg [IN_BITS-1:0] input_reg;

    // ─── Accumulators ───
    reg signed [ACC_BITS-1:0] acc [0:NUM_CLASSES-1];

    // ─── Bit counter ───
    reg [$clog2(IN_BITS):0] bit_idx;

    // ─── FSM ───
    localparam S_IDLE    = 3'd0;
    localparam S_INIT    = 3'd1;
    localparam S_COMPUTE = 3'd2;
    localparam S_ARG_P1  = 3'd3;  // Argmax stage 1: pairwise compare
    localparam S_ARG_P2  = 3'd4;  // Argmax stage 2: final compare
    localparam S_DONE    = 3'd5;

    reg [2:0] state;

    // ─── Weight reads ───
    wire signed [W_BITS-1:0] cur_weight [0:NUM_CLASSES-1];
    genvar g;
    generate
        for (g = 0; g < NUM_CLASSES; g = g + 1) begin : gen_w
            assign cur_weight[g] = $signed(weights[g * IN_BITS + bit_idx[$clog2(IN_BITS)-1:0]]);
        end
    endgenerate

    // ─── Argmax pipeline registers ───
    // Stage 1 outputs: winners of pair comparisons
    reg signed [ACC_BITS-1:0] cand_01_score;    // winner of acc[0] vs acc[1]
    reg [2:0]                 cand_01_class;
    reg signed [ACC_BITS-1:0] cand_23_score;    // winner of acc[2] vs acc[3]
    reg [2:0]                 cand_23_class;
    reg signed [ACC_BITS-1:0] cand_4_score;     // acc[4] passes through
    reg [2:0]                 cand_4_class;

    // ─── FSM logic ───
    integer ci;
    reg signed [ACC_BITS-1:0] cand_0123_score;
    reg [2:0]                 cand_0123_class;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            bit_idx   <= 0;
            o_done    <= 1'b0;
            o_class   <= 3'd0;
            o_score   <= 0;
            for (ci = 0; ci < NUM_CLASSES; ci = ci + 1)
                acc[ci] <= 0;
        end else begin
            o_done <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        input_reg <= i_data;
                        state     <= S_INIT;
                    end
                end

                S_INIT: begin
                    for (ci = 0; ci < NUM_CLASSES; ci = ci + 1)
                        acc[ci] <= {{(ACC_BITS-W_BITS){bias[ci][W_BITS-1]}}, $signed(bias[ci])};
                    bit_idx <= 0;
                    state   <= S_COMPUTE;
                end

                S_COMPUTE: begin
                    for (ci = 0; ci < NUM_CLASSES; ci = ci + 1) begin
                        if (input_reg[bit_idx[$clog2(IN_BITS)-1:0]])
                            acc[ci] <= acc[ci] + {{(ACC_BITS-W_BITS){cur_weight[ci][W_BITS-1]}}, cur_weight[ci]};
                        else
                            acc[ci] <= acc[ci] - {{(ACC_BITS-W_BITS){cur_weight[ci][W_BITS-1]}}, cur_weight[ci]};
                    end

                    if (bit_idx == IN_BITS - 1)
                        state <= S_ARG_P1;
                    bit_idx <= bit_idx + 1;
                end

                // ── Argmax Stage 1: pairwise comparison (registered) ──
                // Compare: acc[0] vs acc[1], acc[2] vs acc[3], pass acc[4]
                S_ARG_P1: begin
                    // Pair 0 vs 1
                    if (acc[0] >= acc[1]) begin
                        cand_01_score <= acc[0];
                        cand_01_class <= 3'd0;
                    end else begin
                        cand_01_score <= acc[1];
                        cand_01_class <= 3'd1;
                    end

                    // Pair 2 vs 3
                    if (acc[2] >= acc[3]) begin
                        cand_23_score <= acc[2];
                        cand_23_class <= 3'd2;
                    end else begin
                        cand_23_score <= acc[3];
                        cand_23_class <= 3'd3;
                    end

                    // Pass acc[4] through
                    cand_4_score <= acc[4];
                    cand_4_class <= 3'd4;

                    state <= S_ARG_P2;
                end

                // ── Argmax Stage 2: final 3-way comparison ──
                S_ARG_P2: begin
                    // Compare winner_01 vs winner_23
                    if (cand_01_score >= cand_23_score) begin
                        cand_0123_score = cand_01_score;
                        cand_0123_class = cand_01_class;
                    end else begin
                        cand_0123_score = cand_23_score;
                        cand_0123_class = cand_23_class;
                    end

                    // Compare with acc[4]
                    if (cand_0123_score >= cand_4_score) begin
                        o_class <= cand_0123_class;
                        o_score <= cand_0123_score;
                    end else begin
                        o_class <= cand_4_class;
                        o_score <= cand_4_score;
                    end

                    state <= S_DONE;
                end

                S_DONE: begin
                    o_done <= 1'b1;
                    state  <= S_IDLE;
                end
            endcase
        end
    end

endmodule
