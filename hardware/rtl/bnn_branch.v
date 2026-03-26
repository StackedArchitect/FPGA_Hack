// WaveBNN-ECG: BNN Branch — BinConv1d + BN threshold + MaxPool(2)
// 3-stage pipeline: PREFETCH -> COMPUTE -> THRESHOLD

module bnn_branch #(
    parameter IN_LEN    = 24,
    parameter IN_WIDTH  = 12,
    parameter OUT_CH    = 32,
    parameter KERNEL    = 5,
    parameter POOL      = 2,
    parameter CONV_OUT  = IN_LEN - KERNEL + 1,
    parameter POOL_OUT  = CONV_OUT / POOL,
    parameter FLAT_BITS = POOL_OUT * OUT_CH,

    parameter WEIGHT_FILE = "conv_weights.mem",
    parameter THRESH_FILE = "bn_threshold.mem"
)(
    input  wire                   clk,
    input  wire                   rst_n,

    input  wire                   i_start,
    input  wire signed [IN_WIDTH-1:0] i_data,
    input  wire                   i_valid,

    output reg                    o_done,
    output reg  [FLAT_BITS-1:0]   o_features
);

    localparam ACC_WIDTH = IN_WIDTH + $clog2(KERNEL) + 1;

    // ─── Weight & threshold storage ───
    (* rom_style = "distributed" *)
    reg [KERNEL-1:0] weights [0:OUT_CH-1];
    initial $readmemb(WEIGHT_FILE, weights);

    (* rom_style = "distributed" *)
    reg [15:0] thresholds [0:OUT_CH-1];
    initial $readmemh(THRESH_FILE, thresholds);

    // ─── Input buffer ───
    (* max_fanout = 32 *)
    reg signed [IN_WIDTH-1:0] input_buf [0:IN_LEN-1];
    reg [$clog2(IN_LEN):0]   load_cnt;

    // ─── Conv output bits ───
    reg [OUT_CH-1:0] conv_bits [0:CONV_OUT-1];

    // ─── FSM ───
    localparam S_IDLE     = 3'd0;
    localparam S_LOAD     = 3'd1;
    localparam S_PREFETCH = 3'd2;  // Prime: prefetch window for pos=0
    localparam S_COMPUTE  = 3'd3;  // Steady: prefetch(N+1) + compute(N) + threshold(N-1)
    localparam S_DRAIN1   = 3'd4;  // Drain: compute last + threshold(N-1)
    localparam S_DRAIN2   = 3'd5;  // Drain: threshold for last position
    localparam S_POOL     = 3'd6;

    reg [2:0] state;

    (* max_fanout = 16 *)
    reg [$clog2(CONV_OUT):0] pos;         // Next position to prefetch
    reg [$clog2(CONV_OUT):0] compute_pos; // Position whose window is in window[]
    reg [$clog2(CONV_OUT):0] thresh_pos;  // Position whose acc is in acc_reg[]
    reg                      acc_valid;   // acc_reg has valid data

    // ─── Kernel window pre-fetch registers ───
    reg signed [IN_WIDTH-1:0] window [0:KERNEL-1];

    // ─── Registered accumulators ───
    reg signed [ACC_WIDTH-1:0] acc_reg [0:OUT_CH-1];

    // ─── Combinational: accumulate from window registers ───
    wire signed [ACC_WIDTH-1:0] acc_comb [0:OUT_CH-1];
    genvar ch;
    generate
        for (ch = 0; ch < OUT_CH; ch = ch + 1) begin : gen_ch
            reg signed [ACC_WIDTH-1:0] acc;
            integer k;
            always @(*) begin
                acc = 0;
                for (k = 0; k < KERNEL; k = k + 1) begin
                    if (weights[ch][k])
                        acc = acc + window[k];
                    else
                        acc = acc - window[k];
                end
            end
            assign acc_comb[ch] = acc;
        end
    endgenerate

    // ─── Threshold comparison (from registered accumulators) ───
    wire [OUT_CH-1:0] thresh_result;
    generate
        for (ch = 0; ch < OUT_CH; ch = ch + 1) begin : gen_thresh
            assign thresh_result[ch] = ($signed(acc_reg[ch]) > $signed(thresholds[ch])) ? 1'b1 : 1'b0;
        end
    endgenerate

    // ─── FSM logic ───
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            load_cnt    <= 0;
            pos         <= 0;
            compute_pos <= 0;
            thresh_pos  <= 0;
            acc_valid   <= 1'b0;
            o_done      <= 1'b0;
            o_features  <= {FLAT_BITS{1'b0}};
        end else begin
            o_done <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        state    <= S_LOAD;
                        load_cnt <= 0;
                    end
                end

                S_LOAD: begin
                    if (i_valid) begin
                        input_buf[load_cnt] <= i_data;
                        if (load_cnt == IN_LEN - 1) begin
                            state     <= S_PREFETCH;
                            pos       <= 0;
                            acc_valid <= 1'b0;
                        end
                        load_cnt <= load_cnt + 1;
                    end
                end

                // ── Prime pipeline: prefetch window for position 0 ──
                S_PREFETCH: begin
                    for (i = 0; i < KERNEL; i = i + 1)
                        window[i] <= input_buf[pos + i];
                    compute_pos <= pos;
                    pos         <= pos + 1;
                    state       <= S_COMPUTE;
                end

                // ── Steady state: 3 overlapped stages ──
                S_COMPUTE: begin
                    // --- Stage 1: Register accumulators from window ---
                    for (i = 0; i < OUT_CH; i = i + 1)
                        acc_reg[i] <= acc_comb[i];
                    thresh_pos <= compute_pos;  // This acc_reg will hold compute_pos's result

                    // --- Stage 2: Threshold from acc_reg (previous position) ---
                    if (acc_valid)
                        conv_bits[thresh_pos] <= thresh_result;

                    acc_valid <= 1'b1;

                    // --- Stage 0: Prefetch next window ---
                    if (pos <= CONV_OUT - 1) begin
                        for (i = 0; i < KERNEL; i = i + 1)
                            window[i] <= input_buf[pos + i];
                        compute_pos <= pos;
                        pos         <= pos + 1;
                    end else begin
                        state <= S_DRAIN1;
                    end
                end

                // ── Drain 1: compute last window + threshold for prev ──
                S_DRAIN1: begin
                    // Compute last window's accumulators
                    for (i = 0; i < OUT_CH; i = i + 1)
                        acc_reg[i] <= acc_comb[i];
                    thresh_pos <= compute_pos;

                    // Threshold for previous position
                    conv_bits[thresh_pos] <= thresh_result;

                    state <= S_DRAIN2;
                end

                // ── Drain 2: threshold for final position ──
                S_DRAIN2: begin
                    conv_bits[thresh_pos] <= thresh_result;
                    state <= S_POOL;
                end

                S_POOL: begin
                    // MaxPool(2): OR consecutive pairs
                    for (i = 0; i < POOL_OUT; i = i + 1) begin : pool_pos
                        integer ch_idx;
                        for (ch_idx = 0; ch_idx < OUT_CH; ch_idx = ch_idx + 1) begin
                            o_features[ch_idx * POOL_OUT + i] <=
                                conv_bits[2*i][ch_idx] | conv_bits[2*i + 1][ch_idx];
                        end
                    end
                    o_done    <= 1'b1;
                    acc_valid <= 1'b0;
                    state     <= S_IDLE;
                end
            endcase
        end
    end

endmodule
