// WaveBNN-ECG: Inference Core
// Wavelet -> 4x BNN branches -> FC1 -> FC2 -> argmax

module wavebnn_core (
    input  wire        clk,
    input  wire        rst_n,

    // ── Input: stream 187 ECG samples ──
    input  wire        i_start,          // pulse: begin inference
    input  wire signed [7:0] i_sample,   // 8-bit signed ECG sample
    input  wire        i_sample_valid,   // sample strobe

    // ── Output ──
    output wire        o_busy,           // high during inference
    output reg         o_done,           // 1-cycle pulse when classification ready
    output reg  [2:0]  o_class           // predicted AAMI class (0-4)
);

    // Sub-band parameters
    localparam LEN_CA3 = 24;
    localparam LEN_CD3 = 24;
    localparam LEN_CD2 = 47;
    localparam LEN_CD1 = 94;
    localparam MAX_LEN = 94;   // max sub-band length

    // Branch output sizes
    localparam FLAT_CA3 = 320;   // 10 * 32
    localparam FLAT_CD3 = 320;   // 10 * 32
    localparam FLAT_CD2 = 672;   // 21 * 32
    localparam FLAT_CD1 = 736;   // 46 * 16
    localparam CONCAT   = 2048;  // total

    // FSM states
    localparam S_IDLE      = 3'd0;
    localparam S_WAVELET   = 3'd1;
    localparam S_FEED      = 3'd2;
    localparam S_WAIT_BR   = 3'd3;
    localparam S_FC1       = 3'd4;
    localparam S_FC2       = 3'd5;
    localparam S_DONE      = 3'd6;

    reg [2:0]  state;
    reg [6:0]  feed_addr;  // 0..93

    assign o_busy = (state != S_IDLE);

    // Haar Wavelet Module
    // wav_start is combinational so wavelet sees i_start on the same cycle
    wire        wav_start = (state == S_IDLE) && i_start;
    wire        wav_busy, wav_done;
    reg  [6:0]  wav_rd_addr;
    wire signed [9:0]  wav_cD1;
    wire signed [10:0] wav_cD2;
    wire signed [11:0] wav_cD3;
    wire signed [11:0] wav_cA3;

    haar_wavelet_3lvl u_wavelet (
        .clk            (clk),
        .rst_n          (rst_n),
        .i_start        (wav_start),
        .i_sample       (i_sample),
        .i_sample_valid (i_sample_valid),
        .o_busy         (wav_busy),
        .o_done         (wav_done),
        .i_rd_addr      (wav_rd_addr),
        .o_cD1          (wav_cD1),
        .o_cD2          (wav_cD2),
        .o_cD3          (wav_cD3),
        .o_cA3          (wav_cA3)
    );

    // Branch start / valid signals
    // wav_rd_addr is registered (1-cycle lag behind feed_addr).
    // Pipeline-delay the valid signals to match, so data and valid arrive together.
    reg br_start;
    reg        feed_active;  // delayed (state == S_FEED)
    reg [6:0]  feed_addr_d;  // delayed feed_addr

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            feed_active <= 1'b0;
            feed_addr_d <= 7'd0;
        end else begin
            feed_active <= (state == S_FEED);
            feed_addr_d <= feed_addr;
        end
    end

    wire br_cA3_valid = feed_active && (feed_addr_d < LEN_CA3);
    wire br_cD3_valid = feed_active && (feed_addr_d < LEN_CD3);
    wire br_cD2_valid = feed_active && (feed_addr_d < LEN_CD2);
    wire br_cD1_valid = feed_active && (feed_addr_d < LEN_CD1);

    // BNN Branch: cA3 (24 × 12-bit → 320 flat bits)
    wire        br_cA3_done;
    wire [FLAT_CA3-1:0] br_cA3_feat;

    bnn_branch #(
        .IN_LEN     (LEN_CA3),
        .IN_WIDTH   (12),
        .OUT_CH     (32),
        .KERNEL     (5),
        .POOL       (2),
        .WEIGHT_FILE("cA3_conv_weights.mem"),
        .THRESH_FILE("cA3_bn_threshold.mem")
    ) u_br_cA3 (
        .clk        (clk),
        .rst_n      (rst_n),
        .i_start    (br_start),
        .i_data     (wav_cA3),
        .i_valid    (br_cA3_valid),
        .o_done     (br_cA3_done),
        .o_features (br_cA3_feat)
    );

    // BNN Branch: cD3 (24 × 12-bit → 320 flat bits)
    wire        br_cD3_done;
    wire [FLAT_CD3-1:0] br_cD3_feat;

    bnn_branch #(
        .IN_LEN     (LEN_CD3),
        .IN_WIDTH   (12),
        .OUT_CH     (32),
        .KERNEL     (5),
        .POOL       (2),
        .WEIGHT_FILE("cD3_conv_weights.mem"),
        .THRESH_FILE("cD3_bn_threshold.mem")
    ) u_br_cD3 (
        .clk        (clk),
        .rst_n      (rst_n),
        .i_start    (br_start),
        .i_data     (wav_cD3),
        .i_valid    (br_cD3_valid),
        .o_done     (br_cD3_done),
        .o_features (br_cD3_feat)
    );

    // BNN Branch: cD2 (47 × 11-bit → 672 flat bits)
    wire        br_cD2_done;
    wire [FLAT_CD2-1:0] br_cD2_feat;

    bnn_branch #(
        .IN_LEN     (LEN_CD2),
        .IN_WIDTH   (11),
        .OUT_CH     (32),
        .KERNEL     (5),
        .POOL       (2),
        .WEIGHT_FILE("cD2_conv_weights.mem"),
        .THRESH_FILE("cD2_bn_threshold.mem")
    ) u_br_cD2 (
        .clk        (clk),
        .rst_n      (rst_n),
        .i_start    (br_start),
        .i_data     (wav_cD2),
        .i_valid    (br_cD2_valid),
        .o_done     (br_cD2_done),
        .o_features (br_cD2_feat)
    );

    // BNN Branch: cD1 (94 × 10-bit → 736 flat bits)
    wire        br_cD1_done;
    wire [FLAT_CD1-1:0] br_cD1_feat;

    bnn_branch #(
        .IN_LEN     (LEN_CD1),
        .IN_WIDTH   (10),
        .OUT_CH     (16),
        .KERNEL     (3),
        .POOL       (2),
        .WEIGHT_FILE("cD1_conv_weights.mem"),
        .THRESH_FILE("cD1_bn_threshold.mem")
    ) u_br_cD1 (
        .clk        (clk),
        .rst_n      (rst_n),
        .i_start    (br_start),
        .i_data     (wav_cD1),
        .i_valid    (br_cD1_valid),
        .o_done     (br_cD1_done),
        .o_features (br_cD1_feat)
    );

    // ─── All branches done ───
    wire all_br_done = br_cA3_done & br_cD3_done & br_cD2_done & br_cD1_done;

    // Track branches that have finished (they pulse o_done at different times)
    reg br_cA3_finished, br_cD3_finished, br_cD2_finished, br_cD1_finished;
    wire all_br_finished = (br_cA3_finished | br_cA3_done) &
                           (br_cD3_finished | br_cD3_done) &
                           (br_cD2_finished | br_cD2_done) &
                           (br_cD1_finished | br_cD1_done);

    // ─── Concatenate branch outputs (cA3 at LSB, matching PyTorch cat order) ───
    wire [CONCAT-1:0] concat_features = {br_cD1_feat, br_cD2_feat, br_cD3_feat, br_cA3_feat};

    // Binary FC1 (2048 → 128)
    reg         fc1_start;
    wire        fc1_done;
    wire [127:0] fc1_out;

    bin_fc1 #(
        .IN_BITS     (CONCAT),
        .OUT_BITS    (128),
        .WEIGHT_FILE ("fc1_weights.mem"),
        .THRESH_FILE ("fc1_bn_threshold.mem")
    ) u_fc1 (
        .clk    (clk),
        .rst_n  (rst_n),
        .i_start(fc1_start),
        .i_data (concat_features),
        .o_done (fc1_done),
        .o_data (fc1_out)
    );

    // FC2 Output (128 → 5) + Argmax
    reg         fc2_start;
    wire        fc2_done;
    wire [2:0]  fc2_class;

    fc_output #(
        .IN_BITS     (128),
        .NUM_CLASSES (5),
        .WEIGHT_FILE ("fc2_weights.mem"),
        .BIAS_FILE   ("fc2_bias.mem")
    ) u_fc2 (
        .clk     (clk),
        .rst_n   (rst_n),
        .i_start (fc2_start),
        .i_data  (fc1_out),
        .o_done  (fc2_done),
        .o_class (fc2_class),
        .o_score ()            // unused debug port
    );

    // Main FSM
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            feed_addr <= 7'd0;
            br_start  <= 1'b0;
            fc1_start <= 1'b0;
            fc2_start <= 1'b0;
            o_done    <= 1'b0;
            o_class   <= 3'd0;
            br_cA3_finished <= 1'b0;
            br_cD3_finished <= 1'b0;
            br_cD2_finished <= 1'b0;
            br_cD1_finished <= 1'b0;
        end else begin
            // Default: deassert pulses
            br_start  <= 1'b0;
            fc1_start <= 1'b0;
            fc2_start <= 1'b0;
            o_done    <= 1'b0;

            // Track branch completions
            if (br_cA3_done) br_cA3_finished <= 1'b1;
            if (br_cD3_done) br_cD3_finished <= 1'b1;
            if (br_cD2_done) br_cD2_finished <= 1'b1;
            if (br_cD1_done) br_cD1_finished <= 1'b1;

            case (state)
                // ─── IDLE: wait for start ───
                S_IDLE: begin
                    if (i_start) begin
                        state     <= S_WAVELET;
                    end
                end

                // ─── WAVELET: wait for DWT completion ───
                S_WAVELET: begin
                    if (wav_done) begin
                        state     <= S_FEED;
                        feed_addr <= 7'd0;
                        br_start  <= 1'b1;  // start all branches
                        wav_rd_addr <= 7'd0;
                        br_cA3_finished <= 1'b0;
                        br_cD3_finished <= 1'b0;
                        br_cD2_finished <= 1'b0;
                        br_cD1_finished <= 1'b0;
                    end
                end

                // ─── FEED: read sub-bands and stream to branches ───
                S_FEED: begin
                    wav_rd_addr <= feed_addr;

                    if (feed_addr == MAX_LEN - 1) begin
                        state <= S_WAIT_BR;
                    end
                    feed_addr <= feed_addr + 7'd1;
                end

                // ─── WAIT_BR: wait for all branches to complete ───
                S_WAIT_BR: begin
                    if (all_br_finished) begin
                        fc1_start <= 1'b1;
                        state     <= S_FC1;
                    end
                end

                // ─── FC1: binary fully-connected ───
                S_FC1: begin
                    if (fc1_done) begin
                        fc2_start <= 1'b1;
                        state     <= S_FC2;
                    end
                end

                // ─── FC2: output + argmax ───
                S_FC2: begin
                    if (fc2_done) begin
                        o_class <= fc2_class;
                        state   <= S_DONE;
                    end
                end

                // ─── DONE: signal completion ───
                S_DONE: begin
                    o_done <= 1'b1;
                    state  <= S_IDLE;
                end
            endcase
        end
    end

endmodule
