// ============================================================
// HDC Core — Top-level Hyperdimensional Computing Engine
// ============================================================
// Full pipeline: Sample Encoding → Bundling → Classification
//
// Interface:
//   - Feed I/Q sample pairs one at a time via amp_in/pdiff_in
//   - After WINDOW_SIZE samples, result_valid goes high with class ID
//   - Assert new_window before feeding the next window
//
// Timing (at 100 MHz, D=4096, CHUNK_W=32, NUM_CHUNKS=128):
//   Encoding: WINDOW_SIZE × NUM_CHUNKS × 4 cycles = 128 × 128 × 4 = 65536 cycles = 655 µs
//   Thresholding: NUM_CHUNKS × 3 = 384 cycles = 3.8 µs
//   Classification: NUM_CLASSES × NUM_CHUNKS × 3 = 11 × 128 × 3 = 4224 cycles = 42 µs
//   Total: ~70000 cycles = ~700 µs per window
//
// For higher throughput: increase CHUNK_W or pipeline the classification
//
// Target: Nexys A7-100T (XC7A100TCSG324-1)
// ============================================================

module hdc_core #(
    parameter INPUT_W      = 8,
    parameter Q_BITS       = 4,
    parameter CHUNK_W      = 32,
    parameter NUM_CHUNKS   = 128,      // D / CHUNK_W
    parameter CHUNK_ADDR_W = 7,
    parameter CB_DEPTH     = 2048,     // Q * NUM_CHUNKS
    parameter CB_ADDR_W    = 11,
    parameter COUNTER_W    = 8,
    parameter WINDOW_SIZE  = 128,
    parameter NUM_CLASSES  = 11,
    parameter CLASS_W      = 4,
    parameter DIST_W       = 13,
    parameter PROTO_DEPTH  = 1408,
    parameter PROTO_ADDR_W = 11,
    parameter CB_A_HEX     = "codebook_i.hex",
    parameter CB_B_HEX     = "codebook_q.hex",
    parameter PROTO_HEX    = "prototypes.hex"
)(
    input  wire                clk,
    input  wire                rst_n,

    // Input interface
    input  wire                new_window,      // Assert to start a new classification window
    input  wire                sample_valid,     // Pulse high when amp_in/pdiff_in valid
    input  wire [INPUT_W-1:0]  amp_in,          // Amplitude feature (unsigned 8-bit)
    input  wire [INPUT_W-1:0]  pdiff_in,        // Phase-diff feature (unsigned 8-bit)

    // Output interface
    output reg  [CLASS_W-1:0]  result_class,    // Predicted modulation class (latched)
    output reg  [DIST_W-1:0]   result_dist,     // Hamming distance of best match (latched)
    output reg                 result_valid,     // Classification result is valid (stays high until new_window)

    // Status
    output wire                busy,            // Core is processing
    output reg  [7:0]          sample_count     // Number of samples received in current window
);

    // ================================================================
    // Codebook ROMs (amplitude and phase-difference channels)
    // ================================================================
    wire [CB_ADDR_W-1:0] cb_a_addr, cb_b_addr;
    wire [CHUNK_W-1:0]   cb_a_data, cb_b_data;

    codebook_rom #(
        .DEPTH   (CB_DEPTH),
        .ADDR_W  (CB_ADDR_W),
        .DATA_W  (CHUNK_W),
        .HEX_FILE(CB_A_HEX)
    ) codebook_a (
        .clk     (clk),
        .addr    (cb_a_addr),
        .data_out(cb_a_data)
    );

    codebook_rom #(
        .DEPTH   (CB_DEPTH),
        .ADDR_W  (CB_ADDR_W),
        .DATA_W  (CHUNK_W),
        .HEX_FILE(CB_B_HEX)
    ) codebook_b (
        .clk     (clk),
        .addr    (cb_b_addr),
        .data_out(cb_b_data)
    );

    // ================================================================
    // Sample Encoder
    // ================================================================
    wire [CHUNK_W-1:0]      enc_chunk;
    wire                    enc_chunk_valid;
    wire [CHUNK_ADDR_W-1:0] enc_chunk_idx;
    wire                    enc_sample_done;

    sample_encoder #(
        .INPUT_W    (INPUT_W),
        .Q_BITS     (Q_BITS),
        .CHUNK_W    (CHUNK_W),
        .NUM_CHUNKS (NUM_CHUNKS),
        .CHUNK_ADDR_W(CHUNK_ADDR_W),
        .CB_DEPTH   (CB_DEPTH),
        .CB_ADDR_W  (CB_ADDR_W)
    ) encoder (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (sample_valid),
        .amp_in     (amp_in),
        .pdiff_in   (pdiff_in),
        .cb_a_addr  (cb_a_addr),
        .cb_a_data  (cb_a_data),
        .cb_b_addr  (cb_b_addr),
        .cb_b_data  (cb_b_data),
        .chunk_out  (enc_chunk),
        .chunk_valid(enc_chunk_valid),
        .chunk_idx  (enc_chunk_idx),
        .sample_done(enc_sample_done)
    );

    // ================================================================
    // Window Bundler
    // ================================================================
    wire [CHUNK_W-1:0]      query_chunk;
    wire                    query_valid;
    wire [CHUNK_ADDR_W-1:0] query_chunk_idx;
    wire                    query_done;

    window_bundler #(
        .CHUNK_W     (CHUNK_W),
        .NUM_CHUNKS  (NUM_CHUNKS),
        .CHUNK_ADDR_W(CHUNK_ADDR_W),
        .COUNTER_W   (COUNTER_W),
        .WINDOW_SIZE (WINDOW_SIZE)
    ) bundler (
        .clk            (clk),
        .rst_n          (rst_n),
        .clear          (new_window),
        .threshold_start(threshold_go),
        .chunk_in       (enc_chunk),
        .chunk_valid    (enc_chunk_valid),
        .chunk_idx_in   (enc_chunk_idx),
        .query_chunk    (query_chunk),
        .query_valid    (query_valid),
        .query_chunk_idx(query_chunk_idx),
        .query_done     (query_done)
    );

    // ================================================================
    // Classifier
    // ================================================================
    wire [CLASS_W-1:0] cls_result_class;
    wire [DIST_W-1:0]  cls_result_dist;
    wire               cls_result_valid;
    wire               cls_busy;

    classifier #(
        .CHUNK_W     (CHUNK_W),
        .NUM_CHUNKS  (NUM_CHUNKS),
        .CHUNK_ADDR_W(CHUNK_ADDR_W),
        .NUM_CLASSES (NUM_CLASSES),
        .CLASS_W     (CLASS_W),
        .DIST_W      (DIST_W),
        .PROTO_DEPTH (PROTO_DEPTH),
        .PROTO_ADDR_W(PROTO_ADDR_W),
        .PROTO_HEX   (PROTO_HEX)
    ) cls (
        .clk            (clk),
        .rst_n          (rst_n),
        .start          (classify_go),
        .query_chunk    (query_chunk),
        .query_valid    (query_valid),
        .query_chunk_idx(query_chunk_idx),
        .query_load_done(query_done),
        .result_class   (cls_result_class),
        .result_dist    (cls_result_dist),
        .result_valid   (cls_result_valid),
        .busy           (cls_busy)
    );

    // ================================================================
    // Top-Level Control FSM
    // ================================================================
    localparam FSM_IDLE       = 3'd0;
    localparam FSM_CLEARING   = 3'd1;
    localparam FSM_ENCODING   = 3'd2;
    localparam FSM_THRESHOLD  = 3'd3;
    localparam FSM_CLASSIFY   = 3'd4;
    localparam FSM_DONE       = 3'd5;

    reg [2:0] fsm_state;
    reg threshold_go;
    reg classify_go;
    reg all_samples_done;
    reg [CHUNK_ADDR_W:0] clear_cnt;   // Counter for bundler clear wait

    assign busy = (fsm_state != FSM_IDLE) && (fsm_state != FSM_DONE);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fsm_state        <= FSM_IDLE;
            sample_count     <= 0;
            threshold_go     <= 1'b0;
            classify_go      <= 1'b0;
            all_samples_done <= 1'b0;
            clear_cnt        <= 0;
            result_valid     <= 1'b0;
            result_class     <= 0;
            result_dist      <= 0;
        end
        else begin
            threshold_go <= 1'b0;
            classify_go  <= 1'b0;

            case (fsm_state)
                FSM_IDLE: begin
                    if (new_window) begin
                        sample_count     <= 0;
                        all_samples_done <= 1'b0;
                        result_valid     <= 1'b0;
                        clear_cnt        <= 0;
                        fsm_state        <= FSM_CLEARING;
                    end
                end

                FSM_CLEARING: begin
                    // Wait for bundler to finish clearing (NUM_CHUNKS + 2 cycles)
                    clear_cnt <= clear_cnt + 1;
                    if (clear_cnt >= NUM_CHUNKS + 2) begin
                        fsm_state <= FSM_ENCODING;
                    end
                end

                FSM_ENCODING: begin
                    // Count completed samples
                    if (enc_sample_done) begin
                        sample_count <= sample_count + 1;
                        if (sample_count == WINDOW_SIZE - 1) begin
                            all_samples_done <= 1'b1;
                        end
                    end

                    // Transition AFTER all_samples_done is registered (next cycle)
                    if (all_samples_done) begin
                        fsm_state <= FSM_THRESHOLD;
                    end
                end

                FSM_THRESHOLD: begin
                    threshold_go <= 1'b1;
                    fsm_state <= FSM_CLASSIFY;
                end

                FSM_CLASSIFY: begin
                    // Wait for query to be fully loaded, then start classification
                    if (query_done && !cls_busy) begin
                        classify_go <= 1'b1;
                    end
                    // Latch classification result
                    if (cls_result_valid) begin
                        result_class <= cls_result_class;
                        result_dist  <= cls_result_dist;
                        result_valid <= 1'b1;
                        fsm_state    <= FSM_DONE;
                    end
                end

                FSM_DONE: begin
                    // Result stays valid until next new_window
                    if (new_window) begin
                        sample_count     <= 0;
                        all_samples_done <= 1'b0;
                        result_valid     <= 1'b0;
                        clear_cnt        <= 0;
                        fsm_state        <= FSM_CLEARING;
                    end
                end

                default: fsm_state <= FSM_IDLE;
            endcase
        end
    end

endmodule
