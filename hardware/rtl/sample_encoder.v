// ============================================================
// Sample Encoder — Encodes one I/Q sample into a CHUNK_W-bit vector chunk
// ============================================================
// For each time sample:
//   1. Quantize amplitude and phase-diff to levels
//   2. Look up codebook_A[level_A] and codebook_B[level_B] (BRAM read)
//   3. XOR the two vectors → sample_vector chunk
//
// This module processes CHUNK_W bits per clock cycle.
// A full D-bit sample vector is built over NUM_CHUNKS cycles.
//
// FPGA: 2 BRAM reads + CHUNK_W XOR gates per cycle
// ============================================================

module sample_encoder #(
    parameter INPUT_W     = 8,
    parameter Q_BITS      = 4,
    parameter CHUNK_W     = 32,
    parameter NUM_CHUNKS  = 128,        // D / CHUNK_W
    parameter CHUNK_ADDR_W = 7,         // log2(NUM_CHUNKS)
    parameter CB_DEPTH    = 2048,       // Q * NUM_CHUNKS
    parameter CB_ADDR_W   = 11          // ceil(log2(CB_DEPTH))
)(
    input  wire                    clk,
    input  wire                    rst_n,

    // Input sample (valid for 1 cycle when start asserted)
    input  wire                    start,      // Begin encoding a new sample
    input  wire [INPUT_W-1:0]     amp_in,      // Amplitude (unsigned 8-bit)
    input  wire [INPUT_W-1:0]     pdiff_in,    // Phase difference (unsigned 8-bit)

    // Codebook ROM interfaces
    output wire [CB_ADDR_W-1:0]   cb_a_addr,   // Codebook A (amplitude) address
    input  wire [CHUNK_W-1:0]     cb_a_data,   // Codebook A read data
    output wire [CB_ADDR_W-1:0]   cb_b_addr,   // Codebook B (phase-diff) address
    input  wire [CHUNK_W-1:0]     cb_b_data,   // Codebook B read data

    // Output: one chunk of the encoded sample vector per cycle
    output reg  [CHUNK_W-1:0]     chunk_out,   // XOR result chunk
    output reg                    chunk_valid,  // Chunk output is valid
    output reg  [CHUNK_ADDR_W-1:0] chunk_idx,  // Which chunk [0..NUM_CHUNKS-1]
    output reg                    sample_done   // All chunks for this sample output
);

    // Level quantization (pure combinational — just bit selection)
    wire [Q_BITS-1:0] level_a, level_b;
    level_quantizer #(.INPUT_W(INPUT_W), .Q_BITS(Q_BITS)) quant_a (
        .data_in(amp_in), .level_out(level_a)
    );
    level_quantizer #(.INPUT_W(INPUT_W), .Q_BITS(Q_BITS)) quant_b (
        .data_in(pdiff_in), .level_out(level_b)
    );

    // FSM states
    localparam IDLE   = 3'd0;
    localparam READ   = 3'd1;  // Issue BRAM read address
    localparam WAIT   = 3'd2;  // Wait for BRAM data (1 cycle latency)
    localparam OUTPUT = 3'd3;  // Output XOR result
    localparam PAUSE  = 3'd4;  // Extra cycle — lets bundler return to IDLE

    reg [2:0] state;
    reg [CHUNK_ADDR_W-1:0] chunk_cnt;
    reg [Q_BITS-1:0] latched_level_a, latched_level_b;

    // Codebook addresses: level * NUM_CHUNKS + chunk_index
    assign cb_a_addr = latched_level_a * NUM_CHUNKS + chunk_cnt;
    assign cb_b_addr = latched_level_b * NUM_CHUNKS + chunk_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= IDLE;
            chunk_cnt   <= 0;
            chunk_out   <= 0;
            chunk_valid <= 1'b0;
            chunk_idx   <= 0;
            sample_done <= 1'b0;
            latched_level_a <= 0;
            latched_level_b <= 0;
        end
        else begin
            chunk_valid <= 1'b0;
            sample_done <= 1'b0;

            case (state)
                IDLE: begin
                    if (start) begin
                        latched_level_a <= level_a;
                        latched_level_b <= level_b;
                        chunk_cnt <= 0;
                        state <= READ;
                    end
                end

                READ: begin
                    // Address is already set via continuous assign
                    // Wait one cycle for BRAM read latency
                    state <= WAIT;
                end

                WAIT: begin
                    state <= OUTPUT;
                end

                OUTPUT: begin
                    // XOR the two codebook chunks → encoded sample chunk
                    chunk_out   <= cb_a_data ^ cb_b_data;
                    chunk_valid <= 1'b1;
                    chunk_idx   <= chunk_cnt;

                    if (chunk_cnt == NUM_CHUNKS - 1) begin
                        sample_done <= 1'b1;
                        state <= IDLE;
                    end
                    else begin
                        chunk_cnt <= chunk_cnt + 1;
                        state <= PAUSE;
                    end
                end

                PAUSE: begin
                    // Extra wait — bundler needs 3 cycles per chunk
                    // (ACC_READ → ACC_WAIT → ACC_WRITE → IDLE)
                    // Without this pause, chunk_valid fires while bundler
                    // is still in ACC_WRITE, causing a dropped chunk.
                    state <= READ;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
