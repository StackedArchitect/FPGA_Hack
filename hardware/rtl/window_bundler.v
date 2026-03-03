// ============================================================
// Window Bundler — Accumulates sample vectors via majority vote
// ============================================================
// For each bit position in the D-bit hypervector, maintains a counter
// counting how many sample vectors had a '1' at that position.
// After all samples in the window are processed, thresholds the
// counters to produce the final D-bit query vector.
//
// Processing:
//   Phase 1 (ACCUMULATE): For each sample, for each chunk:
//     - Read counter chunk from BRAM
//     - For each bit: counter += sample_bit
//     - Write updated counters back
//   Phase 2 (THRESHOLD): For each chunk:
//     - Read counter chunk
//     - Threshold: query_bit = (counter > WINDOW_SIZE/2) ? 1 : 0
//     - Store query chunk
//
// FPGA: Counter BRAM + adder array + comparator array
//       Counter memory: NUM_CHUNKS × CHUNK_W counters × COUNTER_W bits
//       Stored in BRAM: depth = NUM_CHUNKS, width = CHUNK_W * COUNTER_W
//       For CHUNK_W=32, COUNTER_W=8: 256 bits per word → 8 BRAM36
// ============================================================

module window_bundler #(
    parameter CHUNK_W      = 32,
    parameter NUM_CHUNKS   = 128,
    parameter CHUNK_ADDR_W = 7,
    parameter COUNTER_W    = 8,         // log2(WINDOW_SIZE) + 1
    parameter WINDOW_SIZE  = 128
)(
    input  wire                    clk,
    input  wire                    rst_n,

    // Control
    input  wire                    clear,          // Reset all counters (new window)
    input  wire                    threshold_start, // Begin thresholding

    // Input: encoded sample chunks (from sample_encoder)
    input  wire [CHUNK_W-1:0]     chunk_in,
    input  wire                    chunk_valid,
    input  wire [CHUNK_ADDR_W-1:0] chunk_idx_in,

    // Output: thresholded query vector chunks
    output reg  [CHUNK_W-1:0]     query_chunk,
    output reg                    query_valid,
    output reg  [CHUNK_ADDR_W-1:0] query_chunk_idx,
    output reg                    query_done       // All query chunks output
);

    // Counter storage — BRAM
    // Each address holds CHUNK_W counters of COUNTER_W bits each
    // Total width = CHUNK_W * COUNTER_W = 32 * 8 = 256 bits
    localparam COUNTER_MEM_W = CHUNK_W * COUNTER_W;

    (* ram_style = "block" *)
    reg [COUNTER_MEM_W-1:0] counter_mem [0:NUM_CHUNKS-1];

    // Threshold value
    localparam [COUNTER_W-1:0] THRESH = WINDOW_SIZE / 2;

    // FSM
    localparam S_IDLE      = 3'd0;
    localparam S_ACC_READ  = 3'd1;
    localparam S_ACC_WAIT  = 3'd2;
    localparam S_ACC_WRITE = 3'd3;
    localparam S_THR_READ  = 3'd4;
    localparam S_THR_WAIT  = 3'd5;
    localparam S_THR_OUT   = 3'd6;
    localparam S_CLEAR     = 3'd7;

    reg [2:0] state;
    reg [CHUNK_ADDR_W-1:0] addr_cnt;
    reg [CHUNK_W-1:0] latched_chunk;
    reg [CHUNK_ADDR_W-1:0] latched_idx;
    reg [COUNTER_MEM_W-1:0] read_data;

    integer i;

    // Clear counter memory
    integer ci;
    always @(posedge clk) begin
        if (!rst_n || state == S_CLEAR) begin
            // Sequential clear — takes NUM_CHUNKS cycles
            // For synthesis, initialize in reset or use a clear FSM
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            addr_cnt       <= 0;
            query_valid    <= 1'b0;
            query_done     <= 1'b0;
            query_chunk    <= 0;
            query_chunk_idx <= 0;
            latched_chunk  <= 0;
            latched_idx    <= 0;
            read_data      <= 0;
        end
        else begin
            query_valid <= 1'b0;
            query_done  <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (clear) begin
                        addr_cnt <= 0;
                        state <= S_CLEAR;
                    end
                    else if (chunk_valid) begin
                        latched_chunk <= chunk_in;
                        latched_idx   <= chunk_idx_in;
                        state <= S_ACC_READ;
                    end
                    else if (threshold_start) begin
                        addr_cnt <= 0;
                        state <= S_THR_READ;
                    end
                end

                // ---- Accumulate: read counter, add sample bit, write back ----
                S_ACC_READ: begin
                    // BRAM read address is latched_idx
                    read_data <= counter_mem[latched_idx];
                    state <= S_ACC_WAIT;
                end

                S_ACC_WAIT: begin
                    state <= S_ACC_WRITE;
                end

                S_ACC_WRITE: begin
                    // Add each sample bit to corresponding counter
                    begin : acc_block
                        reg [COUNTER_MEM_W-1:0] new_data;
                        new_data = read_data;
                        for (i = 0; i < CHUNK_W; i = i + 1) begin
                            if (latched_chunk[i]) begin
                                new_data[i*COUNTER_W +: COUNTER_W] =
                                    read_data[i*COUNTER_W +: COUNTER_W] + 1;
                            end
                        end
                        counter_mem[latched_idx] <= new_data;
                    end
                    state <= S_IDLE;
                end

                // ---- Threshold: read counters, compare, output query ----
                S_THR_READ: begin
                    read_data <= counter_mem[addr_cnt];
                    state <= S_THR_WAIT;
                end

                S_THR_WAIT: begin
                    state <= S_THR_OUT;
                end

                S_THR_OUT: begin
                    // Threshold each counter → query bit
                    begin : thr_block
                        reg [CHUNK_W-1:0] q_chunk;
                        for (i = 0; i < CHUNK_W; i = i + 1) begin
                            q_chunk[i] = (read_data[i*COUNTER_W +: COUNTER_W] > THRESH) ? 1'b1 : 1'b0;
                        end
                        query_chunk <= q_chunk;
                    end
                    query_valid     <= 1'b1;
                    query_chunk_idx <= addr_cnt;

                    if (addr_cnt == NUM_CHUNKS - 1) begin
                        query_done <= 1'b1;
                        state <= S_IDLE;
                    end
                    else begin
                        addr_cnt <= addr_cnt + 1;
                        state <= S_THR_READ;
                    end
                end

                // ---- Clear all counters ----
                S_CLEAR: begin
                    counter_mem[addr_cnt] <= {COUNTER_MEM_W{1'b0}};
                    if (addr_cnt == NUM_CHUNKS - 1) begin
                        state <= S_IDLE;
                    end
                    else begin
                        addr_cnt <= addr_cnt + 1;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
