// ============================================================
// Classifier — Computes Hamming distances to all class prototypes
// ============================================================
// Takes the thresholded D-bit query vector (chunk by chunk) and
// computes Hamming distance to each of NUM_CLASSES prototype vectors.
//
// Algorithm:
//   For each class c (0..NUM_CLASSES-1):
//     For each chunk k (0..NUM_CHUNKS-1):
//       partial_dist = popcount(query_chunk[k] XOR prototype[c][k])
//       total_dist[c] += partial_dist
//   result = argmin(total_dist)
//
// Architecture: Sequential over classes (saves area vs parallel).
// Processes one class at a time, one chunk per 3 cycles (BRAM read + XOR + acc).
//
// Latency: NUM_CLASSES × (NUM_CHUNKS × 3 + overhead) cycles
// For 11 classes × 128 chunks: ~4224 cycles = 42.2 µs @ 100MHz
//
// FPGA: 1 prototype BRAM + 1 Hamming unit + 1 accumulator + argmin logic
// ============================================================

module classifier #(
    parameter CHUNK_W      = 32,
    parameter NUM_CHUNKS   = 128,
    parameter CHUNK_ADDR_W = 7,
    parameter NUM_CLASSES  = 11,
    parameter CLASS_W      = 4,
    parameter DIST_W       = 13,             // log2(D) + 1
    parameter PROTO_DEPTH  = 1408,           // NUM_CLASSES * NUM_CHUNKS
    parameter PROTO_ADDR_W = 11,
    parameter PROTO_HEX    = "prototypes.hex"
)(
    input  wire                    clk,
    input  wire                    rst_n,

    // Control
    input  wire                    start,        // Begin classification

    // Query vector input (stored internally from bundler output)
    input  wire [CHUNK_W-1:0]     query_chunk,
    input  wire                    query_valid,
    input  wire [CHUNK_ADDR_W-1:0] query_chunk_idx,
    input  wire                    query_load_done,  // All query chunks loaded

    // Classification result
    output reg  [CLASS_W-1:0]     result_class,
    output reg  [DIST_W-1:0]      result_dist,
    output reg                    result_valid,
    output reg                    busy
);

    // ---- Query vector storage (BRAM) ----
    (* ram_style = "block" *)
    reg [CHUNK_W-1:0] query_mem [0:NUM_CHUNKS-1];

    // Store incoming query chunks
    always @(posedge clk) begin
        if (query_valid)
            query_mem[query_chunk_idx] <= query_chunk;
    end

    // ---- Prototype ROM ----
    (* ram_style = "block" *)
    reg [CHUNK_W-1:0] proto_mem [0:PROTO_DEPTH-1];

    initial begin
        $readmemh(PROTO_HEX, proto_mem);
    end

    // Synchronous reads
    reg [PROTO_ADDR_W-1:0] proto_addr;
    reg [CHUNK_W-1:0] proto_data;
    reg [CHUNK_ADDR_W-1:0] query_addr;
    reg [CHUNK_W-1:0] query_data;

    always @(posedge clk) begin
        proto_data <= proto_mem[proto_addr];
        query_data <= query_mem[query_addr];
    end

    // ---- Hamming distance (combinational) ----
    localparam PARTIAL_W = $clog2(CHUNK_W) + 1;
    wire [PARTIAL_W-1:0] partial_dist;

    hamming_distance #(.CHUNK_W(CHUNK_W)) hd_unit (
        .query_chunk(query_data),
        .proto_chunk(proto_data),
        .partial_dist(partial_dist)
    );

    // ---- FSM ----
    localparam S_IDLE  = 3'd0;
    localparam S_LOAD  = 3'd1;    // Wait for query loading
    localparam S_READ  = 3'd2;    // Issue BRAM read
    localparam S_WAIT1 = 3'd3;    // Wait for data
    localparam S_ACC   = 3'd4;    // Accumulate partial distance
    localparam S_NEXT  = 3'd5;    // Move to next class or finish

    reg [2:0] state;
    reg [CLASS_W-1:0]      class_cnt;
    reg [CHUNK_ADDR_W-1:0] chunk_cnt;
    reg [DIST_W-1:0]       acc_dist;       // Running distance for current class
    reg [DIST_W-1:0]       best_dist;      // Best (minimum) distance so far
    reg [CLASS_W-1:0]      best_class;     // Class with minimum distance
    reg                    query_loaded;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            class_cnt    <= 0;
            chunk_cnt    <= 0;
            acc_dist     <= 0;
            best_dist    <= {DIST_W{1'b1}};  // Maximum possible
            best_class   <= 0;
            result_valid <= 1'b0;
            result_class <= 0;
            result_dist  <= 0;
            busy         <= 1'b0;
            query_loaded <= 1'b0;
            proto_addr   <= 0;
            query_addr   <= 0;
        end
        else begin
            result_valid <= 1'b0;

            // Track query loading
            if (query_load_done)
                query_loaded <= 1'b1;

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start && query_loaded) begin
                        class_cnt  <= 0;
                        chunk_cnt  <= 0;
                        acc_dist   <= 0;
                        best_dist  <= {DIST_W{1'b1}};
                        best_class <= 0;
                        busy       <= 1'b1;
                        state      <= S_READ;
                    end
                end

                S_READ: begin
                    // Set BRAM addresses
                    proto_addr <= class_cnt * NUM_CHUNKS + chunk_cnt;
                    query_addr <= chunk_cnt;
                    state <= S_WAIT1;
                end

                S_WAIT1: begin
                    // Wait for BRAM read (1 cycle latency)
                    state <= S_ACC;
                end

                S_ACC: begin
                    // Accumulate partial Hamming distance
                    acc_dist <= acc_dist + {{(DIST_W-PARTIAL_W){1'b0}}, partial_dist};

                    if (chunk_cnt == NUM_CHUNKS - 1) begin
                        // Done with this class
                        state <= S_NEXT;
                    end
                    else begin
                        chunk_cnt <= chunk_cnt + 1;
                        state <= S_READ;
                    end
                end

                S_NEXT: begin
                    // Check if this class has a smaller distance
                    // Need to use acc_dist + last partial (since ACC happens same cycle)
                    if (acc_dist < best_dist) begin
                        best_dist  <= acc_dist;
                        best_class <= class_cnt;
                    end

                    if (class_cnt == NUM_CLASSES - 1) begin
                        // Done — output result
                        result_class <= (acc_dist < best_dist) ? class_cnt : best_class;
                        result_dist  <= (acc_dist < best_dist) ? acc_dist : best_dist;
                        result_valid <= 1'b1;
                        query_loaded <= 1'b0;
                        state <= S_IDLE;
                    end
                    else begin
                        class_cnt <= class_cnt + 1;
                        chunk_cnt <= 0;
                        acc_dist  <= 0;
                        state <= S_READ;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
