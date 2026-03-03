// ============================================================
// Hamming Distance Calculator — XOR + Popcount for one chunk
// ============================================================
// Computes partial Hamming distance for a CHUNK_W-bit slice of
// the query vector vs a class prototype vector.
//
// Full Hamming distance = sum of partial distances across all chunks.
//
// FPGA: CHUNK_W XOR gates + popcount tree. Pure LUT logic.
// ============================================================

module hamming_distance #(
    parameter CHUNK_W = 32,
    parameter OUT_W   = $clog2(CHUNK_W) + 1  // Bits for partial distance
)(
    input  wire [CHUNK_W-1:0]  query_chunk,
    input  wire [CHUNK_W-1:0]  proto_chunk,
    output wire [OUT_W-1:0]    partial_dist     // Number of differing bits
);

    wire [CHUNK_W-1:0] diff;

    // XOR: find differing bit positions
    assign diff = query_chunk ^ proto_chunk;

    // Popcount: count number of 1s (differing bits)
    popcount #(.WIDTH(CHUNK_W)) pc (
        .data_in  (diff),
        .count_out(partial_dist)
    );

endmodule
