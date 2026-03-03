// ============================================================
// Popcount Module — Counts number of 1-bits in input vector
// ============================================================
// Uses a hierarchical adder tree for efficient LUT mapping.
// This is the fundamental building block for Hamming distance.
//
// FPGA: Pure combinational logic in LUTs. No DSP, no BRAM.
// Latency: 0 cycles (combinational) or 1 cycle (registered output)
//
// For CHUNK_W=32: popcount of 32 bits → 6-bit result [0..32]
// LUT usage: ~32 LUTs (Artix-7 uses 6-input LUTs)
// ============================================================

module popcount #(
    parameter WIDTH  = 32,                      // Input width
    parameter OUT_W  = $clog2(WIDTH) + 1        // Output width (enough bits for max count)
)(
    input  wire [WIDTH-1:0]  data_in,
    output wire [OUT_W-1:0]  count_out
);

    // Hierarchical popcount using generate blocks
    // Recursively split input in half, count each half, add

    generate
        if (WIDTH == 1) begin : base_case
            assign count_out = {{(OUT_W-1){1'b0}}, data_in[0]};
        end
        else if (WIDTH == 2) begin : base_2
            assign count_out = {{(OUT_W-2){1'b0}}, data_in[0], 1'b0}
                             + {{(OUT_W-1){1'b0}}, data_in[1]};
        end
        else begin : recursive
            localparam HALF  = WIDTH / 2;
            localparam REST  = WIDTH - HALF;
            localparam SUB_W = $clog2(HALF > REST ? HALF : REST) + 1;

            wire [SUB_W-1:0] count_lo, count_hi;

            popcount #(.WIDTH(HALF))  pc_lo (
                .data_in  (data_in[HALF-1:0]),
                .count_out(count_lo)
            );

            popcount #(.WIDTH(REST))  pc_hi (
                .data_in  (data_in[WIDTH-1:HALF]),
                .count_out(count_hi)
            );

            // Zero-extend and add
            assign count_out = {{(OUT_W-SUB_W){1'b0}}, count_lo}
                             + {{(OUT_W-SUB_W){1'b0}}, count_hi};
        end
    endgenerate

endmodule
