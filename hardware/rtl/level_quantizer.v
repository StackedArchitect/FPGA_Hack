// ============================================================
// Level Quantizer — Maps 8-bit unsigned input to Q-level index
// ============================================================
// Input:  8-bit unsigned value [0..255]
// Output: Q_BITS-bit level index [0..Q-1]
//
// Linear quantization: level = input >> (INPUT_W - Q_BITS)
// For INPUT_W=8, Q=16, Q_BITS=4: level = input[7:4]
//
// FPGA: Zero logic! Just wire selection (bit slice).
// ============================================================

module level_quantizer #(
    parameter INPUT_W = 8,          // Input bit width
    parameter Q_BITS  = 4           // Output level index bit width (Q = 2^Q_BITS)
)(
    input  wire [INPUT_W-1:0]  data_in,
    output wire [Q_BITS-1:0]   level_out
);

    // Simple bit truncation — take the MSBs
    // This maps [0, 255] → [0, 15] for Q=16
    // Equivalent to floor(data_in / (2^(INPUT_W - Q_BITS)))
    assign level_out = data_in[INPUT_W-1 -: Q_BITS];

endmodule
