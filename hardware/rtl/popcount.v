// =============================================================================
// Parameterized Popcount (Count 1s in an N-bit vector)
// =============================================================================
// Uses adder tree — synthesizes efficiently into LUT fabric.
// Vivado 2024.2 compatible, fully synthesizable.
//
// Target: PYNQ-Z2 (xc7z020clg484-1) @ 100 MHz
// =============================================================================

module popcount #(
    parameter IN_WIDTH  = 64,
    parameter OUT_WIDTH = $clog2(IN_WIDTH) + 1   // ceil(log2(N)) + 1
)(
    input  wire [IN_WIDTH-1:0]  i_data,
    output wire [OUT_WIDTH-1:0] o_count
);

    // ─── Pure Verilog-2001 Combinational Loop ───
    // Note: Vivado's synthesis engine natively unrolls this into an optimal 
    // parallel balanced adder tree using LUTs. It does NOT synthesize sequentially
    // into logic over time, so cycle latency and functionality remain identical.
    reg [OUT_WIDTH-1:0] count_val;
    integer i;

    always @* begin
        count_val = 0;
        for (i = 0; i < IN_WIDTH; i = i + 1) begin
            count_val = count_val + i_data[i];
        end
    end

    assign o_count = count_val;

endmodule
