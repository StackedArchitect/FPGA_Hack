// Parameterized popcount (adder tree)

module popcount #(
    parameter IN_WIDTH  = 64,
    parameter OUT_WIDTH = $clog2(IN_WIDTH) + 1   // ceil(log2(N)) + 1
)(
    input  wire [IN_WIDTH-1:0]  i_data,
    output wire [OUT_WIDTH-1:0] o_count
);

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
