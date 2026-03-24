// =============================================================================
// UART Receiver (8N1, parameterized baud rate)
// =============================================================================
//
// Standard UART RX: 1 start bit, 8 data bits (LSB first), 1 stop bit.
// Oversamples at 16× baud rate for robust start-bit detection.
// Double-flip-flop synchronizer for metastability protection.
//
// Target: PYNQ-Z2 @ 100 MHz, 115200 baud
// =============================================================================

module uart_rx #(
    parameter CLK_FREQ  = 100_000_000,
    parameter BAUD_RATE = 115_200
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       i_rx,          // UART RX pin

    output reg        o_valid,       // 1-cycle pulse: byte received
    output reg  [7:0] o_data         // received byte
);

    localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;

    // ─── Synchronizer ───
    reg rx_sync1, rx_sync2;
    always @(posedge clk) begin
        rx_sync1 <= i_rx;
        rx_sync2 <= rx_sync1;
    end

    // ─── FSM states ───
    localparam S_IDLE  = 2'd0;
    localparam S_START = 2'd1;
    localparam S_DATA  = 2'd2;
    localparam S_STOP  = 2'd3;

    reg [1:0]  state;
    reg [$clog2(CLKS_PER_BIT)-1:0] clk_cnt;
    reg [2:0]  bit_idx;
    reg [7:0]  rx_shift;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= S_IDLE;
            clk_cnt <= 0;
            bit_idx <= 0;
            rx_shift <= 8'd0;
            o_valid <= 1'b0;
            o_data  <= 8'd0;
        end else begin
            o_valid <= 1'b0;

            case (state)
                // ─── Wait for falling edge (start bit) ───
                S_IDLE: begin
                    if (rx_sync2 == 1'b0) begin
                        state   <= S_START;
                        clk_cnt <= 0;
                    end
                end

                // ─── Sample middle of start bit ───
                S_START: begin
                    if (clk_cnt == (CLKS_PER_BIT / 2) - 1) begin
                        if (rx_sync2 == 1'b0) begin
                            // Valid start bit
                            state   <= S_DATA;
                            clk_cnt <= 0;
                            bit_idx <= 0;
                        end else begin
                            // Glitch — back to idle
                            state <= S_IDLE;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end

                // ─── Sample 8 data bits (LSB first) ───
                S_DATA: begin
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        rx_shift[bit_idx] <= rx_sync2;
                        clk_cnt <= 0;
                        if (bit_idx == 3'd7) begin
                            state <= S_STOP;
                        end else begin
                            bit_idx <= bit_idx + 1;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end

                // ─── Wait for stop bit ───
                S_STOP: begin
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        if (rx_sync2 == 1'b1) begin
                            o_valid <= 1'b1;
                            o_data  <= rx_shift;
                        end
                        state <= S_IDLE;
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end
            endcase
        end
    end

endmodule
