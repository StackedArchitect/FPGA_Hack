// =============================================================================
// UART Transmitter (8N1, parameterized baud rate)
// =============================================================================
//
// Standard UART TX: 1 start bit, 8 data bits (LSB first), 1 stop bit.
//
// Target: PYNQ-Z2 @ 100 MHz, 115200 baud
// =============================================================================

module uart_tx #(
    parameter CLK_FREQ  = 100_000_000,
    parameter BAUD_RATE = 115_200
)(
    input  wire       clk,
    input  wire       rst_n,

    input  wire       i_valid,       // pulse: send byte
    input  wire [7:0] i_data,        // byte to send

    output reg        o_tx,          // UART TX pin
    output wire       o_busy         // high while transmitting
);

    localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;

    // ─── FSM states ───
    localparam S_IDLE  = 2'd0;
    localparam S_START = 2'd1;
    localparam S_DATA  = 2'd2;
    localparam S_STOP  = 2'd3;

    reg [1:0]  state;
    reg [$clog2(CLKS_PER_BIT)-1:0] clk_cnt;
    reg [2:0]  bit_idx;
    reg [7:0]  tx_shift;

    assign o_busy = (state != S_IDLE);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            clk_cnt  <= 0;
            bit_idx  <= 0;
            tx_shift <= 8'd0;
            o_tx     <= 1'b1;  // idle high
        end else begin
            case (state)
                // ─── Idle: wait for send request ───
                S_IDLE: begin
                    o_tx <= 1'b1;
                    if (i_valid) begin
                        tx_shift <= i_data;
                        state    <= S_START;
                        clk_cnt  <= 0;
                    end
                end

                // ─── Start bit (low) ───
                S_START: begin
                    o_tx <= 1'b0;
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt <= 0;
                        bit_idx <= 0;
                        state   <= S_DATA;
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end

                // ─── Data bits (LSB first) ───
                S_DATA: begin
                    o_tx <= tx_shift[bit_idx];
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
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

                // ─── Stop bit (high) ───
                S_STOP: begin
                    o_tx <= 1'b1;
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        state <= S_IDLE;
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end
            endcase
        end
    end

endmodule
