// ============================================================
// UART Transmitter — 8N1, configurable baud rate
// ============================================================
// Standard UART transmitter: 1 start bit, 8 data bits, 1 stop bit.
// Sends classification results back to host PC.
//
// FPGA: ~30 LUTs + 15 FFs on Artix-7
// ============================================================

module uart_tx #(
    parameter CLK_FREQ  = 100_000_000,
    parameter BAUD_RATE = 115200
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire [7:0] data_in,      // Byte to transmit
    input  wire       data_valid,   // Assert to start transmission
    output reg        tx,           // UART TX pin
    output reg        busy          // High while transmitting
);

    localparam TICK_COUNT = CLK_FREQ / BAUD_RATE;

    localparam S_IDLE  = 2'd0;
    localparam S_START = 2'd1;
    localparam S_DATA  = 2'd2;
    localparam S_STOP  = 2'd3;

    reg [1:0]  state;
    reg [15:0] tick_cnt;
    reg [2:0]  bit_cnt;
    reg [7:0]  shift_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            tx        <= 1'b1;   // Idle high
            busy      <= 1'b0;
            tick_cnt  <= 0;
            bit_cnt   <= 0;
            shift_reg <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    tx   <= 1'b1;
                    busy <= 1'b0;
                    if (data_valid) begin
                        shift_reg <= data_in;
                        busy      <= 1'b1;
                        tick_cnt  <= 0;
                        state     <= S_START;
                    end
                end

                S_START: begin
                    tx <= 1'b0;  // Start bit
                    if (tick_cnt == TICK_COUNT - 1) begin
                        tick_cnt <= 0;
                        bit_cnt  <= 0;
                        state    <= S_DATA;
                    end else begin
                        tick_cnt <= tick_cnt + 1;
                    end
                end

                S_DATA: begin
                    tx <= shift_reg[0];  // LSB first
                    if (tick_cnt == TICK_COUNT - 1) begin
                        tick_cnt  <= 0;
                        shift_reg <= {1'b0, shift_reg[7:1]};
                        if (bit_cnt == 7) begin
                            state <= S_STOP;
                        end else begin
                            bit_cnt <= bit_cnt + 1;
                        end
                    end else begin
                        tick_cnt <= tick_cnt + 1;
                    end
                end

                S_STOP: begin
                    tx <= 1'b1;  // Stop bit
                    if (tick_cnt == TICK_COUNT - 1) begin
                        tick_cnt <= 0;
                        state    <= S_IDLE;
                    end else begin
                        tick_cnt <= tick_cnt + 1;
                    end
                end
            endcase
        end
    end

endmodule
