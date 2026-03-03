// ============================================================
// UART Receiver — 8N1, configurable baud rate
// ============================================================
// Standard UART receiver with oversampling (16x baud rate).
// Receives 8 data bits, no parity, 1 stop bit.
//
// For demo: receives I/Q data from host PC for real-time classification.
//
// FPGA: ~50 LUTs + 20 FFs on Artix-7
// ============================================================

module uart_rx #(
    parameter CLK_FREQ  = 100_000_000,  // System clock frequency (Hz)
    parameter BAUD_RATE = 115200         // UART baud rate
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       rx,           // UART RX pin (active high, idle = 1)
    output reg  [7:0] data_out,     // Received byte
    output reg        data_valid    // Pulse high for 1 cycle when byte received
);

    // Oversampling rate
    localparam OVERSAMPLE = 16;
    localparam TICK_COUNT = CLK_FREQ / (BAUD_RATE * OVERSAMPLE);

    // State machine
    localparam S_IDLE  = 2'd0;
    localparam S_START = 2'd1;
    localparam S_DATA  = 2'd2;
    localparam S_STOP  = 2'd3;

    reg [1:0]  state;
    reg [15:0] tick_cnt;        // Baud rate counter
    reg [3:0]  os_cnt;          // Oversample counter (0..15)
    reg [2:0]  bit_cnt;         // Data bit counter (0..7)
    reg [7:0]  shift_reg;       // Receive shift register
    reg        rx_sync1, rx_sync2;  // Input synchronizer

    // Double-flop synchronizer for metastability
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_sync1 <= 1'b1;
            rx_sync2 <= 1'b1;
        end else begin
            rx_sync1 <= rx;
            rx_sync2 <= rx_sync1;
        end
    end

    wire rx_in = rx_sync2;

    // Baud tick generation
    reg tick;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tick_cnt <= 0;
            tick <= 1'b0;
        end else begin
            if (tick_cnt == TICK_COUNT - 1) begin
                tick_cnt <= 0;
                tick <= 1'b1;
            end else begin
                tick_cnt <= tick_cnt + 1;
                tick <= 1'b0;
            end
        end
    end

    // UART RX state machine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            os_cnt     <= 0;
            bit_cnt    <= 0;
            shift_reg  <= 0;
            data_out   <= 0;
            data_valid <= 1'b0;
        end else begin
            data_valid <= 1'b0;

            if (tick) begin
                case (state)
                    S_IDLE: begin
                        os_cnt <= 0;
                        if (rx_in == 1'b0) begin
                            // Possible start bit
                            state <= S_START;
                            os_cnt <= 0;
                        end
                    end

                    S_START: begin
                        os_cnt <= os_cnt + 1;
                        if (os_cnt == OVERSAMPLE/2 - 1) begin
                            // Sample at middle of start bit
                            if (rx_in == 1'b0) begin
                                // Valid start bit
                                os_cnt <= 0;
                                bit_cnt <= 0;
                                state <= S_DATA;
                            end else begin
                                // False start
                                state <= S_IDLE;
                            end
                        end
                    end

                    S_DATA: begin
                        os_cnt <= os_cnt + 1;
                        if (os_cnt == OVERSAMPLE - 1) begin
                            os_cnt <= 0;
                            // Sample data bit at center
                            shift_reg <= {rx_in, shift_reg[7:1]};
                            if (bit_cnt == 7) begin
                                state <= S_STOP;
                            end else begin
                                bit_cnt <= bit_cnt + 1;
                            end
                        end
                    end

                    S_STOP: begin
                        os_cnt <= os_cnt + 1;
                        if (os_cnt == OVERSAMPLE - 1) begin
                            // Stop bit (should be high)
                            if (rx_in == 1'b1) begin
                                data_out <= shift_reg;
                                data_valid <= 1'b1;
                            end
                            state <= S_IDLE;
                        end
                    end
                endcase
            end
        end
    end

endmodule
