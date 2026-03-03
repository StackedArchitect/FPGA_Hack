// ============================================================
// System Top — Board-level integration for Nexys A7-100T
// ============================================================
// Wires UART interface to HDC core for live demo.
//
// Protocol (host → FPGA via UART):
//   Byte 0: Command
//     0x01 = New window (reset and start)
//     0x02 = Load sample (followed by 2 bytes: amplitude, phase-diff)
//     0x03 = Classify (trigger classification after all samples loaded)
//   After command 0x02: next 2 bytes are amplitude[7:0], phase_diff[7:0]
//   After command 0x03: FPGA sends back result byte (class_id)
//
// LED[3:0] = result class (binary)
// LED[15:8] = sample count
// LED[7:4] = state indicator
// 7-segment display: shows class number (optional)
//
// Target: Nexys A7-100T (XC7A100TCSG324-1)
// CLK: 100 MHz from onboard oscillator
// Reset: BTNC (active high → active low internally)
// UART: USB-UART bridge (RX/TX on JA PMOD or built-in USB port)
// ============================================================

module system_top #(
    // HDC Parameters (match Python-exported values)
    parameter INPUT_W      = 8,
    parameter Q_BITS       = 4,
    parameter CHUNK_W      = 32,
    parameter NUM_CHUNKS   = 128,
    parameter CHUNK_ADDR_W = 7,
    parameter CB_DEPTH     = 2048,
    parameter CB_ADDR_W    = 11,
    parameter COUNTER_W    = 8,
    parameter WINDOW_SIZE  = 128,
    parameter NUM_CLASSES  = 11,
    parameter CLASS_W      = 4,
    parameter DIST_W       = 13,
    parameter PROTO_DEPTH  = 1408,
    parameter PROTO_ADDR_W = 11,
    // UART
    parameter CLK_FREQ     = 100_000_000,
    parameter BAUD_RATE    = 115200,
    // Memory init files
    parameter CB_A_HEX     = "codebook_i.hex",
    parameter CB_B_HEX     = "codebook_q.hex",
    parameter PROTO_HEX    = "prototypes.hex"
)(
    input  wire        CLK100MHZ,       // 100 MHz system clock
    input  wire        BTNC,            // Center button (reset, active high)

    // UART (directly from USB-UART bridge on Nexys A7)
    input  wire        UART_TXD_IN,     // UART RX (from PC → FPGA)
    output wire        UART_RXD_OUT,    // UART TX (from FPGA → PC)

    // LEDs
    output reg  [15:0] LED,

    // 7-segment (accent display, active low)
    output reg  [6:0]  SEG,            // Segments a-g
    output reg  [7:0]  AN              // Anode enables (active low)
);

    wire clk = CLK100MHZ;
    wire rst_n = ~BTNC;  // Active-low reset

    // ================================================================
    // UART RX
    // ================================================================
    wire [7:0] rx_data;
    wire       rx_valid;

    uart_rx #(
        .CLK_FREQ (CLK_FREQ),
        .BAUD_RATE(BAUD_RATE)
    ) u_uart_rx (
        .clk      (clk),
        .rst_n    (rst_n),
        .rx       (UART_TXD_IN),
        .data_out (rx_data),
        .data_valid(rx_valid)
    );

    // ================================================================
    // UART TX
    // ================================================================
    reg  [7:0] tx_data;
    reg        tx_start;
    wire       tx_busy;

    uart_tx #(
        .CLK_FREQ (CLK_FREQ),
        .BAUD_RATE(BAUD_RATE)
    ) u_uart_tx (
        .clk      (clk),
        .rst_n    (rst_n),
        .data_in  (tx_data),
        .data_valid(tx_start),
        .tx       (UART_RXD_OUT),
        .busy     (tx_busy)
    );

    // ================================================================
    // HDC Core
    // ================================================================
    reg                new_window;
    reg                sample_valid;
    reg  [INPUT_W-1:0] amp_reg;
    reg  [INPUT_W-1:0] pdiff_reg;

    wire [CLASS_W-1:0] result_class;
    wire [DIST_W-1:0]  result_dist;
    wire               result_valid_hdc;
    wire               hdc_busy;
    wire [7:0]         sample_count;

    hdc_core #(
        .INPUT_W     (INPUT_W),
        .Q_BITS      (Q_BITS),
        .CHUNK_W     (CHUNK_W),
        .NUM_CHUNKS  (NUM_CHUNKS),
        .CHUNK_ADDR_W(CHUNK_ADDR_W),
        .CB_DEPTH    (CB_DEPTH),
        .CB_ADDR_W   (CB_ADDR_W),
        .COUNTER_W   (COUNTER_W),
        .WINDOW_SIZE (WINDOW_SIZE),
        .NUM_CLASSES (NUM_CLASSES),
        .CLASS_W     (CLASS_W),
        .DIST_W      (DIST_W),
        .PROTO_DEPTH (PROTO_DEPTH),
        .PROTO_ADDR_W(PROTO_ADDR_W),
        .CB_A_HEX    (CB_A_HEX),
        .CB_B_HEX    (CB_B_HEX),
        .PROTO_HEX   (PROTO_HEX)
    ) u_hdc_core (
        .clk         (clk),
        .rst_n       (rst_n),
        .new_window  (new_window),
        .sample_valid(sample_valid),
        .amp_in      (amp_reg),
        .pdiff_in    (pdiff_reg),
        .result_class(result_class),
        .result_dist (result_dist),
        .result_valid(result_valid_hdc),
        .busy        (hdc_busy),
        .sample_count(sample_count)
    );

    // ================================================================
    // Command Protocol FSM
    // ================================================================
    localparam CMD_NEW_WINDOW = 8'h01;
    localparam CMD_LOAD_SAMPLE = 8'h02;
    localparam CMD_CLASSIFY   = 8'h03;

    localparam P_IDLE     = 3'd0;
    localparam P_CMD      = 3'd1;
    localparam P_AMP      = 3'd2;  // Waiting for amplitude byte
    localparam P_PDIFF    = 3'd3;  // Waiting for phase-diff byte
    localparam P_CLASSIFY = 3'd4;  // Waiting for classification result
    localparam P_SEND     = 3'd5;  // Sending result via UART

    reg [2:0] proto_state;
    reg [CLASS_W-1:0] last_result;
    reg [DIST_W-1:0]  last_dist;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            proto_state  <= P_IDLE;
            new_window   <= 1'b0;
            sample_valid <= 1'b0;
            amp_reg      <= 0;
            pdiff_reg    <= 0;
            tx_data      <= 0;
            tx_start     <= 1'b0;
            last_result  <= 0;
            last_dist    <= 0;
        end else begin
            new_window   <= 1'b0;
            sample_valid <= 1'b0;
            tx_start     <= 1'b0;

            case (proto_state)
                P_IDLE: begin
                    if (rx_valid) begin
                        case (rx_data)
                            CMD_NEW_WINDOW: begin
                                new_window <= 1'b1;
                                proto_state <= P_IDLE;
                            end
                            CMD_LOAD_SAMPLE: begin
                                proto_state <= P_AMP;
                            end
                            CMD_CLASSIFY: begin
                                proto_state <= P_CLASSIFY;
                            end
                            default: proto_state <= P_IDLE;
                        endcase
                    end
                end

                P_AMP: begin
                    if (rx_valid) begin
                        amp_reg <= rx_data;
                        proto_state <= P_PDIFF;
                    end
                end

                P_PDIFF: begin
                    if (rx_valid) begin
                        pdiff_reg    <= rx_data;
                        sample_valid <= 1'b1;
                        proto_state  <= P_IDLE;
                    end
                end

                P_CLASSIFY: begin
                    if (result_valid_hdc) begin
                        last_result <= result_class;
                        last_dist   <= result_dist;
                        // Send result byte (class ID)
                        tx_data <= {{(8-CLASS_W){1'b0}}, result_class};
                        tx_start <= 1'b1;
                        proto_state <= P_SEND;
                    end
                end

                P_SEND: begin
                    if (!tx_busy) begin
                        proto_state <= P_IDLE;
                    end
                end

                default: proto_state <= P_IDLE;
            endcase
        end
    end

    // ================================================================
    // LED Display
    // ================================================================
    always @(posedge clk) begin
        LED[3:0]   <= {{(4-CLASS_W){1'b0}}, last_result};  // Class result
        LED[7:4]   <= {hdc_busy, proto_state};              // Status
        LED[15:8]  <= sample_count;                          // Sample count
    end

    // ================================================================
    // 7-Segment Display — Show class number
    // ================================================================
    // Simple hex display on rightmost digit
    reg [3:0] hex_digit;

    always @(*) begin
        hex_digit = last_result[3:0];
        // 7-segment decoder (active low, pattern: gfedcba)
        case (hex_digit)
            4'h0: SEG = 7'b1000000;
            4'h1: SEG = 7'b1111001;
            4'h2: SEG = 7'b0100100;
            4'h3: SEG = 7'b0110000;
            4'h4: SEG = 7'b0011001;
            4'h5: SEG = 7'b0010010;
            4'h6: SEG = 7'b0000010;
            4'h7: SEG = 7'b1111000;
            4'h8: SEG = 7'b0000000;
            4'h9: SEG = 7'b0010000;
            4'hA: SEG = 7'b0001000;
            4'hB: SEG = 7'b0000011;
            4'hC: SEG = 7'b1000110;
            4'hD: SEG = 7'b0100001;
            4'hE: SEG = 7'b0000110;
            4'hF: SEG = 7'b0001110;
            default: SEG = 7'b1111111;
        endcase
        // Enable only rightmost digit
        AN = 8'b11111110;
    end

endmodule
