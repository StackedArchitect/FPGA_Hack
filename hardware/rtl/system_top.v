// =============================================================================
// WaveBNN-ECG: System Top (ZC702 Evaluation Board)
// =============================================================================
//
// Protocol:
//   PC -> FPGA: 187 bytes (int8 ECG samples, unsigned over UART)
//   FPGA -> PC: 1 byte (class 0-4)
//
// Data flow:
//   uart_rx -> sample counter -> wavebnn_core -> uart_tx
//
// Status LEDs:
//   led[0] = alive (heartbeat blink ~1 Hz)
//   led[1] = busy (inference running)
//   led[2] = done (last result valid)
//   led[3] = rx_activity (blink on byte received)
//
// Target: ZC702 (xc7z020clg484-1) @ 200 MHz LVDS input, 100 MHz via MMCM
// =============================================================================

module system_top (
    input  wire       sys_clk_p,     // 200 MHz LVDS+ (ZC702 Bank 35)
    input  wire       sys_clk_n,     // 200 MHz LVDS-
    input  wire       rst_n_btn,     // active-low pushbutton (GPIO_SW_N)

    // -- UART (PMOD1 J62) --
    input  wire       uart_rxd,      // UART RX (from PC)
    output wire       uart_txd,      // UART TX (to PC)

    // -- Status LEDs (DS15-DS18) --
    output wire [3:0] led
);

    // ---------------------------------------------
    // Clock generation: 200 MHz LVDS -> 100 MHz via MMCM
    // In simulation, bypass IBUFDS+MMCM (TB provides 100 MHz directly)
    // ---------------------------------------------
    wire clk_100, clk_fb, mmcm_locked;

`ifdef SIMULATION
    // ── Simulation bypass: TB drives 100 MHz on sys_clk_p port ──
    assign clk_100     = sys_clk_p;
    assign mmcm_locked = 1'b1;
`else
    // ── Synthesis: IBUFDS + MMCM ──
    wire clk_200;
    IBUFDS u_ibufds (
        .I  (sys_clk_p),
        .IB (sys_clk_n),
        .O  (clk_200)
    );

    MMCME2_BASE #(
        .CLKIN1_PERIOD  (5.000),   // 200 MHz = 5 ns
        .CLKFBOUT_MULT_F(5.0),    // VCO = 200 * 5 = 1000 MHz
        .CLKOUT0_DIVIDE_F(10.0)   // 1000 / 10 = 100 MHz
    ) u_mmcm (
        .CLKIN1  (clk_200),
        .CLKFBIN (clk_fb),
        .CLKFBOUT(clk_fb),
        .CLKOUT0 (clk_100),
        .LOCKED  (mmcm_locked),
        .PWRDWN  (1'b0),
        .RST     (~rst_n_btn)
    );
`endif

    wire clk = clk_100;

    // --- Reset synchronizer ---
    reg [2:0] rst_sync;
    wire rst_n = rst_sync[2];

    always @(posedge clk or negedge mmcm_locked) begin
        if (!mmcm_locked)
            rst_sync <= 3'b000;
        else
            rst_sync <= {rst_sync[1:0], 1'b1};
    end

    // ---------------------------------------------
    // UART RX
    // ---------------------------------------------
    wire       rx_valid;
    wire [7:0] rx_data;

    uart_rx #(
        .CLK_FREQ  (100_000_000),
        .BAUD_RATE (115_200)
    ) u_uart_rx (
        .clk    (clk),
        .rst_n  (rst_n),
        .i_rx   (uart_rxd),
        .o_valid(rx_valid),
        .o_data (rx_data)
    );

    // ---------------------------------------------
    // Sample ingestion: collect 187 bytes -> start inference
    // ---------------------------------------------
    localparam NUM_SAMPLES = 187;

    reg [7:0] sample_cnt;
    reg       core_start;
    reg       core_sample_valid;
    (* max_fanout = 32 *)
    reg signed [7:0] core_sample;

    wire core_busy;
    wire core_done;
    wire [2:0] core_class;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sample_cnt        <= 8'd0;
            core_start        <= 1'b0;
            core_sample_valid <= 1'b0;
            core_sample       <= 8'sd0;
        end else begin
            core_start        <= 1'b0;
            core_sample_valid <= 1'b0;

            // FIX: Allow ingestion to continue even if core_busy is 1, as long as we are
            // in the middle of our 187-sample streak. 
            if (rx_valid && (!core_busy || sample_cnt != 8'd0)) begin
                // Reinterpret unsigned UART byte as signed int8
                core_sample       <= $signed(rx_data);
                core_sample_valid <= 1'b1;

                if (sample_cnt == 0) begin
                    // First byte -> start wavelet ingestion
                    core_start <= 1'b1;
                end

                if (sample_cnt == NUM_SAMPLES - 1)
                    sample_cnt <= 8'd0;
                else
                    sample_cnt <= sample_cnt + 8'd1;
            end
        end
    end

    // ---------------------------------------------
    // WaveBNN Inference Core
    // ---------------------------------------------

    wavebnn_core u_core (
        .clk            (clk),
        .rst_n          (rst_n),
        .i_start        (core_start),
        .i_sample       (core_sample),
        .i_sample_valid (core_sample_valid),
        .o_busy         (core_busy),
        .o_done         (core_done),
        .o_class        (core_class)
    );

    // ---------------------------------------------
    // UART TX: send result
    // ---------------------------------------------
    reg       tx_valid;
    reg [7:0] tx_data;
    wire      tx_busy;
    reg [2:0] last_class;
    reg       result_ready;

    uart_tx #(
        .CLK_FREQ  (100_000_000),
        .BAUD_RATE (115_200)
    ) u_uart_tx (
        .clk    (clk),
        .rst_n  (rst_n),
        .i_valid(tx_valid),
        .i_data (tx_data),
        .o_tx   (uart_txd),
        .o_busy (tx_busy)
    );

    // Latch result and send when TX is free
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx_valid     <= 1'b0;
            tx_data      <= 8'd0;
            last_class   <= 3'd0;
            result_ready <= 1'b0;
        end else begin
            tx_valid <= 1'b0;

            if (core_done) begin
                last_class   <= core_class;
                result_ready <= 1'b1;
            end

            if (result_ready && !tx_busy) begin
                tx_valid     <= 1'b1;
                tx_data      <= {5'd0, last_class};
                result_ready <= 1'b0;
            end
        end
    end

    // ---------------------------------------------
    // Status LEDs
    // ---------------------------------------------
    reg [26:0] heartbeat_cnt;
    reg        rx_activity;
    reg [19:0] rx_blink_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            heartbeat_cnt <= 27'd0;
            rx_activity   <= 1'b0;
            rx_blink_cnt  <= 20'd0;
        end else begin
            heartbeat_cnt <= heartbeat_cnt + 1;

            if (rx_valid) begin
                rx_activity  <= 1'b1;
                rx_blink_cnt <= 20'hFFFFF;
            end else if (rx_blink_cnt != 0) begin
                rx_blink_cnt <= rx_blink_cnt - 1;
            end else begin
                rx_activity <= 1'b0;
            end
        end
    end

    assign led[0] = heartbeat_cnt[26];     // ~0.75 Hz blink
    assign led[1] = core_busy;             // inference active
    assign led[2] = result_ready | core_done; // result available
    assign led[3] = rx_activity;           // RX activity

endmodule
