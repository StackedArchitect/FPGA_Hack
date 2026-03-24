// =============================================================================
// Testbench: System Top (Full Architecture with UART)
// =============================================================================
//
// End-to-end test of the complete system_top module:
//   PC(TB) --UART TX--> system_top --inference--> system_top --UART RX--> PC(TB)
//
// Clock Strategy:
//   - Vivado: TB drives 200 MHz differential clock on sys_clk_p/sys_clk_n.
//             The DUT's IBUFDS + MMCM produces 100 MHz internally.
//   - Icarus: Uses -DSIMULATION flag to bypass IBUFDS/MMCM.
//             TB drives 100 MHz directly on sys_clk_p.
//
// Simulation time: ~16.3 ms per test vector (UART limited)
// Vivado: run 50 ms (1 vector) or run 100 ms (2 vectors)
//
// Usage (Icarus):
//   cd hardware/tb/test_vectors
//   iverilog -g2012 -DSIMULATION -o tb_sys_sv \
//       ../tb_system_top_sv.sv \
//       ../../rtl/system_top.v ../../rtl/wavebnn_core.v \
//       ../../rtl/haar_wavelet_3lvl.v ../../rtl/bnn_branch.v \
//       ../../rtl/popcount.v ../../rtl/bin_fc1.v ../../rtl/fc_output.v \
//       ../../rtl/uart_rx.v ../../rtl/uart_tx.v
//   vvp tb_sys_sv
// =============================================================================

`timescale 1ns / 1ps

module tb_system_top_sv;

    // =========================================================================
    // Parameters
    // =========================================================================
`ifdef SIMULATION
    // Icarus: drive 100 MHz directly (MMCM bypassed)
    localparam real  CLK_PERIOD     = 10.0;         // 100 MHz
    localparam       USE_DIFF_CLK   = 0;
`else
    // Vivado: drive 200 MHz differential (real IBUFDS + MMCM path)
    localparam real  CLK_PERIOD     = 5.0;          // 200 MHz
    localparam       USE_DIFF_CLK   = 1;
`endif

    localparam int   BAUD_RATE      = 115200;
    localparam int   BIT_PERIOD     = 8681;         // ~1e9/115200 ns
    localparam int   NUM_SAMPLES    = 187;
    localparam int   NUM_TESTS      = 2;
    localparam int   INTER_BYTE_GAP = 0;

    // =========================================================================
    // Signals
    // =========================================================================
    logic        clk;
    logic        rst_n_btn;
    logic        uart_rxd;       // TB -> DUT
    wire         uart_txd;       // DUT -> TB
    wire  [3:0]  led;

    // Differential clock pair
    wire sys_clk_p_w, sys_clk_n_w;
    assign sys_clk_p_w = clk;
    assign sys_clk_n_w = ~clk;

    // =========================================================================
    // DUT
    // =========================================================================
    system_top u_dut (
        .sys_clk_p (sys_clk_p_w),
        .sys_clk_n (sys_clk_n_w),
        .rst_n_btn (rst_n_btn),
        .uart_rxd  (uart_rxd),
        .uart_txd  (uart_txd),
        .led       (led)
    );

    // =========================================================================
    // Clock Generation
    // =========================================================================
    initial clk = 0;
    always #(CLK_PERIOD / 2.0) clk = ~clk;

    // =========================================================================
    // UART TX Task: Send one byte to DUT (8N1, LSB first)
    // =========================================================================
    task automatic send_uart_byte(input [7:0] data);
        int i;
        uart_rxd = 1'b0;         // Start bit
        #(BIT_PERIOD);
        for (i = 0; i < 8; i++) begin
            uart_rxd = data[i];   // Data bits LSB first
            #(BIT_PERIOD);
        end
        uart_rxd = 1'b1;         // Stop bit
        #(BIT_PERIOD);
        if (INTER_BYTE_GAP > 0) #(INTER_BYTE_GAP);
    endtask

    // =========================================================================
    // UART RX Task: Receive one byte from DUT (8N1, LSB first)
    // =========================================================================
    task automatic receive_uart_byte(output [7:0] data);
        int i;
        @(negedge uart_txd);      // Wait for start bit
        #(BIT_PERIOD / 2);        // Move to middle of start bit
        if (uart_txd !== 1'b0)
            $display("[%0t] WARNING: False start bit!", $time);
        #(BIT_PERIOD);            // Move to middle of first data bit
        for (i = 0; i < 8; i++) begin
            data[i] = uart_txd;
            if (i < 7) #(BIT_PERIOD);
        end
        #(BIT_PERIOD);            // Move to stop bit
        if (uart_txd !== 1'b1)
            $display("[%0t] WARNING: Framing error!", $time);
    endtask

    // =========================================================================
    // Test Vector Memory
    // =========================================================================
    logic [7:0] test_mem       [0:NUM_TESTS*NUM_SAMPLES-1];
    logic [2:0] expected_class [0:NUM_TESTS-1];

    initial begin
        $readmemh("test_input.mem",  test_mem);
        $readmemh("test_labels.mem", expected_class);
    end

    // =========================================================================
    // LED Monitor
    // =========================================================================
    logic [3:0] prev_led;

    always @(led) begin
        if (led !== prev_led && led !== 4'bxxxx) begin
            $display("[%0t] LED: heartbeat=%b busy=%b done=%b rx_activity=%b",
                     $time, led[0], led[1], led[2], led[3]);
            prev_led <= led;
        end
    end

    // =========================================================================
    // Main Test Procedure
    // =========================================================================
    int t, i;
    int pass_cnt;
    logic [7:0] rx_byte;

    initial begin
        $dumpfile("tb_system_top_sv.vcd");
        $dumpvars(0, tb_system_top_sv);

        // ── Initialize ──
        uart_rxd  = 1'b1;
        rst_n_btn = 1'b0;
        pass_cnt  = 0;
        prev_led  = 4'bxxxx;

        $display("");
        $display("=============================================================");
        $display(" WaveBNN-ECG System Top Testbench (with UART)");
        $display("=============================================================");
        if (USE_DIFF_CLK)
            $display(" Clock:        200 MHz differential -> MMCM -> 100 MHz");
        else
            $display(" Clock:        100 MHz direct (MMCM bypassed)");
        $display(" UART Baud:    %0d", BAUD_RATE);
        $display(" Bit Period:   ~%0d ns", BIT_PERIOD);
        $display(" Test Vectors: %0d", NUM_TESTS);
        $display(" Samples/beat: %0d", NUM_SAMPLES);
        $display("=============================================================");
        $display("");

        // ── Reset Phase ──
        $display("[%0t] Applying system reset...", $time);
        #10000;                    // 10 µs reset hold
        rst_n_btn = 1'b1;
        $display("[%0t] Reset released. Waiting for MMCM lock...", $time);

        // Wait for MMCM to lock (real Vivado MMCM: ~50-100 µs)
        wait (u_dut.mmcm_locked === 1'b1);
        $display("[%0t] MMCM locked.", $time);

        // Wait for reset synchronizer (3 cycles + margin)
        #500;
        $display("[%0t] System ready (rst_n=%b).", $time, u_dut.rst_n);
        $display("");

        // ── Run Test Vectors ──
        for (t = 0; t < NUM_TESTS; t++) begin
            $display("=============================================================");
            $display("[%0t] TEST %0d: Sending %0d samples via UART (expected class: %0d)",
                     $time, t, NUM_SAMPLES, expected_class[t]);
            $display("=============================================================");

            // ─── Phase 1+2+3: TX samples + RX result (concurrent) ───
            // The DUT's inference may finish before the last TX byte ends,
            // so we must listen for the result byte concurrently with TX.
            $display("[%0t] PHASE 1: UART TX (%0d bytes @ %0d baud)", $time, NUM_SAMPLES, BAUD_RATE);
            $display("[%0t]          (RX listener active concurrently)", $time);

            fork
                // --- Thread 1: Send all UART bytes ---
                begin
                    for (i = 0; i < NUM_SAMPLES; i++) begin
                        if (i == 0 || i == NUM_SAMPLES - 1 || i % 50 == 0)
                            $display("[%0t]   Byte %3d / %0d : 0x%02h", $time, i, NUM_SAMPLES, test_mem[t * NUM_SAMPLES + i]);
                        send_uart_byte(test_mem[t * NUM_SAMPLES + i]);
                    end
                    $display("[%0t]   TX complete (%0d bytes sent).", $time, NUM_SAMPLES);
                end

                // --- Thread 2: Listen for result byte ---
                begin
                    receive_uart_byte(rx_byte);
                    $display("[%0t]   RX received: 0x%02h (class=%0d)", $time, rx_byte, rx_byte[2:0]);
                end
            join

            $display("[%0t]   Received byte:   0x%02h (class = %0d)", $time, rx_byte, rx_byte[2:0]);
            $display("[%0t]   Expected class:   %0d", $time, expected_class[t]);

            if (rx_byte[2:0] === expected_class[t]) begin
                $display("[%0t]   >>> TEST %0d: PASS <<<", $time, t);
                pass_cnt++;
            end else begin
                $display("[%0t]   >>> TEST %0d: *** FAIL *** <<<", $time, t);
            end
            $display("");
            #(BIT_PERIOD * 2);
        end

        // ── Summary ──
        $display("=============================================================");
        $display(" SUMMARY: %0d / %0d PASSED", pass_cnt, NUM_TESTS);
        $display("=============================================================");
        $display("");

        if (pass_cnt == NUM_TESTS)
            $display(" >>> ALL SYSTEM TESTS PASSED <<<");
        else
            $display(" >>> SOME SYSTEM TESTS FAILED <<<");

        $display("");
        #1000;
        $finish;
    end

    // =========================================================================
    // Watchdog Timer
    // =========================================================================
    initial begin
        #500_000_000;
        $display("");
        $display("[%0t] FATAL: Watchdog timeout (500 ms)!", $time);
        $display("  led=%b  Core state=%0d  wav_state=%0d  wav_cnt=%0d",
                 led, u_dut.u_core.state, u_dut.u_core.u_wavelet.state,
                 u_dut.u_core.u_wavelet.sample_cnt);
        $display("  sys_cnt=%0d  result_ready=%b  tx_busy=%b  mmcm_locked=%b  rst_n=%b",
                 u_dut.sample_cnt, u_dut.result_ready, u_dut.tx_busy,
                 u_dut.mmcm_locked, u_dut.rst_n);
        $finish;
    end

endmodule
