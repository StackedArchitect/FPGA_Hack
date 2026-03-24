// =============================================================================
// Testbench: WaveBNN Core — SystemVerilog (No UART)
// =============================================================================
//
// Comprehensive testbench for wavebnn_core:
//   - Loads test vectors from .mem files
//   - Feeds 187-sample ECG beats directly to the core
//   - Monitors all pipeline stages with timestamps
//   - Tracks per-test latency and pass/fail status
//   - Provides detailed debug output on failure
//
// Simulation time: ~100µs for 10 vectors @ 100 MHz
// Vivado: run 100 ms  |  Icarus: self-terminating
//
// Usage (Icarus):
//   cd hardware/tb/test_vectors
//   iverilog -g2012 -o tb_core_sv \
//       ../tb_wavebnn_core_sv.sv \
//       ../../rtl/wavebnn_core.v ../../rtl/haar_wavelet_3lvl.v \
//       ../../rtl/bnn_branch.v ../../rtl/popcount.v \
//       ../../rtl/bin_fc1.v ../../rtl/fc_output.v
//   vvp tb_core_sv
// =============================================================================

`timescale 1ns / 1ps

module tb_wavebnn_core_sv;

    // =========================================================================
    // Parameters
    // =========================================================================
    localparam real  CLK_PERIOD  = 10.0;   // 100 MHz
    localparam int   NUM_SAMPLES = 187;
    localparam int   NUM_TESTS   = 10;
    localparam int   WATCHDOG_CYCLES = 2_000_000;

    // FSM state names (must match wavebnn_core.v localparams)
    localparam int S_IDLE    = 0;
    localparam int S_WAVELET = 1;
    localparam int S_FEED    = 2;
    localparam int S_WAIT_BR = 3;
    localparam int S_FC1     = 4;
    localparam int S_FC2     = 5;
    localparam int S_DONE    = 6;

    // =========================================================================
    // Signals
    // =========================================================================
    logic        clk;
    logic        rst_n;
    logic        start;
    logic signed [7:0] sample;
    logic        sample_valid;

    wire         busy;
    wire         done;
    wire  [2:0]  pred_class;

    // =========================================================================
    // DUT
    // =========================================================================
    wavebnn_core u_dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .i_start        (start),
        .i_sample       (sample),
        .i_sample_valid (sample_valid),
        .o_busy         (busy),
        .o_done         (done),
        .o_class        (pred_class)
    );

    // =========================================================================
    // Clock Generation
    // =========================================================================
    initial clk = 0;
    always #(CLK_PERIOD / 2.0) clk = ~clk;

    // =========================================================================
    // Test Vector Memory
    // =========================================================================
    logic [7:0] ecg_mem       [0:NUM_TESTS*NUM_SAMPLES-1];
    logic [2:0] expected_class [0:NUM_TESTS-1];

    initial begin
        $readmemh("test_input.mem",  ecg_mem);
        $readmemh("test_labels.mem", expected_class);
    end

    // =========================================================================
    // FSM State Name Helper
    // =========================================================================
    function string state_name(input [2:0] s);
        case (s)
            S_IDLE:    return "IDLE";
            S_WAVELET: return "WAVELET";
            S_FEED:    return "FEED";
            S_WAIT_BR: return "WAIT_BR";
            S_FC1:     return "FC1";
            S_FC2:     return "FC2";
            S_DONE:    return "DONE";
            default:   return "???";
        endcase
    endfunction

    // =========================================================================
    // Pipeline Stage Monitors
    // =========================================================================
    logic [2:0] prev_state;

    always @(posedge clk) begin
        if (rst_n && u_dut.state !== prev_state) begin
            $display("[%0t]   FSM: %s -> %s", $time,
                     state_name(prev_state), state_name(u_dut.state));
            prev_state <= u_dut.state;
        end
    end

    always @(posedge clk) begin
        if (rst_n) begin
            if (u_dut.wav_done)
                $display("[%0t]   >> Wavelet DWT complete", $time);
            if (u_dut.br_cA3_done)
                $display("[%0t]   >> Branch cA3 done", $time);
            if (u_dut.br_cD3_done)
                $display("[%0t]   >> Branch cD3 done", $time);
            if (u_dut.br_cD2_done)
                $display("[%0t]   >> Branch cD2 done", $time);
            if (u_dut.br_cD1_done)
                $display("[%0t]   >> Branch cD1 done", $time);
            if (u_dut.fc1_done)
                $display("[%0t]   >> FC1 (2048->128) complete", $time);
            if (u_dut.fc2_done)
                $display("[%0t]   >> FC2 (128->5) + argmax complete", $time);
        end
    end

    // =========================================================================
    // Main Test Procedure
    // =========================================================================
    int t, s;
    int pass_cnt;
    int cycle_cnt;
    int latencies [0:NUM_TESTS-1];

    initial begin
        $dumpfile("tb_wavebnn_core_sv.vcd");
        $dumpvars(0, tb_wavebnn_core_sv);

        // ── Initialize ──
        rst_n        = 0;
        start        = 0;
        sample       = 8'sd0;
        sample_valid = 0;
        pass_cnt     = 0;
        prev_state   = 3'd0;

        // ── Reset ──
        repeat (10) @(posedge clk);
        rst_n = 1;
        repeat (5) @(posedge clk);

        $display("");
        $display("=============================================================");
        $display(" WaveBNN Core SystemVerilog Testbench");
        $display("=============================================================");
        $display(" Clock:        100 MHz (%0.1f ns period)", CLK_PERIOD);
        $display(" Test Vectors: %0d", NUM_TESTS);
        $display(" Samples/beat: %0d", NUM_SAMPLES);
        $display("=============================================================");
        $display("");

        // ── Run Test Vectors ──
        for (t = 0; t < NUM_TESTS; t++) begin
            $display("─────────────────────────────────────────────────────────────");
            $display("[%0t] TEST %0d: Feeding %0d samples (expected class: %0d)",
                     $time, t, NUM_SAMPLES, expected_class[t]);
            $display("─────────────────────────────────────────────────────────────");

            // Start + first sample (use #1 to avoid delta-cycle race with
            // combinational wav_start signal in wavebnn_core)
            @(posedge clk); #1;
            start        = 1;
            sample       = $signed(ecg_mem[t * NUM_SAMPLES]);
            sample_valid = 1;
            cycle_cnt    = 0;

            @(posedge clk); #1;
            start = 0;

            // Feed remaining 186 samples (one per clock cycle)
            for (s = 1; s < NUM_SAMPLES; s++) begin
                sample       = $signed(ecg_mem[t * NUM_SAMPLES + s]);
                sample_valid = 1;
                @(posedge clk); #1;
                cycle_cnt++;
            end
            sample_valid = 0;

            // Wait for inference to complete
            while (!done) begin
                @(posedge clk);
                cycle_cnt++;
                if (cycle_cnt > 50000) begin
                    $display("[%0t] ERROR: Test %0d TIMEOUT (>50000 cycles)", $time, t);
                    $display("  core_state  = %s (%0d)", state_name(u_dut.state), u_dut.state);
                    $display("  wav_busy=%b wav_done=%b", u_dut.wav_busy, u_dut.wav_done);
                    $display("  feed_addr=%0d feed_active=%b", u_dut.feed_addr, u_dut.feed_active);
                    $display("  br_finished: cA3=%b cD3=%b cD2=%b cD1=%b",
                             u_dut.br_cA3_finished, u_dut.br_cD3_finished,
                             u_dut.br_cD2_finished, u_dut.br_cD1_finished);
                    $finish;
                end
            end

            latencies[t] = cycle_cnt;

            // Check result
            if (pred_class === expected_class[t]) begin
                $display("[%0t]   RESULT: PASS (class=%0d, latency=%0d cycles)",
                         $time, pred_class, cycle_cnt);
                pass_cnt++;
            end else begin
                $display("[%0t]   RESULT: *** FAIL *** (expected=%0d, got=%0d, latency=%0d)",
                         $time, expected_class[t], pred_class, cycle_cnt);
            end
            $display("");

            // Brief pause between tests
            repeat (5) @(posedge clk);
        end

        // ── Summary ──
        $display("=============================================================");
        $display(" SUMMARY");
        $display("=============================================================");
        $display("");
        $display(" Test | Expected | Got | Latency | Status");
        $display(" -----|----------|-----|---------|-------");
        for (t = 0; t < NUM_TESTS; t++) begin
            $display("  %2d  |    %0d     |  %0d  |  %4d   | %s",
                     t, expected_class[t],
                     (t < NUM_TESTS) ? ecg_mem[0] : 0, // placeholder; real class printed below
                     latencies[t],
                     "see above");
        end
        $display("");
        $display(" Results: %0d / %0d PASSED", pass_cnt, NUM_TESTS);
        $display("");

        if (pass_cnt == NUM_TESTS)
            $display(" >>> ALL TESTS PASSED <<<");
        else
            $display(" >>> SOME TESTS FAILED <<<");

        $display("=============================================================");
        $finish;
    end

    // =========================================================================
    // Hard Watchdog
    // =========================================================================
    initial begin
        #(CLK_PERIOD * WATCHDOG_CYCLES);
        $display("");
        $display("[%0t] FATAL: Watchdog timeout after %0d cycles!", $time, WATCHDOG_CYCLES);
        $display("  core_state = %s (%0d)", state_name(u_dut.state), u_dut.state);
        $finish;
    end

endmodule
