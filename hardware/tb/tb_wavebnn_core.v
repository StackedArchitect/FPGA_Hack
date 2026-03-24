// =============================================================================
// Testbench: WaveBNN Core (end-to-end pipeline)
// =============================================================================
//
// Reads 187-sample ECG beat from test_vectors/ecg_input.mem
// Feeds to wavebnn_core, waits for classification, prints result.
//
// Usage (Icarus Verilog):
//   iverilog -o tb_wavebnn tb_wavebnn_core.v \
//       ../rtl/wavebnn_core.v ../rtl/haar_wavelet_3lvl.v \
//       ../rtl/bnn_branch.v ../rtl/popcount.v \
//       ../rtl/bin_fc1.v ../rtl/fc_output.v
//   vvp tb_wavebnn
//
// Weight .mem files must be in the simulation working directory.
// =============================================================================

`timescale 1ns / 1ps

module tb_wavebnn_core;

    // ── Parameters ──
    localparam CLK_PERIOD = 10;  // 100 MHz
    localparam NUM_SAMPLES = 187;
    localparam NUM_TESTS = 10;

    // ── Signals ──
    reg         clk;
    reg         rst_n;
    reg         start;
    reg  signed [7:0] sample;
    reg         sample_valid;

    wire        busy;
    wire        done;
    wire [2:0]  pred_class;

    // ── DUT ──
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

    // ── Clock ──
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // ── Test vector memory ──
    // Format: NUM_TESTS * NUM_SAMPLES entries, 8-bit hex
    reg [7:0] ecg_mem [0:NUM_TESTS*NUM_SAMPLES-1];
    reg [2:0] expected_class [0:NUM_TESTS-1];

    initial begin
        $readmemh("test_input.mem", ecg_mem);
        $readmemh("test_labels.mem", expected_class);
    end

    // ── Test procedure ──
    integer t, s;
    integer pass_cnt;
    integer cycle_cnt;

    initial begin
        $dumpfile("tb_wavebnn_core.vcd");
        $dumpvars(0, tb_wavebnn_core);

        rst_n        = 0;
        start        = 0;
        sample       = 0;
        sample_valid = 0;
        pass_cnt     = 0;

        // Reset
        repeat (10) @(posedge clk);
        rst_n = 1;
        repeat (5) @(posedge clk);

        $display("=== WaveBNN Core Testbench ===");
        $display("Running %0d test vectors...\n", NUM_TESTS);

        for (t = 0; t < NUM_TESTS; t = t + 1) begin
            // Start inference (#1 avoids delta-cycle race with combinational wav_start)
            @(posedge clk); #1;
            start = 1;
            sample = $signed(ecg_mem[t * NUM_SAMPLES]);
            sample_valid = 1;
            cycle_cnt = 0;

            @(posedge clk); #1;
            start = 0;

            // Feed remaining 186 samples (first already sent with start)
            for (s = 1; s < NUM_SAMPLES; s = s + 1) begin
                sample = $signed(ecg_mem[t * NUM_SAMPLES + s]);
                sample_valid = 1;
                @(posedge clk); #1;
                cycle_cnt = cycle_cnt + 1;
            end
            sample_valid = 0;

            // Wait for done
            while (!done) begin
                @(posedge clk);
                cycle_cnt = cycle_cnt + 1;
            end

            // Check result
            if (pred_class == expected_class[t]) begin
                $display("  Test %0d: PASS (class=%0d, latency=%0d cycles)",
                         t, pred_class, cycle_cnt);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("  Test %0d: FAIL (expected=%0d, got=%0d, latency=%0d)",
                         t, expected_class[t], pred_class, cycle_cnt);
            end

            // Brief pause between tests
            repeat (5) @(posedge clk);
        end

        $display("\n=== Results: %0d / %0d PASSED ===", pass_cnt, NUM_TESTS);

        if (pass_cnt == NUM_TESTS)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");

        $finish;
    end

    // ── Watchdog timer (prevent infinite hang) ───
    initial begin
        #(CLK_PERIOD * 2_000_000);
        $display("ERROR: Watchdog timeout!");
        $finish;
    end

endmodule
