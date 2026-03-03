// ============================================================
// Testbench: tb_hdc_core
// ============================================================
// Functional verification of the HDC core pipeline.
//
// 1. Loads test input vectors from hex files (exported by Python)
// 2. Feeds samples into hdc_core one at a time
// 3. Waits for classification result
// 4. Compares against expected class (from test_expected.hex)
// 5. Reports PASS/FAIL
//
// Run in Vivado:
//   1. Add all RTL sources + this testbench
//   2. Place .hex files in simulation working directory
//   3. Run behavioral simulation
// ============================================================

`timescale 1ns / 1ps

module tb_hdc_core;

    // ============================================================
    // Parameters — match your exported config
    // ============================================================
    parameter INPUT_W       = 8;
    parameter Q_BITS        = 4;
    parameter CHUNK_W       = 32;
    parameter NUM_CHUNKS    = 128;
    parameter CHUNK_ADDR_W  = 7;
    parameter CB_DEPTH      = 2048;
    parameter CB_ADDR_W     = 11;
    parameter COUNTER_W     = 8;
    parameter WINDOW_SIZE   = 128;
    parameter NUM_CLASSES   = 11;
    parameter CLASS_W       = 4;
    parameter DIST_W        = 13;
    parameter PROTO_DEPTH   = 1408;
    parameter PROTO_ADDR_W  = 11;
    parameter NUM_TEST_WINDOWS = 20;  // Number of test windows in hex file

    parameter CB_A_HEX  = "codebook_i.hex";
    parameter CB_B_HEX  = "codebook_q.hex";
    parameter PROTO_HEX = "prototypes.hex";

    // 10 ns period = 100 MHz
    parameter CLK_PERIOD = 10;

    // ============================================================
    // Signals
    // ============================================================
    reg                clk;
    reg                rst_n;
    reg                new_window;
    reg                sample_valid;
    reg  [INPUT_W-1:0] amp_in;
    reg  [INPUT_W-1:0] pdiff_in;
    wire [CLASS_W-1:0] result_class;
    wire [DIST_W-1:0]  result_dist;
    wire               result_valid;
    wire               busy;
    wire [7:0]         sample_count;

    // ============================================================
    // DUT
    // ============================================================
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
    ) dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .new_window  (new_window),
        .sample_valid(sample_valid),
        .amp_in      (amp_in),
        .pdiff_in    (pdiff_in),
        .result_class(result_class),
        .result_dist (result_dist),
        .result_valid(result_valid),
        .busy        (busy),
        .sample_count(sample_count)
    );

    // ============================================================
    // Clock generation
    // ============================================================
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // ============================================================
    // Test vector storage
    // ============================================================
    // Each test window has WINDOW_SIZE entries, each entry is {amp[7:0], pdiff[7:0]} = 16 bits
    // Total entries = NUM_TEST_WINDOWS * WINDOW_SIZE
    reg [15:0] test_inputs [0:NUM_TEST_WINDOWS*WINDOW_SIZE-1];
    reg [7:0]  test_expected [0:NUM_TEST_WINDOWS-1];

    initial begin
        $readmemh("test_input.hex", test_inputs);
        $readmemh("test_expected.hex", test_expected);
    end

    // ============================================================
    // Test process
    // ============================================================
    integer win_idx;
    integer samp_idx;
    integer base_idx;
    integer pass_count;
    integer fail_count;
    integer total_count;

    reg [CLASS_W-1:0] expected_class;
    reg [CLASS_W-1:0] got_class;

    task feed_one_window;
        input integer window_num;
        begin
            base_idx = window_num * WINDOW_SIZE;

            // Assert new_window
            @(posedge clk);
            new_window <= 1'b1;
            @(posedge clk);
            new_window <= 1'b0;

            // Wait for bundler clear to finish (NUM_CHUNKS + overhead)
            repeat (NUM_CHUNKS + 10) @(posedge clk);

            // Feed all samples
            for (samp_idx = 0; samp_idx < WINDOW_SIZE; samp_idx = samp_idx + 1) begin
                amp_in       <= test_inputs[base_idx + samp_idx][15:8];
                pdiff_in     <= test_inputs[base_idx + samp_idx][7:0];
                sample_valid <= 1'b1;
                @(posedge clk);
                sample_valid <= 1'b0;

                // Wait for encoder to finish this sample
                // (~NUM_CHUNKS * 4 cycles for all chunks, with PAUSE state)
                repeat (NUM_CHUNKS * 4 + 10) @(posedge clk);
            end
        end
    endtask

    task wait_for_result;
        output [CLASS_W-1:0] cls;
        begin
            // Wait for result_valid to go high (with timeout)
            fork : result_wait
                begin
                    wait (result_valid);
                end
                begin
                    repeat (200000) @(posedge clk);
                    $display("ERROR: Timeout waiting for result at time %0t", $time);
                end
            join_any
            disable result_wait;

            cls = result_class;
        end
    endtask

    initial begin
        // Initialize
        rst_n        = 0;
        new_window   = 0;
        sample_valid = 0;
        amp_in       = 0;
        pdiff_in     = 0;
        pass_count   = 0;
        fail_count   = 0;
        total_count  = 0;

        // Reset
        repeat (20) @(posedge clk);
        rst_n = 1;
        repeat (10) @(posedge clk);

        $display("==============================================");
        $display("    HDC Core Testbench — Starting");
        $display("    D=%0d, CHUNK_W=%0d, WINDOW=%0d, Classes=%0d",
                 NUM_CHUNKS * CHUNK_W, CHUNK_W, WINDOW_SIZE, NUM_CLASSES);
        $display("==============================================");

        for (win_idx = 0; win_idx < NUM_TEST_WINDOWS; win_idx = win_idx + 1) begin
            expected_class = test_expected[win_idx][CLASS_W-1:0];

            $display("[Window %0d] Feeding %0d samples, Expected class: %0d",
                     win_idx, WINDOW_SIZE, expected_class);

            feed_one_window(win_idx);
            wait_for_result(got_class);

            total_count = total_count + 1;
            if (got_class == expected_class) begin
                pass_count = pass_count + 1;
                $display("[Window %0d] PASS — Got class %0d (dist=%0d)",
                         win_idx, got_class, result_dist);
            end else begin
                fail_count = fail_count + 1;
                $display("[Window %0d] FAIL — Got class %0d, Expected %0d (dist=%0d)",
                         win_idx, got_class, expected_class, result_dist);
            end

            // Small gap between windows
            repeat (10) @(posedge clk);
        end

        $display("==============================================");
        $display("    Test Complete");
        $display("    Total: %0d  Pass: %0d  Fail: %0d", total_count, pass_count, fail_count);
        $display("    Accuracy: %0d%%", (pass_count * 100) / total_count);
        $display("==============================================");

        $finish;
    end

    // ============================================================
    // Waveform dump (for GTKWave or Vivado waveform viewer)
    // ============================================================
    initial begin
        $dumpfile("tb_hdc_core.vcd");
        $dumpvars(0, tb_hdc_core);
    end

    // ============================================================
    // Watchdog timer
    // ============================================================
    initial begin
        #100_000_000; // 100 ms timeout
        $display("WATCHDOG: Simulation timeout at 100 ms");
        $finish;
    end

endmodule
