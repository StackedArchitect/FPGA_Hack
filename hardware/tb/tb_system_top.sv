// =============================================================================
// Testbench: System Top-Level Architecture
// =============================================================================
// Simulates the entire FPGA architecture including clocks, UART, and the
// core classification engine.
//
// Features:
// - Uses SystemVerilog structure
// - Injects data via simulated UART serial line
// - Captures classification result via UART serial line
// - Logs exact timestamps for waveform inspection
//
// Usage:
//   iverilog -g2012 -o tb_system_top tb_system_top.sv ../rtl/*.v
//   vvp tb_system_top
// =============================================================================

`timescale 1ns / 1ps

module tb_system_top;

    // ── Parameters ──
    localparam CLK_125_PERIOD = 8.0;      // 125 MHz board clock
    localparam BAUD_RATE      = 115200;
    localparam BIT_PERIOD     = 1000000000 / BAUD_RATE; // ns per UART bit (~8680 ns)
    
    localparam NUM_SAMPLES    = 187; // Samples per ECG beat

    // ── Signals ──
    logic clk_125;
    logic rst_n_btn;
    logic uart_rxd;
    logic uart_txd;
    logic [3:0] led;

    // ── DUT ──
    // Note: We bypass the MMCM for standard sim or simulate a dummy one 
    // if Icarus doesn't support the primitive natively.
    // If MMCM is an issue, we can force the 100MHz clock inside the top wrapper.
    `ifdef SIM_BYPASS_MMCM
    // If you need to bypass xilinx primitives in raw iverilog:
    // This is often needed unless you compile with glbl and unisims
    `else
    system_top u_dut (
        .clk_125   (clk_125),
        .rst_n_btn (rst_n_btn),
        .uart_rxd  (uart_rxd),
        .uart_txd  (uart_txd),
        .led       (led)
    );
    `endif

    // ── Clock ──
    initial clk_125 = 0;
    always #(CLK_125_PERIOD/2.0) clk_125 = ~clk_125;

    // ── UART Task: Send Byte to FPGA ──
    task send_uart_byte(input [7:0] data);
        integer i;
        begin
            // Start bit
            uart_rxd = 0;
            #(BIT_PERIOD);
            // Data bits (LSB first)
            for (i = 0; i < 8; i++) begin
                uart_rxd = data[i];
                #(BIT_PERIOD);
            end
            // Stop bit
            uart_rxd = 1;
            #(BIT_PERIOD);
        end
    endtask

    // ── UART Task: Receive Byte from FPGA ──
    task receive_uart_byte(output [7:0] data);
        integer i;
        begin
            // Wait for start bit
            @(negedge uart_txd);
            #(BIT_PERIOD / 2); // Sample at middle of start bit
            
            if (uart_txd !== 0) $display("[%0t] Warning: False start bit detected!", $time);
            
            #(BIT_PERIOD); // Move to middle of bit 0
            
            // Read 8 data bits
            for (i = 0; i < 8; i++) begin
                data[i] = uart_txd;
                #(BIT_PERIOD);
            end
            
            // Wait for stop bit
            if (uart_txd !== 1) $display("[%0t] Warning: Framing error (no stop bit)!", $time);
        end
    endtask

    // ── Test Procedure ──
    logic [7:0] test_mem [0:NUM_SAMPLES-1];
    logic [2:0] expected_class;
    logic [7:0] rx_class;
    integer i;

    initial begin
        $dumpfile("tb_system_top.vcd");
        $dumpvars(0, tb_system_top);

        $display("\n=======================================================");
        $display("[%0t] SYSTEM TESTBENCH STARTED", $time);
        $display("=======================================================\n");

        // Load test data
        // NOTE: In Vivado Simulator, if the .mem files are added as Simulation Sources,
        // they are placed in the root of the run directory. So we just use the filename.
        $readmemh("test_input.mem", test_mem);
        // We'll just test the first 187 samples (1 ECG beat)
        expected_class = 3'b000; // Assuming class 0 for the first vector

        // Initialize
        uart_rxd  = 1; // Idle high
        rst_n_btn = 0; // Assert reset (active low depending on top module)

        // Reset phase
        $display("[%0t] Applying System Reset...", $time);
        #10000;
        rst_n_btn = 1; // Release reset
        $display("[%0t] Reset Released. Waiting for MMCM lock...", $time);
        
        // Wait for system to initialize (MMCM lock etc)
        // Note: MMCM simulation models in Vivado can take ~100us to achieve lock!.
        #150000;

        // --- Phase 1: TX Data to FPGA ---
        $display("[%0t] =======================================", $time);
        $display("[%0t] PHASE 1: TRANSMITTING 187 SAMPLES TO FPGA VIA UART", $time);
        for (i = 0; i < NUM_SAMPLES; i++) begin
            if (i == 0 || i == NUM_SAMPLES-1 || i % 50 == 0)
                $display("[%0t] Sending Sample %0d: 0x%h", $time, i, test_mem[i]);
            
            send_uart_byte(test_mem[i]);
        end
        $display("[%0t] Finished sending 187 samples.", $time);

        // --- Phase 2: Processing ---
        $display("[%0t] =======================================", $time);
        $display("[%0t] PHASE 2: INFERENCE PROCESSING", $time);
        $display("[%0t] Waiting for Core Inference to complete...", $time);
        
        // We can optionally monitor the busy LED (led[1]) if exposed
        
        // --- Phase 3: RX Data from FPGA ---
        $display("[%0t] =======================================", $time);
        $display("[%0t] PHASE 3: RECEIVING CLASSIFICATION RESULT VIA UART", $time);
        
        // Block until UART RX receives a byte
        receive_uart_byte(rx_class);
        
        $display("[%0t] Received Result Byte:  0x%h (Class %0d)", $time, rx_class, rx_class);
        $display("[%0t] Expected Result:      0x%h (Class %0d)", $time, expected_class, expected_class);

        if (rx_class[2:0] === expected_class) begin
            $display("\n[%0t] >>> TEST FULL SYSTEM: PASS <<<", $time);
        end else begin
            $display("\n[%0t] >>> TEST FULL SYSTEM: FAIL <<<", $time);
        end

        #1000;
        $display("[%0t] SYSTEM TESTBENCH FINISHED", $time);
        $finish;
    end

    // Watchdog
    initial begin
        #50_000_000; // 50ms total simulation bounding
        $display("[%0t] ERROR: System simulation timeout!", $time);
        $finish;
    end

endmodule
