// =============================================================================
// Simulation Stub: MMCME2_BASE (Xilinx 7-Series)
// =============================================================================
// Lightweight behavioral replacement for Icarus Verilog simulation.
// Passes CLKIN1 directly to CLKOUT0 and asserts LOCKED after a short delay.
//
// Parameters mirror the real MMCME2_BASE interface so system_top.v compiles
// without modification.
// =============================================================================

module MMCME2_BASE #(
    parameter real CLKIN1_PERIOD   = 8.000,
    parameter real CLKFBOUT_MULT_F = 8.0,
    parameter real CLKOUT0_DIVIDE_F = 10.0
)(
    input  wire CLKIN1,
    input  wire CLKFBIN,
    output wire CLKFBOUT,
    output wire CLKOUT0,
    output reg  LOCKED,
    input  wire PWRDWN,
    input  wire RST
);

    // Feedback loopback (passthrough)
    assign CLKFBOUT = CLKFBIN;

    // Clock output: pass CLKIN1 directly (no actual PLL in simulation)
    assign CLKOUT0 = CLKIN1;

    // Lock behavior: assert LOCKED after ~200ns (mimics real MMCM settling)
    initial LOCKED = 1'b0;

    always @(posedge CLKIN1 or posedge RST) begin
        if (RST)
            LOCKED <= 1'b0;
        else
            // In real hardware, MMCM takes ~100µs to lock.
            // In sim, we just need a few cycles for the reset synchronizer.
            #200 LOCKED <= 1'b1;
    end

endmodule
