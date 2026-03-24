#!/bin/bash
# =============================================================================
# Run System Top Testbench (with UART + MMCM stub)
# =============================================================================
# Compiles and runs tb_system_top_sv.sv with Icarus Verilog.
# Uses mmcm_stub.v to replace the Xilinx MMCME2_BASE primitive.
#
# Simulation time: ~16.3 ms per test vector (UART limited)
# Vivado equivalent: run 500 ms
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/test_vectors"

echo "=== Compiling System Testbench (with UART) ==="
iverilog -g2012 -DSIMULATION -o tb_sys_sv \
    ../tb_system_top_sv.sv \
    ../../rtl/system_top.v \
    ../../rtl/wavebnn_core.v \
    ../../rtl/haar_wavelet_3lvl.v \
    ../../rtl/bnn_branch.v \
    ../../rtl/popcount.v \
    ../../rtl/bin_fc1.v \
    ../../rtl/fc_output.v \
    ../../rtl/uart_rx.v \
    ../../rtl/uart_tx.v

echo "=== Running Simulation ==="
vvp tb_sys_sv

echo "=== Done ==="
