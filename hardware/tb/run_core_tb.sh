#!/bin/bash
# =============================================================================
# Run WaveBNN Core Testbench (no UART)
# =============================================================================
# Compiles and runs tb_wavebnn_core_sv.sv with Icarus Verilog.
# Must be run from the hardware/tb directory (or it cd's there).
#
# Simulation time: ~100 µs for 10 vectors
# Vivado equivalent: run 100 ms
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/test_vectors"

echo "=== Compiling Core Testbench (no UART) ==="
iverilog -g2012 -o tb_core_sv \
    ../tb_wavebnn_core_sv.sv \
    ../../rtl/wavebnn_core.v \
    ../../rtl/haar_wavelet_3lvl.v \
    ../../rtl/bnn_branch.v \
    ../../rtl/popcount.v \
    ../../rtl/bin_fc1.v \
    ../../rtl/fc_output.v

echo "=== Running Simulation ==="
vvp tb_core_sv

echo "=== Done ==="
