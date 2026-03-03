# ============================================================
# Vivado TCL Script — Create HDC AMC Project
# ============================================================
# Usage (from Vivado Tcl Console or command line):
#   cd <path_to_FPGA_Hack>/hardware/vivado
#   source ../vivado/create_project.tcl
#
# Or from command line:
#   vivado -mode batch -source create_project.tcl
#
# This script:
#   1. Creates a new Vivado project
#   2. Adds all RTL source files
#   3. Adds the constraints file
#   4. Adds the testbench for simulation
#   5. Sets the FPGA target part
#   6. Configures simulation settings
# ============================================================

# Project settings
set project_name "hdc_amc"
set project_dir  "."
set part         "xc7a100tcsg324-1"
set board        "digilentinc.com:nexys-a7-100t:part0:1.3"

# Resolve paths relative to this script
set script_dir [file dirname [info script]]
set hw_dir     [file normalize "${script_dir}/.."]
set rtl_dir    "${hw_dir}/rtl"
set tb_dir     "${hw_dir}/tb"
set constr_dir "${hw_dir}/constraints"
set sw_dir     [file normalize "${hw_dir}/../software"]
set export_dir "${hw_dir}/tb/test_vectors"

# ============================================================
# Create project
# ============================================================
create_project ${project_name} ${project_dir}/${project_name} -part ${part} -force

# Try to set board part (may fail if board files not installed)
catch { set_property board_part ${board} [current_project] }

# ============================================================
# Add RTL sources
# ============================================================
set rtl_files [list \
    "${rtl_dir}/hdc_params.vh" \
    "${rtl_dir}/popcount.v" \
    "${rtl_dir}/level_quantizer.v" \
    "${rtl_dir}/codebook_rom.v" \
    "${rtl_dir}/sample_encoder.v" \
    "${rtl_dir}/window_bundler.v" \
    "${rtl_dir}/hamming_distance.v" \
    "${rtl_dir}/classifier.v" \
    "${rtl_dir}/hdc_core.v" \
    "${rtl_dir}/uart_rx.v" \
    "${rtl_dir}/uart_tx.v" \
    "${rtl_dir}/system_top.v" \
]

foreach f $rtl_files {
    if {[file exists $f]} {
        add_files -norecurse $f
    } else {
        puts "WARNING: RTL file not found: $f"
    }
}

# Set hdc_params.vh as a header file (included, not compiled directly)
set_property file_type {Verilog Header} [get_files -of_objects [get_filesets sources_1] *hdc_params.vh]

# ============================================================
# Add constraints
# ============================================================
set xdc_file "${constr_dir}/nexys_a7_100t.xdc"
if {[file exists $xdc_file]} {
    add_files -fileset constrs_1 -norecurse $xdc_file
} else {
    puts "WARNING: Constraints file not found: $xdc_file"
}

# ============================================================
# Add testbench (simulation only)
# ============================================================
set tb_file "${tb_dir}/tb_hdc_core.v"
if {[file exists $tb_file]} {
    add_files -fileset sim_1 -norecurse $tb_file
    set_property top tb_hdc_core [get_filesets sim_1]
} else {
    puts "WARNING: Testbench file not found: $tb_file"
}

# ============================================================
# Add hex files for BRAM initialization (if exported)
# ============================================================
set hex_files [glob -nocomplain "${export_dir}/*.hex"]
foreach h $hex_files {
    add_files -norecurse $h
    set_property file_type {Memory File} [get_files [file tail $h]]
}

# Also copy hex files to simulation directory so $readmemh works
set sim_dir "${project_dir}/${project_name}.sim/sim_1/behav/xsim"
file mkdir $sim_dir
foreach h $hex_files {
    file copy -force $h $sim_dir
}

# ============================================================
# Set top module
# ============================================================
set_property top system_top [current_fileset]

# ============================================================
# Synthesis settings
# ============================================================
set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.KEEP_EQUIVALENT_REGISTERS true [get_runs synth_1]

# ============================================================
# Implementation settings
# ============================================================
set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]

# ============================================================
# Simulation settings
# ============================================================
set_property -name {xsim.simulate.runtime} -value {1000us} -objects [get_filesets sim_1]
set_property -name {xsim.simulate.log_all_signals} -value {true} -objects [get_filesets sim_1]

# ============================================================
# Include directories (for `include "hdc_params.vh")
# ============================================================
set_property verilog_define {} [current_fileset]
set_property include_dirs $rtl_dir [current_fileset]
set_property include_dirs $rtl_dir [get_filesets sim_1]

# ============================================================
# Done
# ============================================================
puts ""
puts "============================================================"
puts "  Project '${project_name}' created successfully!"
puts "  Part: ${part}"
puts "  Top module: system_top"
puts "  Testbench: tb_hdc_core"
puts "============================================================"
puts ""
puts "Next steps:"
puts "  1. Run 'python software/main.py --export' to generate hex files"
puts "  2. Run Behavioral Simulation to verify"
puts "  3. Run Synthesis"
puts "  4. Run Implementation"
puts "  5. Generate Bitstream"
puts "  6. Program device"
puts ""
