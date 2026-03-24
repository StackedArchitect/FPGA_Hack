# ==============================================================================
# WaveBNN-ECG: Vivado 2024.2 Project Creation Script
# ==============================================================================
#
# Usage:
#   cd hardware/vivado
#   vivado -mode batch -source create_project.tcl
#
# Or from Vivado GUI: Tools → Run Tcl Script → select this file
# ==============================================================================

# ─── Project settings ───
set project_name  "wavebnn_ecg"
set part          "xc7z020clg484-1"
set board_part    "tul.com.tw:pynq-z2:part0:1.0"

# ─── Paths (relative to this script's location) ───
set script_dir    [file dirname [info script]]
set rtl_dir       [file normalize "$script_dir/../rtl"]
set tb_dir        [file normalize "$script_dir/../tb"]
set constr_dir    [file normalize "$script_dir/../constraints"]
set proj_dir      [file normalize "$script_dir/$project_name"]

# ─── Create project ───
if {[file exists $proj_dir]} {
    puts "INFO: Removing existing project directory: $proj_dir"
    file delete -force $proj_dir
}

create_project $project_name $proj_dir -part $part -force

# Try to set board part (may fail if board files not installed)
catch {set_property board_part $board_part [current_project]}

set_property target_language Verilog [current_project]
set_property simulator_language Verilog [current_project]

# ─── Add RTL sources ───
set rtl_files [list \
    "$rtl_dir/system_top.v" \
    "$rtl_dir/wavebnn_core.v" \
    "$rtl_dir/haar_wavelet_3lvl.v" \
    "$rtl_dir/bnn_branch.v" \
    "$rtl_dir/popcount.v" \
    "$rtl_dir/bin_fc1.v" \
    "$rtl_dir/fc_output.v" \
    "$rtl_dir/uart_rx.v" \
    "$rtl_dir/uart_tx.v" \
]

foreach f $rtl_files {
    if {[file exists $f]} {
        add_files -norecurse $f
        puts "INFO: Added source: $f"
    } else {
        puts "WARNING: Source not found: $f"
    }
}

set_property top system_top [current_fileset]

# ─── Add constraints ───
set xdc_file "$constr_dir/pynq_z2.xdc"
if {[file exists $xdc_file]} {
    add_files -fileset constrs_1 -norecurse $xdc_file
    puts "INFO: Added constraints: $xdc_file"
} else {
    puts "WARNING: Constraints not found: $xdc_file"
}

# ─── Add testbench ───
set tb_files [list \
    "$tb_dir/tb_wavebnn_core.v" \
    "$tb_dir/tb_haar_wavelet_3lvl.v" \
]

foreach f $tb_files {
    if {[file exists $f]} {
        add_files -fileset sim_1 -norecurse $f
        puts "INFO: Added testbench: $f"
    } else {
        puts "WARNING: Testbench not found: $f"
    }
}

set_property top tb_wavebnn_core [get_filesets sim_1]

# ─── Add .mem files (weight/threshold data) ───
set mem_dir "$tb_dir/test_vectors"
if {[file isdirectory $mem_dir]} {
    set mem_files [glob -nocomplain "$mem_dir/*.mem"]
    foreach f $mem_files {
        add_files -norecurse $f
        puts "INFO: Added mem file: $f"
    }
}

# ─── Synthesis settings ───
set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY rebuilt [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING on [get_runs synth_1]

# ─── Implementation settings ───
set_property strategy Performance_ExtraTimingOpt [get_runs impl_1]

# ─── Summary ───
puts ""
puts "================================================================"
puts "  Project created: $proj_dir"
puts "  Part:            $part"
puts "  Top module:      system_top"
puts "================================================================"
puts ""
puts "Next steps:"
puts "  1. Open project: open_project $proj_dir/$project_name.xpr"
puts "  2. Run synthesis: launch_runs synth_1 -jobs 4"
puts "  3. Run implementation: launch_runs impl_1 -to_step write_bitstream -jobs 4"
puts ""

close_project
