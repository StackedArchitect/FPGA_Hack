## =============================================================================
## WaveBNN-ECG: ZC702 Evaluation Board Pin Constraints
## =============================================================================
## Board: Xilinx ZC702 (Zynq-7020, xc7z020clg484-1)
## Reference: UG850 ZC702 Evaluation Board User Guide
## =============================================================================

## ─── Clock: 200 MHz LVDS differential (U43 oscillator, Bank 35 MRCC) ───
set_property -dict { PACKAGE_PIN D18   IOSTANDARD LVDS_25 } [get_ports { sys_clk_p }]
set_property -dict { PACKAGE_PIN C19   IOSTANDARD LVDS_25 } [get_ports { sys_clk_n }]
create_clock -add -name sys_clk_pin -period 5.000 -waveform {0 2.5} [get_ports { sys_clk_p }]

## ─── Reset: GPIO_SW_N pushbutton (active-low) ───
set_property -dict { PACKAGE_PIN G19   IOSTANDARD LVCMOS25 } [get_ports { rst_n_btn }]

## ─── UART: PMOD1 (J62) — via TXS0108E level shifter (2.5V→3.3V) ───
## PMOD1_0 = FPGA TX (to PC), PMOD1_1 = FPGA RX (from PC)
set_property -dict { PACKAGE_PIN E15   IOSTANDARD LVCMOS25 } [get_ports { uart_txd }]
set_property -dict { PACKAGE_PIN D15   IOSTANDARD LVCMOS25 } [get_ports { uart_rxd }]

## ─── LEDs: DS15–DS18 (dedicated user LEDs, Bank 33/34) ───
set_property -dict { PACKAGE_PIN P17   IOSTANDARD LVCMOS25 } [get_ports { led[0] }]
set_property -dict { PACKAGE_PIN P18   IOSTANDARD LVCMOS25 } [get_ports { led[1] }]
set_property -dict { PACKAGE_PIN W10   IOSTANDARD LVCMOS25 } [get_ports { led[2] }]
set_property -dict { PACKAGE_PIN V7    IOSTANDARD LVCMOS25 } [get_ports { led[3] }]

## ─── Timing Constraints ───
## MMCM generates 100 MHz from 200 MHz input — auto-derived by Vivado

## False path on async reset
set_false_path -from [get_ports { rst_n_btn }]

## UART is slow (115200 baud) — no tight timing needed
set_false_path -from [get_ports { uart_rxd }]
set_false_path -to   [get_ports { uart_txd }]
set_false_path -to   [get_ports { led[*] }]

## ─── Configuration ───
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 2.5 [current_design]

## ─── Bitstream settings ───
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
set_property BITSTREAM.CONFIG.UNUSEDPIN PULLUP [current_design]
