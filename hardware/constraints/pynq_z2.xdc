## =============================================================================
## WaveBNN-ECG: PYNQ-Z2 Pin Constraints
## =============================================================================
## Board: Digilent PYNQ-Z2 (Zynq-7020, xc7z020clg484-1)
## Reference: PYNQ-Z2 schematic + master XDC
## =============================================================================

## ─── Clock: 125 MHz from Ethernet PHY (directly on PL side) ───
set_property -dict { PACKAGE_PIN H16   IOSTANDARD LVCMOS33 } [get_ports { clk_125 }]
create_clock -add -name sys_clk_pin -period 8.000 -waveform {0 4} [get_ports { clk_125 }]

## ─── Reset: BTN0 (active-low) ───
set_property -dict { PACKAGE_PIN D19   IOSTANDARD LVCMOS33 } [get_ports { rst_n_btn }]

## ─── UART: via USB-UART bridge (directly on PL pins) ───
## PYNQ-Z2: PMODB upper row for UART if USB-serial not available
## Using PMODA pins 0,1 (directly accessible on PL)
## Pin 1 (PMODA[0]) = TX (FPGA→PC), Pin 2 (PMODA[1]) = RX (PC→FPGA)
set_property -dict { PACKAGE_PIN Y18   IOSTANDARD LVCMOS33 } [get_ports { uart_txd }]
set_property -dict { PACKAGE_PIN Y19   IOSTANDARD LVCMOS33 } [get_ports { uart_rxd }]

## ─── LEDs ───
set_property -dict { PACKAGE_PIN R14   IOSTANDARD LVCMOS33 } [get_ports { led[0] }]
set_property -dict { PACKAGE_PIN P14   IOSTANDARD LVCMOS33 } [get_ports { led[1] }]
set_property -dict { PACKAGE_PIN N16   IOSTANDARD LVCMOS33 } [get_ports { led[2] }]
set_property -dict { PACKAGE_PIN M14   IOSTANDARD LVCMOS33 } [get_ports { led[3] }]

## ─── Timing Constraints ───
## MMCM generates 100 MHz → auto-derived by Vivado
## Just constrain the input clock and let tools propagate

## False path on async reset
set_false_path -from [get_ports { rst_n_btn }]

## UART is slow (115200 baud) — no tight timing needed
set_false_path -from [get_ports { uart_rxd }]
set_false_path -to   [get_ports { uart_txd }]
set_false_path -to   [get_ports { led[*] }]

## ─── Configuration ───
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]

## ─── Bitstream settings ───
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
set_property BITSTREAM.CONFIG.UNUSEDPIN PULLUP [current_design]
