# HDC-AMC: Hyperdimensional Computing for Automatic Modulation Classification

An FPGA-accelerated modulation classifier using Hyperdimensional Computing (HDC).  
**Zero DSP blocks** — inference uses only XOR, popcount, and Hamming distance on binary hypervectors.

## Architecture Overview

```
I/Q Samples ──► Preprocess ──► Quantize ──► Codebook ──► XOR Bind ──► Bundle ──► Hamming Match ──► Class
               (amp, Δφ)      (Q=16 lvl)   (BRAM ROM)   (LUT)       (counter)   (vs prototypes)
```

- **Dimension**: D = 4096 bits (processed as 128 × 32-bit chunks)
- **Encoding**: Amplitude + phase-difference features, level-quantized, XOR-bound from binary codebooks
- **Classification**: Hamming distance to learned prototype vectors (majority-vote training)
- **Modulations**: 11 classes (BPSK, QPSK, 8PSK, 16QAM, 64QAM, GFSK, AM-DSB, OOK, CPFSK, FM, PAM4)
- **Latency**: ~540 µs per 128-sample window at 100 MHz

## Target Hardware

| Resource | Used (est.) | Available | Utilization |
|----------|-------------|-----------|-------------|
| **Board** | Nexys A7-100T | XC7A100TCSG324-1 | — |
| LUTs | ~3,000 | 63,400 | ~5% |
| FFs | ~2,000 | 126,800 | ~2% |
| BRAM | ~20 | 135 | ~15% |
| DSP | **0** | 240 | **0%** |

## Project Structure

```
FPGA_Hack/
├── software/
│   ├── src/
│   │   ├── config.py           # All hyperparameters and paths
│   │   ├── hdc_encoder.py      # HDC encoding pipeline
│   │   ├── hdc_classifier.py   # Training and inference
│   │   ├── dataset_loader.py   # RadioML + synthetic data
│   │   ├── evaluate.py         # Metrics and plots
│   │   └── export_to_fpga.py   # Export model to Verilog hex/coe
│   ├── main.py                 # CLI entry point
│   ├── requirements.txt
│   ├── data/                   # Dataset files (download separately)
│   ├── results/                # Plots and reports
│   └── fpga_export/            # Generated hex files for hardware
│
├── hardware/
│   ├── rtl/
│   │   ├── hdc_params.vh       # Default Verilog parameters
│   │   ├── popcount.v          # Hierarchical popcount tree
│   │   ├── level_quantizer.v   # Bit-truncation quantizer
│   │   ├── codebook_rom.v      # BRAM ROM ($readmemh)
│   │   ├── sample_encoder.v    # Per-sample encoding FSM
│   │   ├── window_bundler.v    # Counter-based majority bundler
│   │   ├── hamming_distance.v  # XOR + popcount comparator
│   │   ├── classifier.v        # Sequential argmin classifier
│   │   ├── hdc_core.v          # Top-level HDC pipeline
│   │   ├── uart_rx.v           # UART receiver (8N1)
│   │   ├── uart_tx.v           # UART transmitter (8N1)
│   │   └── system_top.v        # Board-level Nexys A7 wrapper
│   ├── tb/
│   │   └── tb_hdc_core.v       # Testbench (loads test vectors)
│   ├── constraints/
│   │   └── nexys_a7_100t.xdc   # Pin assignments
│   └── vivado/
│       └── create_project.tcl  # Automated project creation
│
└── docs/
    ├── HACKATHON_STRATEGY.md
    └── PROJECT_PROPOSAL.md
```

## Quick Start

### 1. Software — Train Model & Export

```bash
# Create virtual environment
cd FPGA_Hack
python3 -m venv .venv
source .venv/bin/activate
pip install -r software/requirements.txt

# Train and export (synthetic data)
python software/main.py --dataset synthetic --export

# Train with real RadioML data (if downloaded)
python software/main.py --dataset radioml2016 --export --retrain 3
```

This generates the hex files in `software/fpga_export/`:
- `codebook_i.hex` / `codebook_q.hex` — Codebook ROMs
- `prototypes.hex` — Trained class prototypes
- `hdc_params.vh` — Matching Verilog parameters
- `test_input.hex` / `test_expected.hex` — Test vectors for simulation

### 2. Hardware — Vivado Project

```tcl
# In Vivado Tcl Console:
cd <path_to_FPGA_Hack>/hardware/vivado
source create_project.tcl
```

Or open Vivado GUI and run `Tools → Run Tcl Script...`

### 3. Simulate

1. In Vivado, go to **Flow Navigator → Simulation → Run Behavioral Simulation**
2. The testbench (`tb_hdc_core`) loads test vectors and reports Pass/Fail per window
3. Check the Tcl Console for classification results

### 4. Synthesize & Implement

1. **Run Synthesis** — should complete with no errors and 0 DSP blocks used
2. **Run Implementation** — check timing report (should meet 100 MHz easily)
3. **Generate Bitstream**
4. **Program Device** via USB

### 5. Live Demo

The FPGA communicates via UART (115200 baud) over the USB cable.

**Protocol** (host → FPGA):
| Byte | Meaning |
|------|---------|
| `0x01` | New window (reset) |
| `0x02` `AMP` `PDIFF` | Load one sample (amplitude + phase-diff, 8-bit each) |
| `0x03` | Classify (returns 1-byte class ID) |

**LED Display**:
- LED[3:0] — Classification result (binary class ID)
- LED[7:4] — State indicator
- LED[15:8] — Sample counter
- 7-segment — Class number (hex)

## Dataset

**RadioML 2016.10a** (preferred):
- Download from [DeepSig](https://www.deepsig.ai/datasets) or search for mirrors
- Place `RML2016.10a_dict.pkl` in `software/data/`
- 11 modulation classes, 128 I/Q samples per example, SNR range -20 to +18 dB

**Synthetic fallback**: The software generates realistic synthetic modulations for pipeline testing.

## Key Design Decisions

1. **Binary HDC** — All vectors are binary (0/1), enabling XOR for binding and popcount for similarity. No floating-point, no multipliers.
2. **Amplitude + Phase-difference** — Rotation-invariant features instead of raw I/Q. Critical for real-world signals with unknown phase/frequency offsets.
3. **Chunk-based processing** — 4096-bit vectors processed as 128 × 32-bit chunks. Keeps datapath narrow while handling large dimensions.
4. **BRAM codebooks** — Codebook and prototype storage in Block RAM, initialized via `$readmemh()` from Python-exported hex files.
5. **Zero DSP** — All operations are bitwise (XOR, AND, comparison). The design uses zero DSP48 slices.

## Software CLI Options

```
python software/main.py [OPTIONS]

  --dataset {synthetic,radioml2016,radioml2018}   Dataset to use
  --D {512,1024,2048,4096,8192}                   Hypervector dimension
  --Q {4,8,16,32}                                 Quantization levels
  --ngram {1,2,3,4,5}                             N-gram size
  --sweep                                          Run parameter sweep
  --export                                         Export model for FPGA
  --retrain N                                      Retrain iterations
  --snr-filter MIN                                 Keep only SNR >= MIN
```

## License

See [LICENSE](LICENSE).
