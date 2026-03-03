"""
HDC-AMC Configuration
Hyperparameters and paths for Hyperdimensional Computing
Automatic Modulation Classification on FPGA.

Target Board: Nexys A7-100T (XC7A100TCSG324-1)
"""

import os

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
HARDWARE_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "hardware")
EXPORT_DIR = os.path.join(HARDWARE_DIR, "tb", "test_vectors")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)


# ============================================================
# HDC Hyperparameters
# ============================================================
D = 4096              # Hypervector dimension (bits). Sweep: [1024, 2048, 4096, 8192, 10000]
Q = 16                # Number of quantization levels for I and Q channels (4-bit)
N_GRAM = 3            # N-gram length for temporal encoding (1 = no n-gram)
WINDOW_SIZE = 128     # Number of I/Q samples per classification window
                      # RadioML 2016.10a has 128 samples/example
                      # RadioML 2018.01A has 1024 samples/example (we window it)
ENCODE_MODE = 'amp_phase'  # 'iq' (raw I/Q) or 'amp_phase' (amplitude + phase-diff)
                           # amp_phase is strongly recommended for modulation classification
                           # because real RF signals have random carrier phase/freq offsets

# ============================================================
# Dataset Configuration
# ============================================================
DATASET_VERSION = "2016.10a"   # "2016.10a" or "2018.01A"

# RadioML 2016.10a
RADIOML_2016_URL = "https://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2"
RADIOML_2016_FILE = os.path.join(DATA_DIR, "RML2016.10a_dict.pkl")

# RadioML 2018.01A
RADIOML_2018_FILE = os.path.join(DATA_DIR, "GOLD_XYZ_OSC.0001_1024.hdf5")

# Synthetic dataset (for testing without real data)
USE_SYNTHETIC = False   # Set True to use synthetic data for code testing

# ============================================================
# RadioML 2016.10a Modulation Classes (11 classes)
# ============================================================
MODULATIONS_2016 = [
    '8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK',
    'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM'
]

# RadioML 2018.01A Modulation Classes (24 classes)
MODULATIONS_2018 = [
    'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK',
    '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK',
    '16QAM', '32QAM', '64QAM', '128QAM', '256QAM',
    'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
    'FM', 'GMSK', 'OQPSK'
]

# Number of classes based on dataset version
NUM_CLASSES = len(MODULATIONS_2016) if DATASET_VERSION == "2016.10a" else len(MODULATIONS_2018)

# ============================================================
# Training Configuration
# ============================================================
TRAIN_SPLIT = 0.7      # Fraction of data for training
RANDOM_SEED = 42        # For reproducibility
SNR_FILTER = None       # None = use all SNRs, or list like [0, 2, 4, 6, 8, 10]
                        # For quick iteration, try SNR_FILTER = [10, 18] (high SNR only)

# ============================================================
# FPGA Hardware Parameters (must match Verilog parameters)
# ============================================================
FPGA_D = 4096           # Dimension for FPGA (can differ from software sweep)
FPGA_Q = 16             # Quantization levels for FPGA
FPGA_N_GRAM = 1         # N-gram for FPGA v1 (start simple, no rotation needed)
FPGA_NUM_CLASSES = NUM_CLASSES
FPGA_INPUT_WIDTH = 8    # 8-bit ADC input for I and Q
FPGA_CHUNK_WIDTH = 64   # Bits processed per clock cycle in Verilog

# ============================================================
# FPGA Target Board
# ============================================================
FPGA_BOARD = "Nexys A7-100T"
FPGA_PART = "xc7a100tcsg324-1"
FPGA_CLK_FREQ = 100_000_000   # 100 MHz system clock
UART_BAUD_RATE = 115200

# ============================================================
# Evaluation
# ============================================================
SNRS_TO_PLOT = list(range(-20, 20, 2))   # SNR values to include in accuracy vs SNR plot
D_SWEEP = [256, 512, 1024, 2048, 4096, 8192]   # Dimensions for Pareto analysis
Q_SWEEP = [4, 8, 16, 32]                        # Quantization levels to sweep
NGRAM_SWEEP = [1, 2, 3, 5]                      # N-gram lengths to sweep
