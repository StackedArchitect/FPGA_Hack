"""WaveBNN-ECG configuration: hyperparameters and paths."""

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
HARDWARE_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "hardware")
EXPORT_DIR   = os.path.join(HARDWARE_DIR, "tb", "test_vectors")

for _d in [DATA_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(_d, exist_ok=True)

# Dataset — MIT-BIH Arrhythmia Database (PhysioNet)
# Inter-patient split following AAMI recommendation (DS1 / DS2)
MITBIH_RECORDS_TRAIN = [
    101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
    122, 124, 201, 203, 205, 207, 208, 209, 215, 220,
    223, 230,
]
MITBIH_RECORDS_TEST = [
    100, 103, 105, 111, 113, 117, 121, 123,
    200, 202, 210, 212, 213, 214, 219, 221, 222, 228,
    231, 232, 233, 234,
]

# Five AAMI super-classes
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
NUM_CLASSES  = len(AAMI_CLASSES)

# Mapping from PhysioNet annotation symbols → AAMI class index
ANNOT_TO_AAMI = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,   # Normal
    "A": 1, "a": 1, "J": 1, "S": 1,             # Supraventricular
    "V": 2, "E": 2,                               # Ventricular
    "F": 3,                                        # Fusion
    "/": 4, "f": 4, "Q": 4,                       # Unknown / Paced
}

# Preprocessing
SAMPLING_RATE = 360           # MIT-BIH sample rate (Hz)
BEAT_WINDOW   = 187           # Samples per beat (R-peak centred)
BEAT_BEFORE   = 72            # Samples before R-peak
BEAT_AFTER    = 114           # Samples after R-peak  (72 + 114 + 1 = 187)
INPUT_BITS    = 8             # Quantisation bits for FPGA input

# Haar Wavelet Decomposition
WAVELET_LEVELS = 3            # 3-level DWT
# After 3-level Haar on 187 samples:
#   Level 1 → cA1 (94), cD1 (94)        ← padding from odd length handled
#   Level 2 → cA2 (47), cD2 (47)
#   Level 3 → cA3 (24), cD3 (24)
# We keep: cA3 (24), cD3 (24), cD2 (47), cD1 (94)  → total 189 features
SUBBAND_LENGTHS = {
    "cA3": 24,
    "cD3": 24,
    "cD2": 47,
    "cD1": 94,
}

# BNN Model Architecture (4-branch parallel)
# Each branch: one BinaryConv1d → MaxPool(2) → Flatten
# Branch configs: (input_len, out_channels, kernel_size)
BRANCH_CONFIGS = {
    "cA3": {"in_len": 24,  "out_ch": 32, "kernel": 5, "pool": 2},
    "cD3": {"in_len": 24,  "out_ch": 32, "kernel": 5, "pool": 2},
    "cD2": {"in_len": 47,  "out_ch": 32, "kernel": 5, "pool": 2},
    "cD1": {"in_len": 94,  "out_ch": 16, "kernel": 3, "pool": 2},
}

# After each branch: floor((in_len - kernel + 1) / pool) * out_ch
#   cA3 branch output: floor((24-5+1)/2) = 10 positions, 32 channels → 320 bits
#   cD3 branch output: floor((24-5+1)/2) = 10 positions, 32 channels → 320 bits
#   cD2 branch output: floor((47-5+1)/2) = 21 positions, 32 channels → 672 bits
#   cD1 branch output: floor((94-3+1)/2) = 46 positions, 16 channels → 736 bits
# Concatenated: 320 + 320 + 672 + 736 = 2048 bits → power of 2!
CONCAT_BITS = 2048

# Fully-connected layers
FC1_OUT = 128         # BinaryLinear(2048 → 128)
FC2_OUT = NUM_CLASSES  # Linear(128 → 5), full-precision output

# Training
BATCH_SIZE    = 256
NUM_EPOCHS    = 150
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
RANDOM_SEED   = 42
USE_CLASS_WEIGHTS = True       # handle N >> V >> S > F > Q imbalance

# FPGA Hardware
FPGA_BOARD      = "PYNQ-Z2"
FPGA_PART       = "xc7z020clg484-1"
FPGA_CLK_FREQ   = 100_000_000   # 100 MHz
UART_BAUD_RATE  = 115_200
