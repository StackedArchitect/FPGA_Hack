"""
MIT-BIH Arrhythmia Database Loader
===================================
Supports two data sources:
  1. wfdb (downloads from PhysioNet) — full pipeline with R-peak segmentation
  2. Kaggle pre-processed CSV — already segmented, ready to use

Both produce the same output format:
  X: np.ndarray (N, 187) float32 — normalised beat waveforms
  y: np.ndarray (N,)     int64   — AAMI class labels 0–4
"""

import os
import numpy as np

from .config import (
    DATA_DIR, BEAT_WINDOW, BEAT_BEFORE, BEAT_AFTER,
    MITBIH_RECORDS_TRAIN, MITBIH_RECORDS_TEST,
    ANNOT_TO_AAMI, NUM_CLASSES, AAMI_CLASSES, INPUT_BITS,
)


# ────────────────────────────────────────────────────────────
# PhysioNet / wfdb Loader
# ────────────────────────────────────────────────────────────

def download_mitbih(data_dir: str = DATA_DIR) -> None:
    """Download MIT-BIH Arrhythmia Database from PhysioNet using wfdb."""
    import wfdb
    os.makedirs(data_dir, exist_ok=True)
    db_dir = os.path.join(data_dir, "mitdb")
    if os.path.exists(db_dir) and len(os.listdir(db_dir)) > 10:
        print(f"[Dataset] MIT-BIH already downloaded at {db_dir}")
        return
    os.makedirs(db_dir, exist_ok=True)
    print("[Dataset] Downloading MIT-BIH from PhysioNet (≈70 MB)...")
    wfdb.dl_database("mitdb", dl_dir=db_dir)
    print(f"[Dataset] Downloaded to {db_dir}")


def _segment_record(record_path: str) -> tuple:
    """
    Segment one MIT-BIH record into beats.

    Returns:
        beats: list of np.ndarray (each length BEAT_WINDOW)
        labels: list of int (AAMI class index)
    """
    import wfdb

    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, "atr")

    # Use lead 0 (MLII)
    signal = record.p_signal[:, 0].astype(np.float64)

    beats, labels = [], []

    for i, (pos, symbol) in enumerate(zip(ann.sample, ann.symbol)):
        # Skip non-beat annotations
        if symbol not in ANNOT_TO_AAMI:
            continue

        label = ANNOT_TO_AAMI[symbol]

        # Extract window centred on R-peak
        start = pos - BEAT_BEFORE
        end = pos + BEAT_AFTER + 1  # +1 because slice is exclusive

        if start < 0 or end > len(signal):
            continue

        beat = signal[start:end]
        assert len(beat) == BEAT_WINDOW, f"Beat length {len(beat)} != {BEAT_WINDOW}"

        # Per-beat z-normalisation (zero mean, unit variance)
        mu = beat.mean()
        std = beat.std()
        if std < 1e-6:
            continue  # skip flat beats
        beat = (beat - mu) / std

        beats.append(beat.astype(np.float32))
        labels.append(label)

    return beats, labels


def load_mitbih(data_dir: str = DATA_DIR) -> tuple:
    """
    Load MIT-BIH using inter-patient split (AAMI recommendation).

    Returns:
        X_train, y_train, X_test, y_test
    """
    db_dir = os.path.join(data_dir, "mitdb")
    if not os.path.exists(db_dir):
        download_mitbih(data_dir)

    def _load_records(records):
        all_beats, all_labels = [], []
        for rec_num in records:
            rec_path = os.path.join(db_dir, str(rec_num))
            try:
                beats, labels = _segment_record(rec_path)
                all_beats.extend(beats)
                all_labels.extend(labels)
            except Exception as e:
                print(f"  [Warning] Skipping record {rec_num}: {e}")
        return np.array(all_beats, dtype=np.float32), np.array(all_labels, dtype=np.int64)

    print("[Dataset] Loading training records...")
    X_train, y_train = _load_records(MITBIH_RECORDS_TRAIN)
    print(f"  → {len(X_train)} beats")

    print("[Dataset] Loading test records...")
    X_test, y_test = _load_records(MITBIH_RECORDS_TEST)
    print(f"  → {len(X_test)} beats")

    return X_train, y_train, X_test, y_test


# ────────────────────────────────────────────────────────────
# Kaggle Pre-processed CSV Loader (simpler, faster)
# ────────────────────────────────────────────────────────────

def load_kaggle_mitbih(data_dir: str = DATA_DIR) -> tuple:
    """
    Load the Kaggle pre-processed MIT-BIH CSV files.

    Expected files in data_dir:
        mitbih_train.csv  (87554 × 188)
        mitbih_test.csv   (21892 × 188)

    Each row: 187 signal values + 1 label (0–4).
    Download from: https://www.kaggle.com/datasets/shayanfazeli/heartbeat
    """
    train_path = os.path.join(data_dir, "mitbih_train.csv")
    test_path = os.path.join(data_dir, "mitbih_test.csv")

    for p in [train_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Kaggle CSV not found: {p}\n"
                f"Download from: https://www.kaggle.com/datasets/shayanfazeli/heartbeat\n"
                f"Place mitbih_train.csv and mitbih_test.csv in {data_dir}/"
            )

    print("[Dataset] Loading Kaggle MIT-BIH CSV files...")

    train_data = np.loadtxt(train_path, delimiter=",", dtype=np.float32)
    test_data = np.loadtxt(test_path, delimiter=",", dtype=np.float32)

    X_train = train_data[:, :-1]  # (N, 187)
    y_train = train_data[:, -1].astype(np.int64)

    X_test = test_data[:, :-1]
    y_test = test_data[:, -1].astype(np.int64)

    print(f"  Train: {len(X_train)} beats, Test: {len(X_test)} beats")
    return X_train, y_train, X_test, y_test


# ────────────────────────────────────────────────────────────
# Quantisation for FPGA
# ────────────────────────────────────────────────────────────

def quantize_for_fpga(X: np.ndarray, bits: int = INPUT_BITS) -> np.ndarray:
    """
    Quantise float32 beats to signed integer for FPGA input.

    Maps the range [-3σ, +3σ] (after z-norm, most values in [-3, 3])
    to the signed integer range [-(2^(bits-1)), 2^(bits-1) - 1].

    Args:
        X: (N, 187) float32 array (z-normalised)
        bits: number of bits (default 8)

    Returns:
        X_q: (N, 187) int8 array (for 8-bit)
    """
    max_val = 2 ** (bits - 1) - 1  # 127 for 8-bit
    scale = max_val / 3.0          # map ±3 to ±127

    X_q = np.clip(np.round(X * scale), -max_val - 1, max_val).astype(np.int8)
    return X_q
