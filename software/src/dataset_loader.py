"""
Dataset Loader for RadioML and Synthetic Data
Supports RadioML 2016.10a, 2018.01A, and synthetic I/Q data generation.

RadioML 2016.10a: Pickle file, 11 modulations, 128 I/Q samples per example
RadioML 2018.01A: HDF5 file, 24 modulations, 1024 I/Q samples per example
Synthetic: Generated I/Q with basic modulation characteristics for code testing
"""

import os
import pickle
import numpy as np
from typing import Tuple, Optional, List, Dict


def load_radioml_2016(filepath: str,
                      snr_filter: Optional[List[int]] = None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load RadioML 2016.10a dataset.
    
    Format: Pickle dictionary keyed by (modulation_str, snr_int)
            Values are numpy arrays of shape (num_examples, 2, 128)
    
    Args:
        filepath: Path to RML2016.10a_dict.pkl
        snr_filter: Optional list of SNR values to include. None = all SNRs.
        
    Returns:
        X: np.ndarray of shape (N, 2, 128) — I/Q data
        y: np.ndarray of shape (N,) — integer class labels
        snrs: np.ndarray of shape (N,) — SNR value per example
        mod_names: List of modulation class names (index = label)
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # Extract all modulation types and SNRs
    all_mods = sorted(list(set(k[0] for k in data.keys())))
    all_snrs = sorted(list(set(k[1] for k in data.keys())))

    if snr_filter is not None:
        all_snrs = [s for s in all_snrs if s in snr_filter]

    X_list, y_list, snr_list = [], [], []

    for mod_idx, mod in enumerate(all_mods):
        for snr in all_snrs:
            key = (mod, snr)
            if key in data:
                samples = data[key]  # shape: (num_examples, 2, 128)
                X_list.append(samples)
                y_list.append(np.full(samples.shape[0], mod_idx, dtype=np.int32))
                snr_list.append(np.full(samples.shape[0], snr, dtype=np.int32))

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0)
    snrs = np.concatenate(snr_list, axis=0)

    print(f"[Dataset] RadioML 2016.10a loaded: {X.shape[0]} examples, "
          f"{len(all_mods)} classes, {len(all_snrs)} SNR levels")
    print(f"[Dataset] Modulations: {all_mods}")
    print(f"[Dataset] SNR range: {all_snrs[0]} to {all_snrs[-1]} dB")

    return X, y, snrs, all_mods


def load_radioml_2018(filepath: str,
                      snr_filter: Optional[List[int]] = None,
                      window_size: int = 128
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load RadioML 2018.01A dataset.
    
    Format: HDF5 with datasets 'X' (N, 1024, 2), 'Y' (N, 24) one-hot, 'Z' (N,) SNR
    We take the first 'window_size' samples from each 1024-sample example.
    
    Args:
        filepath: Path to GOLD_XYZ_OSC.0001_1024.hdf5
        snr_filter: Optional list of SNR values to include
        window_size: Number of samples to take from each example (default 128)
        
    Returns:
        X: np.ndarray of shape (N, 2, window_size)
        y: np.ndarray of shape (N,)
        snrs: np.ndarray of shape (N,)
        mod_names: List of modulation class names
    """
    import h5py

    mod_names = [
        'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK',
        '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK',
        '16QAM', '32QAM', '64QAM', '128QAM', '256QAM',
        'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
        'FM', 'GMSK', 'OQPSK'
    ]

    with h5py.File(filepath, 'r') as f:
        # Read in chunks to save memory if needed
        X_raw = f['X'][:, :window_size, :]   # (N, window_size, 2)
        Y_onehot = f['Y'][:]                 # (N, 24)
        snrs_raw = f['Z'][:]                 # (N,)

    y = np.argmax(Y_onehot, axis=1).astype(np.int32)
    snrs = snrs_raw.astype(np.int32)

    # Reshape from (N, window_size, 2) to (N, 2, window_size)
    X = np.transpose(X_raw, (0, 2, 1)).astype(np.float32)

    # Apply SNR filter
    if snr_filter is not None:
        mask = np.isin(snrs, snr_filter)
        X, y, snrs = X[mask], y[mask], snrs[mask]

    print(f"[Dataset] RadioML 2018.01A loaded: {X.shape[0]} examples, "
          f"{len(mod_names)} classes")
    print(f"[Dataset] Window size: {window_size}, SNR range: {snrs.min()} to {snrs.max()} dB")

    return X, y, snrs, mod_names


def generate_synthetic_data(num_classes: int = 11,
                            num_per_class: int = 200,
                            window_size: int = 128,
                            snr_levels: Optional[List[int]] = None,
                            seed: int = 42
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Generate synthetic I/Q data mimicking real digital modulation schemes.
    
    This generates realistic-ish I/Q for: BPSK, QPSK, 8PSK, 16QAM, 64QAM,
    GFSK, AM-DSB, OOK, CPFSK, FM, PAM4 — enough to validate the pipeline.
    NOT for accuracy benchmarking (use real RadioML for that).
    
    Args:
        num_classes: Number of synthetic modulation classes
        num_per_class: Number of examples per class per SNR level
        window_size: I/Q samples per example
        snr_levels: List of SNR values in dB
        seed: Random seed
        
    Returns:
        X: np.ndarray of shape (N, 2, window_size)
        y: np.ndarray of shape (N,)
        snrs: np.ndarray of shape (N,)
        mod_names: List of synthetic class names
    """
    rng = np.random.RandomState(seed)

    if snr_levels is None:
        snr_levels = [0, 4, 8, 12, 18]

    mod_names_full = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM',
                      'GFSK', 'AM-DSB', 'OOK', 'CPFSK', 'FM', 'PAM4']
    mod_names = mod_names_full[:num_classes]
    t = np.linspace(0, 1, window_size, endpoint=False)
    fc = 8.0  # Carrier frequency (normalized)

    X_list, y_list, snr_list = [], [], []

    def _add_noise(I_sig, Q_sig, snr_db, rng):
        sig_power = np.mean(I_sig**2 + Q_sig**2) + 1e-12
        snr_lin = 10 ** (snr_db / 10.0)
        noise_std = np.sqrt(sig_power / snr_lin / 2.0)
        return I_sig + rng.normal(0, noise_std, len(I_sig)), \
               Q_sig + rng.normal(0, noise_std, len(Q_sig))

    for class_idx in range(num_classes):
        mod_type = mod_names[class_idx]
        for snr_db in snr_levels:
            for _ in range(num_per_class):
                sps = max(4, window_size // 16)  # Samples per symbol
                num_sym = window_size // sps

                if mod_type == 'BPSK':
                    symbols = rng.choice([-1, 1], num_sym)
                    I_bb = np.repeat(symbols, sps)[:window_size]
                    Q_bb = np.zeros(window_size)
                elif mod_type == 'QPSK':
                    angles = rng.choice([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4], num_sym)
                    I_bb = np.repeat(np.cos(angles), sps)[:window_size]
                    Q_bb = np.repeat(np.sin(angles), sps)[:window_size]
                elif mod_type == '8PSK':
                    angles = rng.choice(np.arange(8) * np.pi / 4, num_sym)
                    I_bb = np.repeat(np.cos(angles), sps)[:window_size]
                    Q_bb = np.repeat(np.sin(angles), sps)[:window_size]
                elif mod_type == '16QAM':
                    levels = [-3, -1, 1, 3]
                    I_sym = rng.choice(levels, num_sym).astype(float) / 3.0
                    Q_sym = rng.choice(levels, num_sym).astype(float) / 3.0
                    I_bb = np.repeat(I_sym, sps)[:window_size]
                    Q_bb = np.repeat(Q_sym, sps)[:window_size]
                elif mod_type == '64QAM':
                    levels = np.arange(-7, 8, 2).astype(float) / 7.0
                    I_sym = rng.choice(levels, num_sym)
                    Q_sym = rng.choice(levels, num_sym)
                    I_bb = np.repeat(I_sym, sps)[:window_size]
                    Q_bb = np.repeat(Q_sym, sps)[:window_size]
                elif mod_type == 'GFSK':
                    bits = rng.choice([-1.0, 1.0], num_sym)
                    freq_dev = np.repeat(bits * 0.3, sps)[:window_size]
                    phase = np.cumsum(freq_dev) * 2 * np.pi / sps
                    I_bb = np.cos(phase)
                    Q_bb = np.sin(phase)
                elif mod_type == 'AM-DSB':
                    msg_freq = rng.uniform(0.3, 2.0)
                    msg = np.cos(2 * np.pi * msg_freq * t)
                    env = 1.0 + 0.8 * msg
                    I_bb = env * np.cos(2 * np.pi * fc * t)
                    Q_bb = env * np.sin(2 * np.pi * fc * t)
                elif mod_type == 'OOK':
                    bits = rng.choice([0.0, 1.0], num_sym)
                    I_bb = np.repeat(bits, sps)[:window_size]
                    Q_bb = np.zeros(window_size)
                elif mod_type == 'CPFSK':
                    bits = rng.choice([-1.0, 1.0], num_sym)
                    freq_dev = np.repeat(bits * 0.5, sps)[:window_size]
                    phase = np.cumsum(freq_dev) * 2 * np.pi / sps
                    I_bb = np.cos(phase)
                    Q_bb = np.sin(phase)
                elif mod_type == 'FM':
                    msg_freq = rng.uniform(0.2, 1.5)
                    msg = np.sin(2 * np.pi * msg_freq * t)
                    phase = np.cumsum(msg) * 2 * np.pi * 0.4 / window_size
                    I_bb = np.cos(phase)
                    Q_bb = np.sin(phase)
                elif mod_type == 'PAM4':
                    levels = [-3.0, -1.0, 1.0, 3.0]
                    syms = rng.choice(levels, num_sym) / 3.0
                    I_bb = np.repeat(syms, sps)[:window_size]
                    Q_bb = np.zeros(window_size)
                else:
                    # Fallback: random PSK-like
                    angles = rng.uniform(0, 2*np.pi, num_sym)
                    I_bb = np.repeat(np.cos(angles), sps)[:window_size]
                    Q_bb = np.repeat(np.sin(angles), sps)[:window_size]

                # Pad to window_size if needed
                if len(I_bb) < window_size:
                    I_bb = np.pad(I_bb, (0, window_size - len(I_bb)))
                    Q_bb = np.pad(Q_bb, (0, window_size - len(Q_bb)))

                # Add random frequency/phase offset per sample
                fo = rng.uniform(-0.5, 0.5)
                po = rng.uniform(0, 2 * np.pi)
                rot = np.exp(1j * (2 * np.pi * fo * t + po))
                iq = (I_bb + 1j * Q_bb) * rot
                I_sig, Q_sig = iq.real, iq.imag

                # Add noise
                I_sig, Q_sig = _add_noise(I_sig, Q_sig, snr_db, rng)

                # Global normalization: scale to roughly [-1, 1]
                max_val = max(np.abs(I_sig).max(), np.abs(Q_sig).max(), 1e-10)
                scale = min(1.0 / max_val, 5.0)  # Don't over-amplify noise
                I_sig = np.clip(I_sig * scale, -1.0, 1.0)
                Q_sig = np.clip(Q_sig * scale, -1.0, 1.0)

                X_list.append(np.stack([I_sig, Q_sig], axis=0))
                y_list.append(class_idx)
                snr_list.append(snr_db)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    snrs = np.array(snr_list, dtype=np.int32)

    # Shuffle
    perm = rng.permutation(len(y))
    X, y, snrs = X[perm], y[perm], snrs[perm]

    print(f"[Dataset] Synthetic data generated: {X.shape[0]} examples, "
          f"{num_classes} classes, {len(snr_levels)} SNR levels")

    return X, y, snrs, mod_names


def load_dataset(version: str = "2016.10a",
                 data_dir: str = ".",
                 snr_filter: Optional[List[int]] = None,
                 window_size: int = 128,
                 use_synthetic: bool = False
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Universal dataset loader. Loads real or synthetic data based on config.
    
    Args:
        version: "2016.10a" or "2018.01A"
        data_dir: Directory containing dataset files
        snr_filter: Optional SNR filter
        window_size: Samples per window (for 2018.01A)
        use_synthetic: If True, use synthetic data regardless
        
    Returns:
        X, y, snrs, mod_names
    """
    if use_synthetic:
        num_classes = 11 if version == "2016.10a" else 24
        return generate_synthetic_data(
            num_classes=num_classes,
            window_size=window_size,
            snr_levels=list(range(-20, 20, 2)) if snr_filter is None else snr_filter
        )

    if version == "2016.10a":
        filepath = os.path.join(data_dir, "RML2016.10a_dict.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"RadioML 2016.10a not found at {filepath}\n"
                f"Download from: https://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2\n"
                f"Extract and place RML2016.10a_dict.pkl in {data_dir}/"
            )
        return load_radioml_2016(filepath, snr_filter)

    elif version == "2018.01A":
        filepath = os.path.join(data_dir, "GOLD_XYZ_OSC.0001_1024.hdf5")
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"RadioML 2018.01A not found at {filepath}\n"
                f"Download from: https://www.deepsig.ai/datasets\n"
                f"Place GOLD_XYZ_OSC.0001_1024.hdf5 in {data_dir}/"
            )
        return load_radioml_2018(filepath, snr_filter, window_size)

    else:
        raise ValueError(f"Unknown dataset version: {version}")


def train_test_split_by_snr(X: np.ndarray, y: np.ndarray, snrs: np.ndarray,
                             train_ratio: float = 0.7, seed: int = 42
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/test sets, stratified by class and SNR.
    
    Args:
        X, y, snrs: Dataset arrays
        train_ratio: Fraction for training
        seed: Random seed
        
    Returns:
        X_train, y_train, snrs_train, X_test, y_test, snrs_test
    """
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []

    unique_classes = np.unique(y)
    unique_snrs = np.unique(snrs)

    for cls in unique_classes:
        for snr in unique_snrs:
            mask = (y == cls) & (snrs == snr)
            indices = np.where(mask)[0]
            rng.shuffle(indices)
            split = int(len(indices) * train_ratio)
            train_idx.extend(indices[:split])
            test_idx.extend(indices[split:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return (X[train_idx], y[train_idx], snrs[train_idx],
            X[test_idx], y[test_idx], snrs[test_idx])
