"""3-level Haar wavelet decomposition (add/sub only, FPGA bit-exact)."""

import numpy as np
from typing import Dict, Tuple


def haar_decompose_one_level(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    One level of Haar wavelet decomposition.

    If the input length is odd, the last sample is repeated to make it even.

    Args:
        signal: 1D array of samples

    Returns:
        cA: approximation coefficients (low-freq), length = ceil(N/2)
        cD: detail coefficients (high-freq),        length = ceil(N/2)
    """
    N = len(signal)

    # Pad to even length if needed (repeat last sample)
    if N % 2 == 1:
        signal = np.append(signal, signal[-1])

    # Haar transform: sum and difference of consecutive pairs
    even = signal[0::2]  # x[0], x[2], x[4], ...
    odd  = signal[1::2]  # x[1], x[3], x[5], ...

    cA = even + odd   # Approximation (low-pass): add
    cD = even - odd   # Detail (high-pass): subtract

    return cA, cD


def haar_dwt_3level(beat: np.ndarray) -> Dict[str, np.ndarray]:
    """
    3-level Haar wavelet decomposition of a single ECG beat.

    Args:
        beat: 1D array of shape (187,) — one ECG beat

    Returns:
        dict with keys: 'cA3', 'cD3', 'cD2', 'cD1'
        Each value is a 1D numpy array.

    Sub-band lengths for input length 187:
        cD1: 94  (high-freq details)
        cD2: 47  (mid-high freq)
        cD3: 24  (mid freq)
        cA3: 24  (low-freq approximation)
    """
    # Level 1: 187 → pad to 188 → cA1(94), cD1(94)
    cA1, cD1 = haar_decompose_one_level(beat)

    # Level 2: 94 (even) → cA2(47), cD2(47)
    cA2, cD2 = haar_decompose_one_level(cA1)

    # Level 3: 47 → pad to 48 → cA3(24), cD3(24)
    cA3, cD3 = haar_decompose_one_level(cA2)

    return {"cA3": cA3, "cD3": cD3, "cD2": cD2, "cD1": cD1}


def haar_dwt_3level_batch(X: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Apply 3-level Haar DWT to a batch of ECG beats.

    Args:
        X: (N, 187) array of ECG beats

    Returns:
        dict of sub-bands, each (N, sub_band_length):
            'cA3': (N, 24)
            'cD3': (N, 24)
            'cD2': (N, 47)
            'cD1': (N, 94)
    """
    N = X.shape[0]
    results = {"cA3": [], "cD3": [], "cD2": [], "cD1": []}

    for i in range(N):
        decomp = haar_dwt_3level(X[i])
        for key in results:
            results[key].append(decomp[key])

    return {key: np.array(val) for key, val in results.items()}


def haar_dwt_3level_int(beat_int: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Integer-exact 3-level Haar DWT for FPGA verification.

    Uses the same add/subtract operations as the Verilog module.
    Input must be integer (e.g., int8 from quantize_for_fpga).

    Args:
        beat_int: 1D int array of shape (187,)

    Returns:
        dict of integer sub-bands (matching hardware exactly)
    """
    beat = beat_int.astype(np.int32)  # widen to avoid overflow

    # Level 1: pad 187 → 188
    if len(beat) % 2 == 1:
        beat = np.append(beat, beat[-1])
    cA1 = beat[0::2] + beat[1::2]
    cD1 = beat[0::2] - beat[1::2]

    # Level 2: 94 is even, no padding needed
    cA2 = cA1[0::2] + cA1[1::2]
    cD2 = cA1[0::2] - cA1[1::2]

    # Level 3: pad 47 → 48
    if len(cA2) % 2 == 1:
        cA2 = np.append(cA2, cA2[-1])
    cA3 = cA2[0::2] + cA2[1::2]
    cD3 = cA2[0::2] - cA2[1::2]

    return {"cA3": cA3, "cD3": cD3, "cD2": cD2, "cD1": cD1}


def haar_dwt_3level_int_batch(X_int: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Apply integer-exact 3-level Haar DWT to a batch of quantized beats.

    Args:
        X_int: (N, 187) int array (from quantize_for_fpga)

    Returns:
        dict of integer sub-bands, each (N, sub_band_length):
            'cA3': (N, 24),  'cD3': (N, 24)
            'cD2': (N, 47),  'cD1': (N, 94)
    """
    N = X_int.shape[0]
    results = {"cA3": [], "cD3": [], "cD2": [], "cD1": []}
    for i in range(N):
        decomp = haar_dwt_3level_int(X_int[i])
        for key in results:
            results[key].append(decomp[key])
    return {key: np.array(val) for key, val in results.items()}
