"""
HDC Encoder Module
Implements Hyperdimensional Computing encoding for I/Q signal data.

Encoding Pipeline:
  1. Level Quantization: Map float I/Q values to Q discrete levels
  2. Codebook Lookup: Map each level to a random D-bit binary vector
  3. I/Q Binding: XOR the I and Q level vectors (combines channels)
  4. N-gram Encoding: Capture temporal patterns via shifted XOR
  5. Bundling: Majority vote across all vectors in a window → single D-bit query

Author: HDC-AMC Team
"""

import numpy as np
from typing import Tuple, Optional


class HDCEncoder:
    """
    Hyperdimensional Computing encoder for I/Q signal classification.
    
    All operations are binary (XOR, majority vote, Hamming distance)
    mapping directly to LUT-only FPGA hardware with zero multipliers.
    
    Supports two encoding modes:
      - 'iq': Raw I/Q level encoding (needs phase-coherent data)
      - 'amp_phase': Amplitude + Phase-difference encoding (phase-rotation invariant)
    
    For RF modulation classification, 'amp_phase' is strongly recommended
    because real signals have random carrier phase/frequency offsets.
    """

    def __init__(self, D: int = 4096, Q: int = 16, n_gram: int = 3,
                 seed: int = 42, mode: str = 'amp_phase'):
        """
        Args:
            D: Hypervector dimension (number of bits). Higher = more accurate, more FPGA resources.
            Q: Number of quantization levels per channel. Must be power of 2 for FPGA.
            n_gram: N-gram length for temporal encoding. 1 = no temporal encoding.
            seed: Random seed for reproducible codebook generation.
            mode: 'iq' for raw I/Q encoding, 'amp_phase' for amplitude + phase-diff encoding.
        """
        self.D = D
        self.Q = Q
        self.n_gram = n_gram
        self.seed = seed
        self.mode = mode
        self.rng = np.random.RandomState(seed)

        # Generate random binary codebooks for two feature channels
        # Channel A: I (or Amplitude), Channel B: Q (or Phase-diff)
        # Each codebook: Q entries x D bits (stored as uint8 arrays of 0/1)
        self.codebook_I = self.rng.randint(0, 2, size=(Q, D)).astype(np.uint8)
        self.codebook_Q = self.rng.randint(0, 2, size=(Q, D)).astype(np.uint8)

        # Position vectors for N-gram encoding (N vectors of D bits each)
        # These provide temporal position sensitivity
        self.pos_vectors = self.rng.randint(0, 2, size=(n_gram, D)).astype(np.uint8)

    def quantize(self, values: np.ndarray, vmin: float = -1.0,
                 vmax: float = 1.0) -> np.ndarray:
        """
        Quantize float values to Q discrete levels [0, Q-1].
        
        Maps the range [vmin, vmax] linearly to [0, Q-1].
        For FPGA, this is a simple shift + truncate operation on fixed-point input.
        
        Args:
            values: Float array of any shape.
            vmin: Minimum value of the input range.
            vmax: Maximum value of the input range.
            
        Returns:
            Integer array of same shape with values in [0, Q-1].
        """
        clipped = np.clip(values, vmin, vmax)
        normalized = (clipped - vmin) / (vmax - vmin + 1e-12)  # → [0, 1]
        quantized = np.floor(normalized * self.Q).astype(np.int32)
        quantized = np.clip(quantized, 0, self.Q - 1)
        return quantized

    def preprocess_iq(self, I_samples: np.ndarray,
                      Q_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess I/Q samples into features for encoding.
        
        In 'iq' mode: returns raw I, Q (clipped to [-1, 1])
        In 'amp_phase' mode: returns amplitude and phase-difference
        
        Amplitude: sqrt(I^2 + Q^2) — captures envelope/power variations
        Phase-diff: successive phase differences — captures instantaneous frequency
        
        Both are invariant to constant carrier phase offset, which is critical
        for modulation classification of real RF signals.
        
        FPGA implementation:
          - Amplitude: CORDIC or alpha-max-beta-min approximation
          - Phase-diff: cross-product method or CORDIC
        
        Args:
            I_samples: Float array of shape (window_size,)
            Q_samples: Float array of shape (window_size,)
            
        Returns:
            Tuple of (feature_A, feature_B) arrays of shape (window_size,)
            Ranges: feature_A in [0, 1], feature_B in [-pi, pi] normalized to [-1, 1]
        """
        if self.mode == 'iq':
            return (np.clip(I_samples, -1.0, 1.0),
                    np.clip(Q_samples, -1.0, 1.0))
        
        # Amplitude: sqrt(I^2 + Q^2), normalized to [0, 1]
        amplitude = np.sqrt(I_samples**2 + Q_samples**2)
        amp_max = amplitude.max() + 1e-12
        amplitude = amplitude / amp_max  # Normalize to [0, 1]
        
        # Instantaneous phase: atan2(Q, I)
        phase = np.arctan2(Q_samples, I_samples)  # [-pi, pi]
        
        # Phase difference (instantaneous frequency proxy)
        # Wraps to [-pi, pi] via angle of complex division
        complex_sig = I_samples + 1j * Q_samples
        # Avoid division by zero
        complex_sig_safe = np.where(np.abs(complex_sig) < 1e-12,
                                     1e-12 + 0j, complex_sig)
        phase_diff = np.angle(complex_sig_safe[1:] * np.conj(complex_sig_safe[:-1]))
        # Pad to same length (prepend 0)
        phase_diff = np.concatenate([[0.0], phase_diff])
        # Normalize to [-1, 1]
        phase_diff = phase_diff / np.pi
        
        return amplitude, phase_diff

    def encode_sample(self, I_level: int, Q_level: int) -> np.ndarray:
        """
        Encode a single I/Q sample into a D-bit hypervector.
        
        Hardware equivalent: 
            codebook_I[level_I] XOR codebook_Q[level_Q]
            = 1 BRAM read + 1 BRAM read + D-bit XOR gate
        
        Args:
            I_level: Quantized I channel value [0, Q-1]
            Q_level: Quantized Q channel value [0, Q-1]
            
        Returns:
            D-bit binary vector (np.uint8 array of 0s and 1s)
        """
        return np.bitwise_xor(self.codebook_I[I_level],
                              self.codebook_Q[Q_level])

    def encode_ngram(self, sample_vectors: np.ndarray) -> np.ndarray:
        """
        Apply N-gram temporal encoding using circular permutation + XOR.
        
        For N-gram of length n:
            ngram[t] = rho^(n-1)(v[t-n+1]) XOR rho^(n-2)(v[t-n+2]) XOR ... XOR v[t]
        
        where rho is circular left shift by 1 bit.
        
        Hardware equivalent: D-bit shift registers + XOR chains
        
        Args:
            sample_vectors: Array of shape (num_samples, D) — encoded I/Q samples
            
        Returns:
            Array of shape (num_samples - n_gram + 1, D) — N-gram encoded vectors
        """
        if self.n_gram == 1:
            return sample_vectors

        num_samples = sample_vectors.shape[0]
        num_ngrams = num_samples - self.n_gram + 1

        if num_ngrams <= 0:
            raise ValueError(f"Not enough samples ({num_samples}) for "
                             f"n_gram={self.n_gram}")

        ngram_vectors = np.zeros((num_ngrams, self.D), dtype=np.uint8)

        for t in range(num_ngrams):
            result = np.zeros(self.D, dtype=np.uint8)
            for k in range(self.n_gram):
                # Get the sample vector at position t + k
                vec = sample_vectors[t + k].copy()
                # Apply circular left shift by (n_gram - 1 - k) positions
                shift_amount = self.n_gram - 1 - k
                if shift_amount > 0:
                    vec = np.roll(vec, -shift_amount)
                # XOR into result
                result = np.bitwise_xor(result, vec)
            ngram_vectors[t] = result

        return ngram_vectors

    def bundle(self, vectors: np.ndarray) -> np.ndarray:
        """
        Bundle multiple D-bit vectors into a single D-bit vector using majority vote.
        
        For each bit position: if more than half the vectors have a 1, output 1.
        Ties are broken randomly (or by setting to 0 for determinism).
        
        Hardware equivalent: D counters (log2(window_size) bits each) + threshold comparators
        
        Args:
            vectors: Array of shape (num_vectors, D) — vectors to bundle
            
        Returns:
            Single D-bit binary vector (np.uint8 array)
        """
        num_vectors = vectors.shape[0]
        # Sum along the vector axis (counts how many 1s per bit position)
        counts = vectors.sum(axis=0)  # shape: (D,)
        # Majority vote: 1 if count > num_vectors/2, else 0
        threshold = num_vectors / 2.0
        bundled = (counts > threshold).astype(np.uint8)
        # Break ties (count == threshold) randomly for odd vector counts this doesn't happen
        ties = (counts == threshold)
        if np.any(ties):
            bundled[ties] = self.rng.randint(0, 2, size=ties.sum()).astype(np.uint8)
        return bundled

    def encode_window(self, I_samples: np.ndarray,
                      Q_samples: np.ndarray) -> np.ndarray:
        """
        Full encoding pipeline for one classification window.
        
        Pipeline: Preprocess → Quantize → Lookup → Bind → N-gram → Bundle → D-bit query vector
        
        Args:
            I_samples: Float array of shape (window_size,) — I channel time series
            Q_samples: Float array of shape (window_size,) — Q channel time series
            
        Returns:
            Single D-bit query hypervector (np.uint8 array of shape (D,))
        """
        # Step 1: Preprocess
        feat_A, feat_B = self.preprocess_iq(I_samples, Q_samples)

        # Step 2: Quantize features to discrete levels
        if self.mode == 'amp_phase':
            A_levels = self.quantize(feat_A, vmin=0.0, vmax=1.0)
            B_levels = self.quantize(feat_B, vmin=-1.0, vmax=1.0)
        else:
            A_levels = self.quantize(feat_A, vmin=-1.0, vmax=1.0)
            B_levels = self.quantize(feat_B, vmin=-1.0, vmax=1.0)

        # Step 3: Encode each sample (codebook lookup + binding)
        num_samples = len(I_samples)
        sample_vectors = np.zeros((num_samples, self.D), dtype=np.uint8)
        for t in range(num_samples):
            sample_vectors[t] = self.encode_sample(A_levels[t], B_levels[t])

        # Step 4: Apply N-gram temporal encoding
        ngram_vectors = self.encode_ngram(sample_vectors)

        # Step 5: Bundle all N-gram vectors into a single query vector
        query_vector = self.bundle(ngram_vectors)

        return query_vector

    def encode_window_vectorized(self, I_samples: np.ndarray,
                                  Q_samples: np.ndarray) -> np.ndarray:
        """
        Vectorized (fast) encoding pipeline. Same result as encode_window
        but uses numpy broadcasting for speed on large datasets.
        
        Pipeline: Preprocess → Quantize → Lookup → Bind → N-gram → Bundle
        
        Args:
            I_samples: Float array of shape (window_size,)
            Q_samples: Float array of shape (window_size,)
            
        Returns:
            Single D-bit query hypervector
        """
        # Preprocess: convert I/Q to features (amplitude/phase-diff or raw I/Q)
        feat_A, feat_B = self.preprocess_iq(I_samples, Q_samples)

        # Quantize both feature channels
        if self.mode == 'amp_phase':
            A_levels = self.quantize(feat_A, vmin=0.0, vmax=1.0)
            B_levels = self.quantize(feat_B, vmin=-1.0, vmax=1.0)
        else:
            A_levels = self.quantize(feat_A, vmin=-1.0, vmax=1.0)
            B_levels = self.quantize(feat_B, vmin=-1.0, vmax=1.0)

        # Vectorized codebook lookup + binding
        sample_vectors = np.bitwise_xor(
            self.codebook_I[A_levels],   # shape: (window_size, D)
            self.codebook_Q[B_levels]    # shape: (window_size, D)
        )

        # N-gram encoding (vectorized)
        if self.n_gram > 1:
            ngram_vecs = np.zeros_like(sample_vectors[:sample_vectors.shape[0] - self.n_gram + 1])
            for k in range(self.n_gram):
                shift_amount = self.n_gram - 1 - k
                sliced = sample_vectors[k:k + ngram_vecs.shape[0]]
                if shift_amount > 0:
                    sliced = np.roll(sliced, -shift_amount, axis=1)
                ngram_vecs = np.bitwise_xor(ngram_vecs, sliced)
            sample_vectors = ngram_vecs

        # Bundle via majority vote
        counts = sample_vectors.astype(np.int32).sum(axis=0)
        threshold = sample_vectors.shape[0] / 2.0
        query = (counts > threshold).astype(np.uint8)
        ties = (counts == threshold)
        if np.any(ties):
            query[ties] = self.rng.randint(0, 2, size=ties.sum()).astype(np.uint8)

        return query

    def encode_batch(self, IQ_data: np.ndarray, batch_chunk: int = 1000) -> np.ndarray:
        """
        Encode a batch of I/Q signal windows using chunked vectorization.
        
        Processes `batch_chunk` windows at a time using fully vectorized numpy
        operations (no per-window Python loop inside each chunk).
        
        Args:
            IQ_data: Array of shape (batch_size, 2, window_size)
            batch_chunk: Number of windows to process at once (memory tradeoff)
        
        Returns:
            Array of shape (batch_size, D) — one query hypervector per window
        """
        batch_size = IQ_data.shape[0]
        queries = np.zeros((batch_size, self.D), dtype=np.uint8)

        for start in range(0, batch_size, batch_chunk):
            end = min(start + batch_chunk, batch_size)
            chunk = IQ_data[start:end]  # (C, 2, W)
            C = chunk.shape[0]
            W = chunk.shape[2]

            I_all = chunk[:, 0, :]  # (C, W)
            Q_all = chunk[:, 1, :]  # (C, W)

            # Preprocess: amplitude + phase-diff for all windows at once
            if self.mode == 'amp_phase':
                amplitude = np.sqrt(I_all**2 + Q_all**2)  # (C, W)
                amp_max = amplitude.max(axis=1, keepdims=True) + 1e-12
                feat_A = amplitude / amp_max  # (C, W) in [0, 1]

                complex_sig = I_all + 1j * Q_all
                complex_safe = np.where(np.abs(complex_sig) < 1e-12,
                                        1e-12 + 0j, complex_sig)
                phase_diff = np.angle(complex_safe[:, 1:] * np.conj(complex_safe[:, :-1]))
                phase_diff = np.concatenate([np.zeros((C, 1)), phase_diff], axis=1)
                feat_B = phase_diff / np.pi  # (C, W) in [-1, 1]

                A_levels = self.quantize(feat_A, vmin=0.0, vmax=1.0)    # (C, W)
                B_levels = self.quantize(feat_B, vmin=-1.0, vmax=1.0)   # (C, W)
            else:
                feat_A = np.clip(I_all, -1.0, 1.0)
                feat_B = np.clip(Q_all, -1.0, 1.0)
                A_levels = self.quantize(feat_A, vmin=-1.0, vmax=1.0)
                B_levels = self.quantize(feat_B, vmin=-1.0, vmax=1.0)

            # Vectorized codebook lookup + XOR binding: shape (C, W, D)
            sample_vecs = np.bitwise_xor(
                self.codebook_I[A_levels],   # (C, W, D)
                self.codebook_Q[B_levels]    # (C, W, D)
            )

            # N-gram encoding (vectorized across batch)
            if self.n_gram > 1:
                L = W - self.n_gram + 1
                ngram_vecs = np.zeros((C, L, self.D), dtype=np.uint8)
                for k in range(self.n_gram):
                    shift_amount = self.n_gram - 1 - k
                    sliced = sample_vecs[:, k:k + L, :]  # (C, L, D)
                    if shift_amount > 0:
                        sliced = np.roll(sliced, -shift_amount, axis=2)
                    ngram_vecs = np.bitwise_xor(ngram_vecs, sliced)
                sample_vecs = ngram_vecs

            # Bundle via majority vote: sum along window axis
            counts = sample_vecs.astype(np.int32).sum(axis=1)  # (C, D)
            threshold = sample_vecs.shape[1] / 2.0
            q = (counts > threshold).astype(np.uint8)
            ties = (counts == threshold)
            if np.any(ties):
                q[ties] = self.rng.randint(0, 2, size=ties.sum()).astype(np.uint8)

            queries[start:end] = q

        return queries

    def get_codebooks_packed(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get codebooks packed as binary arrays (for FPGA export).
        
        Returns:
            Tuple of (codebook_I_packed, codebook_Q_packed)
            Each has shape (Q, D//8) as uint8 bytes.
        """
        cb_I = np.packbits(self.codebook_I, axis=1)  # (Q, D//8)
        cb_Q = np.packbits(self.codebook_Q, axis=1)
        return cb_I, cb_Q

    @staticmethod
    def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
        """
        Compute Hamming distance between two binary vectors.
        
        Hardware: XOR + popcount (pure combinational logic in LUTs)
        
        Args:
            a, b: Binary vectors of shape (D,)
            
        Returns:
            Integer Hamming distance
        """
        return int(np.bitwise_xor(a, b).sum())

    @staticmethod
    def hamming_distance_batch(query: np.ndarray,
                                prototypes: np.ndarray) -> np.ndarray:
        """
        Compute Hamming distance from one query to multiple prototypes.
        
        Args:
            query: Binary vector of shape (D,)
            prototypes: Binary matrix of shape (num_classes, D)
            
        Returns:
            Array of shape (num_classes,) with Hamming distances
        """
        return np.bitwise_xor(query, prototypes).sum(axis=1)
