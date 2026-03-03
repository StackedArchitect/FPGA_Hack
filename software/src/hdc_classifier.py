"""
HDC Classifier Module
Implements single-pass training and Hamming-distance inference for HDC.

Key Property: Training is a SINGLE PASS through the data — no backpropagation,
no gradient computation, no iterative optimization. Each training example is
encoded and added to its class prototype vector. This means:
  - Training can run ON the FPGA itself (on-device learning)
  - Training time scales linearly with dataset size
  - No hyperparameter tuning for training (unlike learning rate, epochs, etc.)
"""

import numpy as np
from typing import Tuple, Optional, Dict
from .hdc_encoder import HDCEncoder


class HDCClassifier:
    """
    Hyperdimensional Computing classifier for I/Q modulation signals.
    
    Training: Encode all training examples → accumulate per-class → threshold → prototypes
    Inference: Encode query → Hamming distance to all prototypes → argmin
    """

    def __init__(self, encoder: HDCEncoder, num_classes: int):
        """
        Args:
            encoder: HDCEncoder instance with codebooks
            num_classes: Number of modulation classes
        """
        self.encoder = encoder
        self.num_classes = num_classes
        self.D = encoder.D

        # Class prototypes: binary vectors representing each class
        self.prototypes = np.zeros((num_classes, self.D), dtype=np.uint8)
        # Accumulator for training (integer counts before thresholding)
        self._accumulators = np.zeros((num_classes, self.D), dtype=np.int32)
        self._counts = np.zeros(num_classes, dtype=np.int32)
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              verbose: bool = True) -> None:
        """
        Single-pass training: encode all examples at once, then accumulate per-class.
        
        This is the HDC equivalent of "training" — no backprop, no gradients, no epochs.
        
        Args:
            X_train: Training data, shape (N, 2, window_size)
            y_train: Training labels, shape (N,) with values in [0, num_classes-1]
            verbose: Print progress
        """
        N = X_train.shape[0]

        # Reset accumulators
        self._accumulators[:] = 0
        self._counts[:] = 0

        if verbose:
            print(f"[HDC Train] Encoding {N} training examples (D={self.D}, "
                  f"Q={self.encoder.Q}, N-gram={self.encoder.n_gram})...")

        # Batch encode ALL training examples at once
        all_queries = self.encoder.encode_batch(IQ_data=X_train)

        if verbose:
            print(f"[HDC Train] Encoding complete. Accumulating prototypes...")

        # Accumulate per class
        for c in range(self.num_classes):
            mask = (y_train == c)
            self._counts[c] = mask.sum()
            if self._counts[c] > 0:
                self._accumulators[c] = all_queries[mask].astype(np.int32).sum(axis=0)

        # Threshold: majority vote for each class
        for c in range(self.num_classes):
            if self._counts[c] > 0:
                threshold = self._counts[c] / 2.0
                self.prototypes[c] = (self._accumulators[c] > threshold).astype(np.uint8)
                # Break ties
                ties = (self._accumulators[c] == threshold)
                if np.any(ties):
                    self.prototypes[c][ties] = 0  # Deterministic tie-breaking

        self.is_trained = True

        if verbose:
            print(f"[HDC Train] Done. Class distribution: "
                  f"{dict(zip(range(self.num_classes), self._counts))}")

    def predict_one(self, query: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Classify a single query hypervector.
        
        Hardware equivalent: XOR with each prototype → popcount → argmin
        
        Args:
            query: D-bit binary vector
            
        Returns:
            (predicted_class, hamming_distances)
        """
        distances = HDCEncoder.hamming_distance_batch(query, self.prototypes)
        predicted = np.argmin(distances)
        return int(predicted), distances

    def predict(self, X_test: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Classify a batch of I/Q windows (vectorized).
        
        Args:
            X_test: Test data, shape (N, 2, window_size)
            verbose: Print progress
            
        Returns:
            Predicted labels, shape (N,)
        """
        N = X_test.shape[0]

        if verbose:
            print(f"[HDC Predict] Encoding {N} examples...")

        # Batch encode
        queries = self.encoder.encode_batch(IQ_data=X_test)

        if verbose:
            print(f"[HDC Predict] Classifying...")

        # Vectorized Hamming distance: XOR + sum
        predictions = self.predict_from_encoded(queries)

        return predictions

    def predict_from_encoded(self, queries: np.ndarray) -> np.ndarray:
        """
        Classify pre-encoded query vectors (fully vectorized).
        
        Args:
            queries: Encoded vectors, shape (N, D)
            
        Returns:
            Predicted labels, shape (N,)
        """
        # Compute all Hamming distances at once
        # queries: (N, D), prototypes: (num_classes, D)
        # XOR → sum over D axis → (N, num_classes) distance matrix
        N = queries.shape[0]
        predictions = np.zeros(N, dtype=np.int32)

        # Process in chunks to avoid massive memory allocation
        chunk = 2000
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            q = queries[start:end]  # (C, D)
            # Broadcast: (C, 1, D) XOR (1, num_classes, D) → (C, num_classes, D)
            dists = np.bitwise_xor(
                q[:, np.newaxis, :], 
                self.prototypes[np.newaxis, :, :]
            ).sum(axis=2)  # (C, num_classes)
            predictions[start:end] = np.argmin(dists, axis=1)

        return predictions

    def retrain_iterative(self, X_train: np.ndarray, y_train: np.ndarray,
                          iterations: int = 3, lr: float = 1.0,
                          verbose: bool = True) -> None:
        """
        Iterative retraining: after initial training, reclassify training examples
        and update prototypes for misclassified ones.
        
        Uses batch encoding (fast) + vectorized correction.
        
        Args:
            X_train: Training data
            y_train: Training labels
            iterations: Number of refinement iterations
            lr: Learning rate for corrections (1.0 = full, 0.5 = half)
            verbose: Print progress
        """
        # Initial training (includes batch encoding)
        self.train(X_train, y_train, verbose=verbose)

        # Encode training data ONCE — reuse across all retrain iterations
        if verbose:
            print(f"\n[HDC Retrain] Pre-encoding all training data for retraining...")
        all_queries = self.encoder.encode_batch(IQ_data=X_train)

        for it in range(iterations):
            if verbose:
                print(f"\n[HDC Retrain] Iteration {it+1}/{iterations}...")

            # Classify all training examples (vectorized)
            preds = self.predict_from_encoded(all_queries)

            # Find misclassifications
            wrong = (preds != y_train)
            corrections = wrong.sum()

            if corrections == 0:
                if verbose:
                    print(f"  Converged at iteration {it+1}!")
                break

            # Work on a COPY of accumulators
            new_acc = self._accumulators.copy()

            # Vectorized correction: for each misclassified example,
            # add to correct class accumulator, subtract from wrong class
            wrong_idx = np.where(wrong)[0]
            wrong_queries = all_queries[wrong_idx].astype(np.int32)
            wrong_true = y_train[wrong_idx]
            wrong_pred = preds[wrong_idx]

            # Add to correct class
            for c in range(self.num_classes):
                mask = (wrong_true == c)
                if mask.any():
                    new_acc[c] += (wrong_queries[mask].sum(axis=0) * lr).astype(np.int32)

            # Subtract from wrong class (conservative: 0.5× weight)
            for c in range(self.num_classes):
                mask = (wrong_pred == c)
                if mask.any():
                    new_acc[c] -= (wrong_queries[mask].sum(axis=0) * lr * 0.5).astype(np.int32)

            # Update accumulators and re-threshold
            self._accumulators = new_acc
            for c in range(self.num_classes):
                if self._counts[c] > 0:
                    threshold = self._counts[c] / 2.0
                    self.prototypes[c] = (self._accumulators[c] > threshold).astype(np.uint8)

            if verbose:
                print(f"  Corrections: {corrections}/{len(y_train)} "
                      f"({100*corrections/len(y_train):.1f}%)")

    def get_prototypes_packed(self) -> np.ndarray:
        """
        Get prototypes packed as bytes for FPGA export.
        
        Returns:
            Array of shape (num_classes, D//8) as uint8
        """
        return np.packbits(self.prototypes, axis=1)

    def save_model(self, filepath: str) -> None:
        """Save trained model (codebooks + prototypes) to numpy archive."""
        np.savez(filepath,
                 codebook_I=self.encoder.codebook_I,
                 codebook_Q=self.encoder.codebook_Q,
                 prototypes=self.prototypes,
                 D=self.D,
                 Q=self.encoder.Q,
                 n_gram=self.encoder.n_gram,
                 num_classes=self.num_classes,
                 counts=self._counts)
        print(f"[HDC] Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load trained model from numpy archive."""
        data = np.load(filepath)
        self.encoder.codebook_I = data['codebook_I']
        self.encoder.codebook_Q = data['codebook_Q']
        self.prototypes = data['prototypes']
        self._counts = data['counts']
        self.is_trained = True
        print(f"[HDC] Model loaded from {filepath}")
