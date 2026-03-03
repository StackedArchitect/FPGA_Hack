#!/usr/bin/env python3
"""
HDC-AMC: Hyperdimensional Computing for Automatic Modulation Classification
Main training, evaluation, and FPGA export pipeline.

Usage:
  python main.py                        # Run with synthetic data (for testing)
  python main.py --dataset 2016.10a     # Run with RadioML 2016.10a
  python main.py --dataset 2018.01A     # Run with RadioML 2018.01A
  python main.py --sweep                # Run hyperparameter sweep (D, Q, N-gram)
  python main.py --export               # Export trained model for FPGA
"""

import sys
import os
import time
import argparse
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import *
from src.hdc_encoder import HDCEncoder
from src.hdc_classifier import HDCClassifier
from src.dataset_loader import load_dataset, train_test_split_by_snr
from src.evaluate import (generate_full_report, compute_accuracy_by_snr,
                          plot_accuracy_vs_snr, plot_accuracy_vs_dimension,
                          print_summary)
from src.export_to_fpga import export_all


def run_single(D: int, Q: int, n_gram: int, X_train, y_train, snrs_train,
               X_test, y_test, snrs_test, mod_names, retrain_iters: int = 2,
               verbose: bool = True):
    """
    Run a single HDC experiment with given hyperparameters.
    
    Returns:
        classifier, y_pred, overall_accuracy, snr_accuracies
    """
    # Create encoder and classifier
    encoder = HDCEncoder(D=D, Q=Q, n_gram=n_gram, seed=RANDOM_SEED,
                         mode=ENCODE_MODE)
    classifier = HDCClassifier(encoder, num_classes=len(mod_names))

    # Train
    t0 = time.time()
    if retrain_iters > 0:
        classifier.retrain_iterative(X_train, y_train,
                                      iterations=retrain_iters, verbose=verbose)
    else:
        classifier.train(X_train, y_train, verbose=verbose)
    train_time = time.time() - t0

    # Predict
    t0 = time.time()
    y_pred = classifier.predict(X_test, verbose=verbose)
    pred_time = time.time() - t0

    # Evaluate
    overall_acc = np.mean(y_pred == y_test)
    snr_acc = compute_accuracy_by_snr(y_test, y_pred, snrs_test)

    if verbose:
        print(f"\n  D={D}, Q={Q}, N={n_gram}: "
              f"Accuracy={overall_acc:.4f}, "
              f"Train={train_time:.1f}s, Pred={pred_time:.1f}s")

    return classifier, y_pred, overall_acc, snr_acc


def run_dimension_sweep(X_train, y_train, snrs_train,
                        X_test, y_test, snrs_test, mod_names):
    """
    Sweep hypervector dimension D and collect accuracy results.
    This generates the key paper figure: accuracy vs dimension Pareto curve.
    """
    print("\n" + "=" * 70)
    print("  DIMENSION SWEEP (Accuracy vs D)")
    print("=" * 70)

    dim_acc = {}
    dim_snr_acc = {}

    for d in D_SWEEP:
        print(f"\n--- D = {d} ---")
        _, _, acc, snr_acc = run_single(
            D=d, Q=Q, n_gram=N_GRAM,
            X_train=X_train, y_train=y_train, snrs_train=snrs_train,
            X_test=X_test, y_test=y_test, snrs_test=snrs_test,
            mod_names=mod_names, retrain_iters=1, verbose=True
        )
        dim_acc[d] = acc
        dim_snr_acc[d] = snr_acc

    # Plot
    plot_accuracy_vs_dimension(
        dim_acc,
        title="HDC-AMC: Accuracy vs Hypervector Dimension",
        save_path=os.path.join(RESULTS_DIR, "accuracy_vs_D.png")
    )

    # Plot all SNR curves together
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    for i, d in enumerate(sorted(dim_snr_acc.keys())):
        snrs_sorted = sorted(dim_snr_acc[d].keys())
        accs = [dim_snr_acc[d][s] for s in snrs_sorted]
        ax.plot(snrs_sorted, accs, '-o', color=colors[i % len(colors)],
                linewidth=1.5, markersize=4, label=f'D={d}')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('HDC-AMC: Accuracy vs SNR for Different Dimensions', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "snr_curves_by_D.png"), dpi=150)
    plt.close()

    print("\n  Dimension Sweep Results:")
    for d in sorted(dim_acc.keys()):
        print(f"    D={d:>5d}: {dim_acc[d]:.4f}")

    return dim_acc


def main():
    parser = argparse.ArgumentParser(description='HDC-AMC: Hyperdimensional Modulation Classification')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset version: "2016.10a", "2018.01A", or "synthetic"')
    parser.add_argument('--D', type=int, default=None, help='Hypervector dimension')
    parser.add_argument('--Q', type=int, default=None, help='Quantization levels')
    parser.add_argument('--ngram', type=int, default=None, help='N-gram length')
    parser.add_argument('--sweep', action='store_true', help='Run dimension sweep')
    parser.add_argument('--export', action='store_true', help='Export for FPGA')
    parser.add_argument('--retrain', type=int, default=2, help='Retrain iterations (0=none)')
    parser.add_argument('--snr-filter', type=int, nargs='+', default=None,
                        help='SNR values to include (e.g., 0 2 4 6 8 10)')
    args = parser.parse_args()

    # Override config with command-line args
    d = args.D if args.D else D
    q = args.Q if args.Q else Q
    n = args.ngram if args.ngram else N_GRAM

    # Determine dataset
    use_synthetic = False
    dataset_ver = DATASET_VERSION
    if args.dataset:
        if args.dataset.lower() == 'synthetic':
            use_synthetic = True
        else:
            dataset_ver = args.dataset

    # Check if real dataset exists, fall back to synthetic
    if not use_synthetic:
        try:
            X, y, snrs, mod_names = load_dataset(
                version=dataset_ver,
                data_dir=DATA_DIR,
                snr_filter=args.snr_filter,
                window_size=WINDOW_SIZE,
                use_synthetic=False
            )
        except FileNotFoundError as e:
            print(f"\n[WARNING] {e}")
            print("[WARNING] Falling back to synthetic data for pipeline testing.\n")
            use_synthetic = True

    if use_synthetic:
        num_cls = 11 if dataset_ver == "2016.10a" else 24
        X, y, snrs, mod_names = load_dataset(
            version=dataset_ver,
            data_dir=DATA_DIR,
            snr_filter=args.snr_filter,
            window_size=WINDOW_SIZE,
            use_synthetic=True
        )

    # Split data
    X_train, y_train, snrs_train, X_test, y_test, snrs_test = \
        train_test_split_by_snr(X, y, snrs, train_ratio=TRAIN_SPLIT, seed=RANDOM_SEED)

    print(f"\n[Data] Train: {len(y_train)}, Test: {len(y_test)}")
    print(f"[Data] Classes: {len(mod_names)}, Window: {X.shape[2]} samples")

    # Run dimension sweep if requested
    if args.sweep:
        run_dimension_sweep(X_train, y_train, snrs_train,
                           X_test, y_test, snrs_test, mod_names)
        return

    # Run single experiment
    print(f"\n[Config] D={d}, Q={q}, N-gram={n}, Retrain={args.retrain}")
    classifier, y_pred, overall_acc, snr_acc = run_single(
        D=d, Q=q, n_gram=n,
        X_train=X_train, y_train=y_train, snrs_train=snrs_train,
        X_test=X_test, y_test=y_test, snrs_test=snrs_test,
        mod_names=mod_names, retrain_iters=args.retrain, verbose=True
    )

    # Generate full report
    generate_full_report(y_test, y_pred, snrs_test, mod_names,
                         D=d, Q=q, n_gram=n, results_dir=RESULTS_DIR)

    # Save model
    model_path = os.path.join(RESULTS_DIR, f"hdc_model_D{d}_Q{q}_N{n}.npz")
    classifier.save_model(model_path)

    # Export for FPGA if requested
    if args.export:
        print("\n" + "=" * 70)
        print("  EXPORTING FOR FPGA")
        print("=" * 70)
        export_all(
            classifier=classifier,
            X_test=X_test, y_test=y_test, y_pred=y_pred,
            output_dir=EXPORT_DIR,
            chunk_width=FPGA_CHUNK_WIDTH,
            window_size=X_test.shape[2]
        )

    print(f"\n{'='*70}")
    print(f"  DONE! Overall accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    print(f"  Results saved to: {RESULTS_DIR}/")
    if args.export:
        print(f"  FPGA exports saved to: {EXPORT_DIR}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
