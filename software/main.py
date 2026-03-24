#!/usr/bin/env python3
"""
WaveBNN-ECG: Full Pipeline
===========================
Run the complete workflow: Load data → Wavelet DWT → Train BNN → Evaluate → Export for FPGA.

Usage:
    python main.py                          # Full pipeline (train + eval + export)
    python main.py --epochs 50              # Custom epoch count
    python main.py --eval-only              # Evaluate saved model (skip training)
    python main.py --export-only            # Export saved model to .mem files
"""

import argparse
import os
import sys
import time
import numpy as np
import torch

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    AAMI_CLASSES, NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    RANDOM_SEED, MODELS_DIR, RESULTS_DIR, EXPORT_DIR,
    FPGA_BOARD, FPGA_PART, SUBBAND_LENGTHS, CONCAT_BITS, FC1_OUT,
    USE_CLASS_WEIGHTS,
)
from src.dataset import load_kaggle_mitbih, quantize_for_fpga
from src.wavelet import haar_dwt_3level_batch, haar_dwt_3level_int, haar_dwt_3level_int_batch
from src.bnn import WaveBNN
from src.train import (
    make_dataloaders, compute_class_weights, train_model,
    full_evaluation, export_for_fpga,
)


def print_header():
    print("=" * 65)
    print("  WaveBNN-ECG: Wavelet + BNN for ECG Arrhythmia Detection")
    print("  Target: {} ({})".format(FPGA_BOARD, FPGA_PART))
    print("=" * 65)


def step_load_data():
    """Step 1: Load and inspect the MIT-BIH dataset."""
    print("\n" + "─" * 65)
    print("  STEP 1: Load MIT-BIH Dataset")
    print("─" * 65)

    X_train, y_train, X_test, y_test = load_kaggle_mitbih()

    print(f"\n  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")

    # Class distribution
    for name, y in [("Train", y_train), ("Test", y_test)]:
        counts = np.bincount(y, minlength=NUM_CLASSES)
        dist = "  ".join(f"{AAMI_CLASSES[i]}={counts[i]:>5d}" for i in range(NUM_CLASSES))
        print(f"  {name}: {dist}  (total: {len(y):,})")

    return X_train, y_train, X_test, y_test


def step_wavelet(X_train, X_test):
    """Step 2: Quantize + integer Haar wavelet (FPGA bit-exact path)."""
    print("\n" + "─" * 65)
    print("  STEP 2: Quantize → Integer Haar Wavelet (3-level, FPGA-exact)")
    print("─" * 65)
    print("  Quantize: float → int8 (map ±3σ to ±127)")
    print("  Haar DWT: add/subtract only (zero multipliers, FPGA bit-exact)")

    t0 = time.time()
    X_train_q = quantize_for_fpga(X_train)
    X_test_q = quantize_for_fpga(X_test)

    train_sub_int = haar_dwt_3level_int_batch(X_train_q)
    test_sub_int = haar_dwt_3level_int_batch(X_test_q)
    elapsed = time.time() - t0

    # Convert int32 → float32 for PyTorch (values remain exact integers)
    train_sub = {k: v.astype(np.float32) for k, v in train_sub_int.items()}
    test_sub = {k: v.astype(np.float32) for k, v in test_sub_int.items()}

    print(f"\n  Quantize + decompose done in {elapsed:.2f}s")
    for key in ["cA3", "cD3", "cD2", "cD1"]:
        v = train_sub_int[key]
        vmax = max(abs(int(v.min())), abs(int(v.max())))
        bits = int(np.ceil(np.log2(vmax + 1))) + 1 if vmax > 0 else 2
        print(f"    {key}: train {v.shape}, range [{v.min():>6d}, {v.max():>6d}] → {bits}-bit signed")

    total_features = sum(v.shape[1] for v in train_sub.values())
    print(f"\n  Total features per beat: {total_features}")
    print(f"  Expected concat after BNN branches: {CONCAT_BITS} bits")
    print(f"  Training on FPGA-exact integer sub-bands (thresholds will match HW)")

    return train_sub, test_sub


def step_train(model, train_sub, y_train, test_sub, y_test, device, epochs):
    """Step 3: Train the WaveBNN model."""
    print("\n" + "─" * 65)
    print("  STEP 3: Train WaveBNN ({} epochs)".format(epochs))
    print("─" * 65)

    # Model summary
    ops = model.count_binary_ops()
    total_params = sum(p.numel() for p in model.parameters())
    binary_params = ops["total_binary"]
    print(f"\n  Parameters: {total_params:,} total")
    print(f"  Binary ops per inference: {binary_params:,} XNOR+popcount")
    print(f"  Real MACs (fc2 only): {ops['fc2_real_macs']:,}")
    print(f"  Concat vector: {CONCAT_BITS} bits → FC1({FC1_OUT}) → FC2({NUM_CLASSES})")

    train_loader, test_loader = make_dataloaders(train_sub, y_train, test_sub, y_test)

    class_weights = None
    if USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(y_train)
        print(f"\n  Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")

    print()
    history = train_model(model, train_loader, test_loader, device,
                          num_epochs=epochs, class_weights=class_weights)

    return history, train_loader, test_loader


def step_evaluate(model, test_loader, device):
    """Step 4: Full evaluation with per-class metrics."""
    print("\n" + "─" * 65)
    print("  STEP 4: Evaluation")
    print("─" * 65)

    results = full_evaluation(model, test_loader, device)

    print(f"\n  Overall Accuracy:  {100*results['accuracy']:.2f}%")
    print(f"  Macro F1:          {results['f1_macro']:.4f}")
    print(f"  Weighted F1:       {results['f1_weighted']:.4f}")

    print(f"\n  Classification Report:")
    print(results["report_str"])

    print(f"  Confusion Matrix:")
    cm = results["confusion_matrix"]
    print(f"  {'':>8}", end="")
    for c in AAMI_CLASSES:
        print(f"  {c:>6}", end="")
    print()
    for i in range(NUM_CLASSES):
        print(f"  {AAMI_CLASSES[i]:>8}", end="")
        for j in range(NUM_CLASSES):
            print(f"  {cm[i,j]:>6d}", end="")
        print()

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savetxt(os.path.join(RESULTS_DIR, "confusion_matrix.csv"),
               cm, delimiter=",", fmt="%d", header=",".join(AAMI_CLASSES))

    with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"F1 Macro: {results['f1_macro']:.4f}\n")
        f.write(f"F1 Weighted: {results['f1_weighted']:.4f}\n\n")
        f.write(results["report_str"])

    # Save training history plots if matplotlib available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Loss
        if "train_loss" in step_evaluate.history:
            h = step_evaluate.history
            axes[0].plot(h["train_loss"], label="Train")
            axes[0].plot(h["test_loss"], label="Test")
            axes[0].set_title("Loss")
            axes[0].legend()

            axes[1].plot(h["train_acc"], label="Train")
            axes[1].plot(h["test_acc"], label="Test")
            axes[1].set_title("Accuracy")
            axes[1].legend()

            axes[2].plot(h["test_f1"], label="F1 (macro)")
            axes[2].set_title("Test F1")
            axes[2].legend()

        # Confusion matrix
        fig2, ax2 = plt.subplots(figsize=(7, 6))
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        im = ax2.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax2.set_xticks(range(NUM_CLASSES))
        ax2.set_yticks(range(NUM_CLASSES))
        ax2.set_xticklabels(AAMI_CLASSES)
        ax2.set_yticklabels(AAMI_CLASSES)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("True")
        ax2.set_title("Confusion Matrix (Normalised)")
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax2.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm[i,j]})",
                         ha="center", va="center", color=color, fontsize=9)
        fig2.colorbar(im)

        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150)
        fig2.tight_layout()
        fig2.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
        plt.close("all")
        print(f"\n  Plots saved to {RESULTS_DIR}/")
    except ImportError:
        print("  (matplotlib not available — skipping plots)")

    return results


def step_export(model):
    """Step 5: Export weights to .mem files for Verilog."""
    print("\n" + "─" * 65)
    print("  STEP 5: Export for FPGA")
    print("─" * 65)

    files = export_for_fpga(model)
    print(f"\n  Exported {len(files)} .mem files to {EXPORT_DIR}/:")
    for f in files:
        size = os.path.getsize(f)
        print(f"    {os.path.basename(f):>30s}  {size:>6d} bytes")

    # Also export test vectors for Verilog testbench
    print(f"\n  Exporting test vectors...")
    X_train, y_train, X_test, y_test = load_kaggle_mitbih()
    n_vectors = min(100, len(X_test))
    X_q = quantize_for_fpga(X_test[:n_vectors])

    os.makedirs(EXPORT_DIR, exist_ok=True)
    tv_input = os.path.join(EXPORT_DIR, "test_input.mem")
    tv_label = os.path.join(EXPORT_DIR, "test_labels.mem")

    with open(tv_input, "w") as f:
        for i in range(n_vectors):
            hex_vals = " ".join(f"{(v & 0xFF):02X}" for v in X_q[i].astype(np.int16))
            f.write(hex_vals + "\n")

    with open(tv_label, "w") as f:
        for i in range(n_vectors):
            f.write(f"{y_test[i]}\n")

    print(f"    test_input.mem:  {n_vectors} beats × 187 samples (8-bit hex)")
    print(f"    test_labels.mem: {n_vectors} expected labels")
    print(f"\n  Done. Use $readmemh / $readmemb in Verilog to load these files.")


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="WaveBNN-ECG Pipeline")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Training epochs")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate saved model")
    parser.add_argument("--export-only", action="store_true", help="Only export saved model")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    args = parser.parse_args()

    # Seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_header()
    print(f"  Device: {device}")

    # ── Load model ──
    model = WaveBNN().to(device)

    if args.eval_only or args.export_only:
        ckpt = os.path.join(MODELS_DIR, "wavebnn_best.pth")
        if not os.path.exists(ckpt):
            print(f"\n  ERROR: No saved model at {ckpt}")
            print(f"  Run training first: python main.py --epochs {NUM_EPOCHS}")
            sys.exit(1)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        print(f"  Loaded checkpoint: {ckpt}")

    if args.eval_only:
        X_train, y_train, X_test, y_test = step_load_data()
        train_sub, test_sub = step_wavelet(X_train, X_test)
        _, test_loader = make_dataloaders(train_sub, y_train, test_sub, y_test)
        step_evaluate.history = {}
        step_evaluate(model, test_loader, device)
        return

    if args.export_only:
        step_export(model)
        return

    # ── Full pipeline ──
    X_train, y_train, X_test, y_test = step_load_data()
    train_sub, test_sub = step_wavelet(X_train, X_test)
    history, train_loader, test_loader = step_train(
        model, train_sub, y_train, test_sub, y_test, device, args.epochs
    )
    step_evaluate.history = history
    step_evaluate(model, test_loader, device)
    step_export(model)

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
