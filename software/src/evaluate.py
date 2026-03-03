"""
Evaluation Module
Metrics, plots, and analysis for HDC-AMC performance evaluation.
Generates paper-quality plots for:
  - Accuracy vs SNR curves
  - Accuracy vs Dimension D (Pareto curve)
  - Confusion matrices
  - Per-class accuracy breakdown
  - Resource vs accuracy tradeoff analysis
"""

import numpy as np
import os
from typing import List, Optional, Dict, Tuple
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def compute_accuracy_by_snr(y_true: np.ndarray, y_pred: np.ndarray,
                            snrs: np.ndarray) -> Dict[int, float]:
    """
    Compute classification accuracy for each SNR level.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        snrs: SNR value per example
        
    Returns:
        Dictionary mapping SNR (int) → accuracy (float)
    """
    results = {}
    for snr in sorted(np.unique(snrs)):
        mask = (snrs == snr)
        if mask.sum() > 0:
            acc = accuracy_score(y_true[mask], y_pred[mask])
            results[snr] = acc
    return results


def compute_per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                                mod_names: List[str]) -> Dict[str, float]:
    """
    Compute accuracy for each modulation class.
    
    Args:
        y_true, y_pred: True and predicted labels
        mod_names: Class names
        
    Returns:
        Dictionary mapping class name → accuracy
    """
    results = {}
    for i, name in enumerate(mod_names):
        mask = (y_true == i)
        if mask.sum() > 0:
            acc = accuracy_score(y_true[mask], y_pred[mask])
            results[name] = acc
    return results


def plot_accuracy_vs_snr(snr_acc: Dict[int, float],
                         title: str = "HDC-AMC Accuracy vs SNR",
                         save_path: Optional[str] = None,
                         comparison: Optional[Dict[str, Dict[int, float]]] = None):
    """
    Plot accuracy vs SNR curve (the standard RadioML evaluation plot).
    
    Args:
        snr_acc: Dictionary {snr: accuracy} for main model
        title: Plot title
        save_path: Path to save figure
        comparison: Optional dict of {model_name: {snr: accuracy}} for comparison curves
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    snrs = sorted(snr_acc.keys())
    accs = [snr_acc[s] for s in snrs]
    ax.plot(snrs, accs, 'b-o', linewidth=2, markersize=6, label='HDC (Ours)')

    if comparison:
        colors = ['r', 'g', 'm', 'c', 'orange']
        markers = ['s', '^', 'D', 'v', 'x']
        for i, (name, data) in enumerate(comparison.items()):
            s = sorted(data.keys())
            a = [data[x] for x in s]
            ax.plot(s, a, f'{colors[i % len(colors)]}-{markers[i % len(markers)]}',
                    linewidth=1.5, markersize=5, label=name)

    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Classification Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.set_xlim([min(snrs) - 1, max(snrs) + 1])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: List[str],
                          title: str = "Confusion Matrix",
                          save_path: Optional[str] = None,
                          normalize: bool = True):
    """
    Plot confusion matrix for all modulation classes.
    
    Args:
        y_true, y_pred: True and predicted labels
        class_names: List of class names
        title: Plot title
        save_path: Path to save figure
        normalize: If True, normalize rows to show percentages
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(1, 1, figsize=(max(10, len(class_names) * 0.7),
                                           max(8, len(class_names) * 0.6)))

    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                xticklabels=class_names, yticklabels=class_names,
                cmap='Blues', ax=ax, square=True,
                annot_kws={'size': max(6, 10 - len(class_names) // 4)})

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved: {save_path}")
    plt.close()


def plot_accuracy_vs_dimension(dim_acc: Dict[int, float],
                                title: str = "Accuracy vs Hypervector Dimension D",
                                save_path: Optional[str] = None):
    """
    Plot accuracy vs D (dimension Pareto curve) — key paper figure.
    
    Shows the tradeoff between accuracy and FPGA resource usage.
    
    Args:
        dim_acc: Dictionary {D: accuracy}
        title: Plot title
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt

    dims = sorted(dim_acc.keys())
    accs = [dim_acc[d] for d in dims]

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    ax1.plot(dims, accs, 'b-o', linewidth=2, markersize=8, label='Accuracy')
    ax1.set_xlabel('Hypervector Dimension D', fontsize=12)
    ax1.set_ylabel('Classification Accuracy', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Add estimated FPGA resource annotation
    ax2 = ax1.twinx()
    bram_kb = [d * 11 / 8 / 1024 for d in dims]  # Rough estimate: (Q*2+num_classes)*D bits
    ax2.bar(dims, bram_kb, alpha=0.2, color='orange', width=[d * 0.15 for d in dims],
            label='BRAM Est. (KB)')
    ax2.set_ylabel('Estimated BRAM Usage (KB)', fontsize=12, color='orange')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='center right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved: {save_path}")
    plt.close()


def print_summary(y_true: np.ndarray, y_pred: np.ndarray, snrs: np.ndarray,
                  mod_names: List[str], D: int, Q: int, n_gram: int):
    """
    Print a comprehensive summary of model performance.
    """
    overall_acc = accuracy_score(y_true, y_pred)
    snr_acc = compute_accuracy_by_snr(y_true, y_pred, snrs)
    class_acc = compute_per_class_accuracy(y_true, y_pred, mod_names)

    print("\n" + "=" * 70)
    print("  HDC-AMC CLASSIFICATION RESULTS")
    print("=" * 70)
    print(f"  Hyperparameters: D={D}, Q={Q}, N-gram={n_gram}")
    print(f"  Total test samples: {len(y_true)}")
    print(f"  Overall accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    print("-" * 70)

    print("\n  Accuracy by SNR (dB):")
    for snr in sorted(snr_acc.keys()):
        bar = "█" * int(snr_acc[snr] * 40)
        print(f"    {snr:+3d} dB: {snr_acc[snr]:.4f}  {bar}")

    print(f"\n  Accuracy by Modulation Class:")
    for name, acc in sorted(class_acc.items(), key=lambda x: x[1]):
        bar = "█" * int(acc * 30)
        print(f"    {name:>12s}: {acc:.4f}  {bar}")

    # FPGA hardware estimates
    num_classes = len(mod_names)
    codebook_bits = 2 * Q * D
    prototype_bits = num_classes * D
    total_bits = codebook_bits + prototype_bits
    total_kb = total_bits / 8 / 1024

    print(f"\n  FPGA Resource Estimates:")
    print(f"    Codebook memory: {2*Q} × {D} bits = {codebook_bits/8/1024:.1f} KB")
    print(f"    Prototype memory: {num_classes} × {D} bits = {prototype_bits/8/1024:.1f} KB")
    print(f"    Total model: {total_kb:.1f} KB")
    print(f"    DSP blocks: 0 (all computation is XOR + popcount)")
    print(f"    Estimated classification latency: {D//64 * num_classes} cycles "
          f"@ 100MHz = {D//64 * num_classes * 10:.0f} ns")
    print("=" * 70)

    return overall_acc, snr_acc, class_acc


def generate_full_report(y_true: np.ndarray, y_pred: np.ndarray,
                         snrs: np.ndarray, mod_names: List[str],
                         D: int, Q: int, n_gram: int,
                         results_dir: str):
    """
    Generate all evaluation plots and save to results directory.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Text summary
    overall_acc, snr_acc, class_acc = print_summary(
        y_true, y_pred, snrs, mod_names, D, Q, n_gram
    )

    # Accuracy vs SNR plot
    plot_accuracy_vs_snr(
        snr_acc,
        title=f"HDC-AMC Accuracy vs SNR (D={D}, Q={Q}, N-gram={n_gram})",
        save_path=os.path.join(results_dir, f"accuracy_vs_snr_D{D}_Q{Q}_N{n_gram}.png")
    )

    # Confusion matrix (all SNRs)
    plot_confusion_matrix(
        y_true, y_pred, mod_names,
        title=f"Confusion Matrix — All SNRs (D={D})",
        save_path=os.path.join(results_dir, f"confusion_all_D{D}.png")
    )

    # Confusion matrix (high SNR only, ≥ 10 dB)
    high_snr_mask = snrs >= 10
    if high_snr_mask.sum() > 0:
        plot_confusion_matrix(
            y_true[high_snr_mask], y_pred[high_snr_mask], mod_names,
            title=f"Confusion Matrix — SNR ≥ 10 dB (D={D})",
            save_path=os.path.join(results_dir, f"confusion_highsnr_D{D}.png")
        )

    # Save raw results
    np.savez(os.path.join(results_dir, f"results_D{D}_Q{Q}_N{n_gram}.npz"),
             y_true=y_true, y_pred=y_pred, snrs=snrs,
             overall_acc=overall_acc,
             snr_keys=list(snr_acc.keys()),
             snr_vals=list(snr_acc.values()))

    print(f"\n[Report] All results saved to {results_dir}/")
    return overall_acc
