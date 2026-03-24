"""
WaveBNN-ECG: Training & Evaluation Utilities
=============================================
Dataset wrapper, training loop, evaluation metrics, and FPGA export.
"""

import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)

from .config import (
    AAMI_CLASSES, NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    WEIGHT_DECAY, RANDOM_SEED, USE_CLASS_WEIGHTS, MODELS_DIR, EXPORT_DIR,
    SUBBAND_LENGTHS, CONCAT_BITS, FC1_OUT,
)


# ────────────────────────────────────────────────────────────
# Focal Loss
# ────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal Loss — handles class imbalance better than weighted CE."""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


# ────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────
class ECGSubbandDataset(Dataset):
    """Wraps wavelet sub-band arrays + labels for PyTorch DataLoader."""
    def __init__(self, subbands_dict, labels, augment=False):
        self.cA3 = torch.tensor(subbands_dict["cA3"], dtype=torch.float32)
        self.cD3 = torch.tensor(subbands_dict["cD3"], dtype=torch.float32)
        self.cD2 = torch.tensor(subbands_dict["cD2"], dtype=torch.float32)
        self.cD1 = torch.tensor(subbands_dict["cD1"], dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        subbands = {
            "cA3": self.cA3[idx].clone(),
            "cD3": self.cD3[idx].clone(),
            "cD2": self.cD2[idx].clone(),
            "cD1": self.cD1[idx].clone(),
        }
        if self.augment:
            # Gaussian noise (50% chance)
            if torch.rand(1).item() < 0.5:
                for k in subbands:
                    subbands[k] = subbands[k] + torch.randn_like(subbands[k]) * 0.02
            # Amplitude scaling [0.9, 1.1] (50% chance)
            if torch.rand(1).item() < 0.5:
                scale = 0.9 + torch.rand(1).item() * 0.2
                for k in subbands:
                    subbands[k] = subbands[k] * scale
        return subbands, self.labels[idx]


def make_dataloaders(train_subbands, y_train, test_subbands, y_test,
                     batch_size=BATCH_SIZE, num_workers=2, augment=False):
    """Create train and test DataLoaders."""
    train_ds = ECGSubbandDataset(train_subbands, y_train, augment=augment)
    test_ds  = ECGSubbandDataset(test_subbands, y_test, augment=False)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    return train_loader, test_loader


# ────────────────────────────────────────────────────────────
# Class weights (inverse frequency)
# ────────────────────────────────────────────────────────────
def compute_class_weights(labels):
    """Compute sqrt-inverse-frequency class weights, capped at 5.0."""
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    raw_weights = len(labels) / (NUM_CLASSES * counts)
    weights = np.minimum(np.sqrt(raw_weights), 5.0)
    return torch.tensor(weights, dtype=torch.float32)


# ────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for subbands, labels in loader:
        subbands = {k: v.to(device) for k, v in subbands.items()}
        labels = labels.to(device)

        logits = model(subbands)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for subbands, labels in loader:
        subbands = {k: v.to(device) for k, v in subbands.items()}
        labels = labels.to(device)

        logits = model(subbands)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_labels))


def train_model(model, train_loader, test_loader, device, num_epochs=NUM_EPOCHS,
                lr=LEARNING_RATE, class_weights=None):
    """Full training loop with warmup, cosine annealing, and best-model checkpointing."""

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    # Linear warmup (5 epochs) + cosine annealing
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "test_f1": []}
    best_f1, best_epoch = 0.0, 0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
        f1 = f1_score(labels, preds, average="macro")

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["test_f1"].append(f1)

        elapsed = time.time() - t0

        # Checkpoint best
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            os.makedirs(MODELS_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "wavebnn_best.pth"))

        marker = '  ★ best' if f1 >= best_f1 and epoch == best_epoch else ''
        if epoch % 5 == 0 or epoch == 1 or marker:
            print(f"Epoch {epoch:3d}/{num_epochs} │ "
                  f"Train {train_loss:.4f} / {100*train_acc:.1f}% │ "
                  f"Test {test_loss:.4f} / {100*test_acc:.1f}% │ "
                  f"F1 {f1:.4f} │ {elapsed:.1f}s{marker}")

    print(f"\nBest test F1: {best_f1:.4f} at epoch {best_epoch}")
    # Reload best weights
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "wavebnn_best.pth"),
                                     weights_only=True))
    return history


# ────────────────────────────────────────────────────────────
# Detailed evaluation
# ────────────────────────────────────────────────────────────
def full_evaluation(model, test_loader, device):
    """Run evaluation and return report + confusion matrix."""
    criterion = nn.CrossEntropyLoss()
    _, acc, preds, labels = evaluate(model, test_loader, criterion, device)

    report = classification_report(
        labels, preds,
        target_names=AAMI_CLASSES,
        digits=4,
        output_dict=True
    )
    report_str = classification_report(
        labels, preds,
        target_names=AAMI_CLASSES,
        digits=4,
    )
    cm = confusion_matrix(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report": report,
        "report_str": report_str,
        "confusion_matrix": cm,
        "predictions": preds,
        "labels": labels,
    }


# ────────────────────────────────────────────────────────────
# FPGA Export — .mem files for Verilog $readmemb / $readmemh
# ────────────────────────────────────────────────────────────
def _bn_to_threshold(bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """
    Fuse BatchNorm after binary layer into a threshold.

    For binary input x_bin ∈ {-1, +1}:
        BN(x) = gamma * (x - mu) / sqrt(var + eps) + beta
        sign(BN(x)) > 0  ⟺  x > mu - beta * sqrt(var+eps) / gamma

    Returns threshold (float) that can be compared against popcount.
    """
    std = torch.sqrt(bn_var + eps)
    threshold = bn_mean - bn_bias * std / bn_weight
    return threshold


def _float_to_fixed(tensor, frac_bits=8, total_bits=16):
    """Convert float tensor to signed fixed-point integer."""
    scale = 2 ** frac_bits
    clamp_max = (2 ** (total_bits - 1)) - 1
    clamp_min = -(2 ** (total_bits - 1))
    return torch.clamp(torch.round(tensor * scale), clamp_min, clamp_max).to(torch.int32)


def export_for_fpga(model, export_dir=None):
    """
    Export model parameters as .mem files for Verilog synthesis.

    Handles BN gamma sign: if gamma < 0 for a channel, flips the binary
    weights and negates the threshold so hardware always uses '>' comparison.

    Files produced:
      - {branch}_conv_weights.mem   — binary (gamma-adjusted)
      - {branch}_bn_threshold.mem   — hex, 16-bit signed integer thresholds
      - fc1_weights.mem             — binary (gamma-adjusted)
      - fc1_bn_threshold.mem        — hex, 16-bit popcount-domain thresholds
      - fc2_weights.mem             — hex, 16-bit fixed-point (8 frac bits)
      - fc2_bias.mem                — hex, 16-bit fixed-point (8 frac bits)
    """
    if export_dir is None:
        export_dir = EXPORT_DIR
    os.makedirs(export_dir, exist_ok=True)

    model.eval()
    params = model.export_params()
    files_written = []

    # ── Branch conv weights + integer thresholds ──
    for name in ["cA3", "cD3", "cD2", "cD1"]:
        w = params[f"{name}_conv_weights"].clone()  # (out_ch, 1, kernel) uint8 0/1
        bn_w = params[f"{name}_bn_weight"]
        bn_b = params[f"{name}_bn_bias"]
        bn_m = params[f"{name}_bn_mean"]
        bn_v = params[f"{name}_bn_var"]
        thresh = _bn_to_threshold(bn_w, bn_b, bn_m, bn_v)

        # Gamma sign: flip weights + negate threshold for gamma < 0 channels
        for ch in range(w.shape[0]):
            if bn_w[ch] < 0:
                w[ch] = 1 - w[ch]
                thresh[ch] = -thresh[ch]

        # Conv weights (binary: one line per channel)
        filepath = os.path.join(export_dir, f"{name}_conv_weights.mem")
        with open(filepath, "w") as f:
            for ch in range(w.shape[0]):
                bits = "".join(str(int(b)) for b in w[ch].flatten())
                f.write(bits + "\n")
        files_written.append(filepath)

        # BN threshold as floor(float) → 16-bit signed integer
        thresh_int = torch.floor(thresh).to(torch.int32)
        thresh_int = torch.clamp(thresh_int, -32768, 32767)
        filepath = os.path.join(export_dir, f"{name}_bn_threshold.mem")
        with open(filepath, "w") as f:
            for val in thresh_int:
                f.write(f"{val.item() & 0xFFFF:04X}\n")
        files_written.append(filepath)

    # ── FC1 weights + popcount-domain thresholds ──
    w1 = params["fc1_weights"].clone()  # (128, 2048) uint8 0/1
    fc1_bn_w = params["fc1_bn_weight"]
    fc1_bn_b = params["fc1_bn_bias"]
    fc1_bn_m = params["fc1_bn_mean"]
    fc1_bn_v = params["fc1_bn_var"]
    thresh1 = _bn_to_threshold(fc1_bn_w, fc1_bn_b, fc1_bn_m, fc1_bn_v)

    for ch in range(w1.shape[0]):
        if fc1_bn_w[ch] < 0:
            w1[ch] = 1 - w1[ch]
            thresh1[ch] = -thresh1[ch]

    # Convert to popcount domain: y = 2*P - N => P > (thresh + N) / 2
    N_fc1 = w1.shape[1]  # 2048
    thresh1_pop = torch.floor((thresh1 + N_fc1) / 2.0).to(torch.int32)
    thresh1_pop = torch.clamp(thresh1_pop, 0, N_fc1)

    filepath = os.path.join(export_dir, "fc1_weights.mem")
    with open(filepath, "w") as f:
        for row in range(w1.shape[0]):
            bits = "".join(str(int(b)) for b in w1[row])
            f.write(bits + "\n")
    files_written.append(filepath)

    filepath = os.path.join(export_dir, "fc1_bn_threshold.mem")
    with open(filepath, "w") as f:
        for val in thresh1_pop:
            f.write(f"{val.item() & 0xFFF:03X}\n")  # 12-bit (0..2048)
    files_written.append(filepath)

    # ── FC2 weights + bias (16-bit fixed-point, 8 fractional bits) ──
    w2 = _float_to_fixed(params["fc2_weights"])  # (5, 128)
    filepath = os.path.join(export_dir, "fc2_weights.mem")
    with open(filepath, "w") as f:
        for row in range(w2.shape[0]):
            for val in w2[row]:
                f.write(f"{val.item() & 0xFFFF:04X}\n")
    files_written.append(filepath)

    b2 = _float_to_fixed(params["fc2_bias"])
    filepath = os.path.join(export_dir, "fc2_bias.mem")
    with open(filepath, "w") as f:
        for val in b2:
            f.write(f"{val.item() & 0xFFFF:04X}\n")
    files_written.append(filepath)

    return files_written
