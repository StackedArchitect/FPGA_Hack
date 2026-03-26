"""WaveBNN-ECG: 4-branch parallel BNN with STE binarization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .config import (BRANCH_CONFIGS, CONCAT_BITS, FC1_OUT, FC2_OUT, NUM_CLASSES)


# Binary primitives
class SignSTE(torch.autograd.Function):
    """Sign binarisation with Straight-Through Estimator."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # STE: pass gradient through where |x| <= 1
        return grad_output * (x.abs() <= 1).float()


def sign_ste(x):
    return SignSTE.apply(x)


class BinaryConv1d(nn.Module):
    """Conv1d with binary weights during forward pass (real weights stored for update)."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        # Kaiming init scaled for binary
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='linear')

    def forward(self, x):
        # Binarise weights: w_b = sign(w)
        w_bin = sign_ste(self.conv.weight)
        return F.conv1d(x, w_bin, self.conv.bias,
                        self.conv.stride, self.conv.padding)


class BinaryLinear(nn.Module):
    """Linear layer with binary weights during forward pass."""
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='linear')

    def forward(self, x):
        w_bin = sign_ste(self.linear.weight)
        return F.linear(x, w_bin, self.linear.bias)


# Branch module (one per sub-band)
class BNNBranch(nn.Module):
    """Single sub-band branch: BinConv1d → BN → Sign → MaxPool → Flatten."""
    def __init__(self, in_len, out_channels, kernel_size, pool_size):
        super().__init__()
        self.conv = BinaryConv1d(1, out_channels, kernel_size)
        self.bn   = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pool_size)

        # Compute output size
        conv_out = in_len - kernel_size + 1
        pool_out = conv_out // pool_size
        self.flat_size = pool_out * out_channels

    def forward(self, x):
        # x: (B, in_len) → (B, 1, in_len)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.bn(x)
        x = sign_ste(x)        # Binary activation → +1/-1
        x = self.pool(x)
        return x.view(x.size(0), -1)   # (B, flat_size)


# Full WaveBNN model
class WaveBNN(nn.Module):
    """
    4-branch parallel Binary Neural Network for ECG arrhythmia classification.

    Input:  dict of sub-band tensors {"cA3": (B,24), "cD3": (B,24),
                                       "cD2": (B,47), "cD1": (B,94)}
    Output: (B, 5) logits
    """
    def __init__(self):
        super().__init__()

        # Create branches from config
        self.branch_names = ["cA3", "cD3", "cD2", "cD1"]
        self.branches = nn.ModuleDict()
        total_bits = 0
        for name in self.branch_names:
            cfg = BRANCH_CONFIGS[name]
            branch = BNNBranch(cfg["in_len"], cfg["out_ch"], cfg["kernel"], cfg["pool"])
            self.branches[name] = branch
            total_bits += branch.flat_size

        assert total_bits == CONCAT_BITS, \
            f"Concatenated bits {total_bits} != expected {CONCAT_BITS}"

        # FC head
        self.fc1 = BinaryLinear(CONCAT_BITS, FC1_OUT)
        self.bn1 = nn.BatchNorm1d(FC1_OUT)
        self.fc2 = nn.Linear(FC1_OUT, FC2_OUT)   # Full-precision output layer

    def forward(self, subbands):
        """
        Args:
            subbands: dict {"cA3": (B,24), "cD3": (B,24), "cD2": (B,47), "cD1": (B,94)}
                      OR tuple/list in order [cA3, cD3, cD2, cD1]
        """
        if isinstance(subbands, (list, tuple)):
            parts = [self.branches[name](subbands[i])
                     for i, name in enumerate(self.branch_names)]
        else:
            parts = [self.branches[name](subbands[name])
                     for name in self.branch_names]

        x = torch.cat(parts, dim=1)        # (B, 1024)
        x = self.fc1(x)                     # BinaryLinear
        x = self.bn1(x)
        x = sign_ste(x)                     # Binary activation
        x = self.fc2(x)                     # Full-precision logits
        return x

    def count_binary_ops(self):
        """Count binary operations (XNOR+popcount) — no real multiplies."""
        ops = {}
        for name in self.branch_names:
            cfg = BRANCH_CONFIGS[name]
            conv_ops = cfg["out_ch"] * (cfg["in_len"] - cfg["kernel"] + 1) * cfg["kernel"]
            ops[f"branch_{name}"] = conv_ops
        ops["fc1"] = CONCAT_BITS * FC1_OUT
        ops["fc2_real_macs"] = FC1_OUT * FC2_OUT   # only real multiplies
        ops["total_binary"] = sum(v for k, v in ops.items() if "real" not in k)
        return ops

    def export_params(self):
        """Export binary weights and BN parameters for FPGA .mem files."""
        params = {}
        for name in self.branch_names:
            branch = self.branches[name]
            # Binary weights: sign(w) → 0/1 encoding (1 for +1, 0 for -1)
            w = branch.conv.conv.weight.data.sign()
            params[f"{name}_conv_weights"] = (w > 0).byte()
            # Batch norm parameters → threshold for FPGA comparator
            bn = branch.bn
            params[f"{name}_bn_weight"] = bn.weight.data.clone()
            params[f"{name}_bn_bias"]   = bn.bias.data.clone()
            params[f"{name}_bn_mean"]   = bn.running_mean.data.clone()
            params[f"{name}_bn_var"]    = bn.running_var.data.clone()

        # FC1 binary weights
        w1 = self.fc1.linear.weight.data.sign()
        params["fc1_weights"] = (w1 > 0).byte()
        params["fc1_bn_weight"] = self.bn1.weight.data.clone()
        params["fc1_bn_bias"]   = self.bn1.bias.data.clone()
        params["fc1_bn_mean"]   = self.bn1.running_mean.data.clone()
        params["fc1_bn_var"]    = self.bn1.running_var.data.clone()

        # FC2 full-precision weights (quantised to int for FPGA)
        params["fc2_weights"] = self.fc2.weight.data.clone()
        params["fc2_bias"]    = self.fc2.bias.data.clone()

        return params
