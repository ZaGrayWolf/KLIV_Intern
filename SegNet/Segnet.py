#!/usr/bin/env python3
"""
SegNet Profiling Script - Complete Analysis with Activation Memory (Immediate per-input printing)
- Prints latency and activation size immediately after each forward sanity check.
Author: Modified for immediate per-input reporting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import warnings
import sys
warnings.filterwarnings("ignore")

# Optional profiling libraries (not required)
try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except Exception:
    PTFLOPS_AVAILABLE = False

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except Exception:
    THOP_AVAILABLE = False

# --------------------------
# SegNet model definition
# --------------------------
class SegNet(nn.Module):
    def __init__(self, input_channels=3, n_labels=21, kernel_size=3):
        super(SegNet, self).__init__()
        # Encoder
        self.conv1_1 = nn.Conv2d(input_channels, 64, kernel_size, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)

        # Decoder
        self.conv5_3_D = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn5_3_D = nn.BatchNorm2d(512)
        self.conv5_2_D = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn5_2_D = nn.BatchNorm2d(512)
        self.conv5_1_D = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn5_1_D = nn.BatchNorm2d(512)

        self.conv4_3_D = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn4_3_D = nn.BatchNorm2d(512)
        self.conv4_2_D = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn4_2_D = nn.BatchNorm2d(512)
        self.conv4_1_D = nn.Conv2d(512, 256, kernel_size, padding=1)
        self.bn4_1_D = nn.BatchNorm2d(256)

        self.conv3_3_D = nn.Conv2d(256, 256, kernel_size, padding=1)
        self.bn3_3_D = nn.BatchNorm2d(256)
        self.conv3_2_D = nn.Conv2d(256, 256, kernel_size, padding=1)
        self.bn3_2_D = nn.BatchNorm2d(256)
        self.conv3_1_D = nn.Conv2d(256, 128, kernel_size, padding=1)
        self.bn3_1_D = nn.BatchNorm2d(128)

        self.conv2_2_D = nn.Conv2d(128, 128, kernel_size, padding=1)
        self.bn2_2_D = nn.BatchNorm2d(128)
        self.conv2_1_D = nn.Conv2d(128, 64, kernel_size, padding=1)
        self.bn2_1_D = nn.BatchNorm2d(64)

        self.conv1_2_D = nn.Conv2d(64, 64, kernel_size, padding=1)
        self.bn1_2_D = nn.BatchNorm2d(64)

        self.classifier = nn.Conv2d(64, n_labels, 1)

        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)

    def forward(self, x):
        sizes = []
        indices = []

        # Encoder 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        sizes.append(x.size())
        x, idx = self.pool(x)
        indices.append(idx)

        # Encoder 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        sizes.append(x.size())
        x, idx = self.pool(x)
        indices.append(idx)

        # Encoder 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))
        sizes.append(x.size())
        x, idx = self.pool(x)
        indices.append(idx)

        # Encoder 4
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = F.relu(self.bn4_3(self.conv4_3(x)))
        sizes.append(x.size())
        x, idx = self.pool(x)
        indices.append(idx)

        # Encoder 5
        x = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(x)))
        x = F.relu(self.bn5_3(self.conv5_3(x)))
        sizes.append(x.size())
        x, idx = self.pool(x)
        indices.append(idx)

        # Decoder (reverse)
        x = self.unpool(x, indices[4], output_size=sizes[4])
        x = F.relu(self.bn5_3_D(self.conv5_3_D(x)))
        x = F.relu(self.bn5_2_D(self.conv5_2_D(x)))
        x = F.relu(self.bn5_1_D(self.conv5_1_D(x)))

        x = self.unpool(x, indices[3], output_size=sizes[3])
        x = F.relu(self.bn4_3_D(self.conv4_3_D(x)))
        x = F.relu(self.bn4_2_D(self.conv4_2_D(x)))
        x = F.relu(self.bn4_1_D(self.conv4_1_D(x)))

        x = self.unpool(x, indices[2], output_size=sizes[2])
        x = F.relu(self.bn3_3_D(self.conv3_3_D(x)))
        x = F.relu(self.bn3_2_D(self.conv3_2_D(x)))
        x = F.relu(self.bn3_1_D(self.conv3_1_D(x)))

        x = self.unpool(x, indices[1], output_size=sizes[1])
        x = F.relu(self.bn2_2_D(self.conv2_2_D(x)))
        x = F.relu(self.bn2_1_D(self.conv2_1_D(x)))

        x = self.unpool(x, indices[0], output_size=sizes[0])
        x = F.relu(self.bn1_2_D(self.conv1_2_D(x)))

        x = self.classifier(x)
        return x

# --------------------------
# FLOPs / activation memory calculators
# --------------------------
def calculate_conv_flops(input_shape, output_channels, kernel_size, stride=1, padding=0):
    """
    input_shape: tuple (C_in, H_in, W_in)
    kernel_size: int (assumed square kernel)
    Returns (flops, output_shape)
    """
    in_channels, in_height, in_width = input_shape
    out_height = (in_height + 2 * padding - kernel_size) // stride + 1
    out_width = (in_width + 2 * padding - kernel_size) // stride + 1

    flops = (kernel_size * kernel_size * in_channels) * out_height * out_width * output_channels * 2
    return flops, (output_channels, out_height, out_width)

def calculate_segnet_theoretical_flops(input_shape, n_labels=21):
    """Calculate theoretical FLOPs for SegNet"""
    total_flops = 0
    current_shape = input_shape

    encoder_configs = [
        [(64, 3, 1, 1), (64, 3, 1, 1)],
        [(128, 3, 1, 1), (128, 3, 1, 1)],
        [(256, 3, 1, 1), (256, 3, 1, 1), (256, 3, 1, 1)],
        [(512, 3, 1, 1), (512, 3, 1, 1), (512, 3, 1, 1)],
        [(512, 3, 1, 1), (512, 3, 1, 1), (512, 3, 1, 1)]
    ]

    # Encoder
    for block in encoder_configs:
        for out_ch, kernel, stride, padding in block:
            flops, current_shape = calculate_conv_flops(current_shape, out_ch, kernel, stride, padding)
            total_flops += flops
        # after each encoder block, pooling halves spatial dims
        current_shape = (current_shape[0], max(1, current_shape[1] // 2), max(1, current_shape[2] // 2))

    decoder_configs = [
        [(512, 3, 1, 1), (512, 3, 1, 1), (512, 3, 1, 1)],
        [(512, 3, 1, 1), (512, 3, 1, 1), (256, 3, 1, 1)],
        [(256, 3, 1, 1), (256, 3, 1, 1), (128, 3, 1, 1)],
        [(128, 3, 1, 1), (64, 3, 1, 1)],
        [(64, 3, 1, 1)]
    ]

    # Decoder
    for block in decoder_configs:
        # unpooling doubles spatial dims
        current_shape = (current_shape[0], current_shape[1] * 2, current_shape[2] * 2)
        for out_ch, kernel, stride, padding in block:
            flops, current_shape = calculate_conv_flops(current_shape, out_ch, kernel, stride, padding)
            total_flops += flops

    final_flops, _ = calculate_conv_flops(current_shape, n_labels, 1, 1, 0)
    total_flops += final_flops
    return total_flops

def calculate_activation_memory(input_shape):
    """
    Estimate forward pass activation memory for SegNet (float32).
    Returns (total_activation_mb, peak_activation_mb).
    This is an estimator derived from intermediate feature-map sizes
    (not exact runtime allocation).
    """
    total_activation_mb = 0.0
    max_activation_mb = 0.0
    current_shape = input_shape

    # encoder activation channels sequence (with pool decisions)
    encoder_channels = [
        64, 64,   # block1
        128, 128, # block2
        256, 256, 256, # block3
        512, 512, 512, # block4
        512, 512, 512  # block5
    ]

    # Track spatial dims and shrink after blocks similarly to theoretical function
    idx = 0
    for channels in encoder_channels:
        activation_elements = channels * current_shape[1] * current_shape[2]
        activation_mb = (activation_elements * 4) / (1024 ** 2)
        total_activation_mb += activation_mb
        max_activation_mb = max(max_activation_mb, activation_mb)
        current_shape = (channels, current_shape[1], current_shape[2])
        idx += 1
        # after finishing each encoder block (2,2,3,3,3 convs) we shrink dims:
        if idx in (2, 4, 7, 10, 13):
            current_shape = (current_shape[0], max(1, current_shape[1] // 2), max(1, current_shape[2] // 2))

    # decoder channels sequence (symmetric)
    decoder_channels = [
        512, 512, 512,
        512, 512, 256,
        256, 256, 128,
        128, 64,
        64
    ]
    idx = 0
    for channels in decoder_channels:
        # grow spatial dims at same points as decoder blocks
        idx += 1
        if idx in (1, 4, 7, 10):
            current_shape = (current_shape[0], current_shape[1] * 2, current_shape[2] * 2)
        activation_elements = channels * current_shape[1] * current_shape[2]
        activation_mb = (activation_elements * 4) / (1024 ** 2)
        total_activation_mb += activation_mb
        max_activation_mb = max(max_activation_mb, activation_mb)
        current_shape = (channels, current_shape[1], current_shape[2])

    # final classifier activation (n_labels x H x W)
    final_elements = 21 * input_shape[1] * input_shape[2]
    final_mb = (final_elements * 4) / (1024 ** 2)
    total_activation_mb += final_mb
    max_activation_mb = max(max_activation_mb, final_mb)

    return total_activation_mb, max_activation_mb

def get_model_parameters(model):
    return sum(p.numel() for p in model.parameters())

# --------------------------
# Utility to measure latency (CPU) and activation estimator
# --------------------------
def measure_latency_and_activation(model, inp, device=torch.device('cpu'), warmup=2, runs=3, repeat=5):
    model = model.to(device).eval()
    x = torch.randn(1, *inp).to(device)
    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    # measure
    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            for _ in range(repeat):
                _ = model(x)
            end = time.perf_counter()
            times.append((end - start) * 1000.0 / repeat)
    avg_ms = sum(times) / len(times)
    # activation estimate (theoretical estimator)
    total_act_mb, peak_act_mb = calculate_activation_memory(inp)
    return avg_ms, total_act_mb, peak_act_mb

# --------------------------
# THOP measurement helper
# --------------------------
def run_thop_for_input(model_ctor, inp):
    if not THOP_AVAILABLE:
        return None, None
    device = torch.device('cpu')
    model = model_ctor().to(device).eval()
    x = torch.randn(1, *inp).to(device)
    try:
        flops, params = profile(model, inputs=(x,), verbose=False)
        return flops, params
    except Exception:
        return None, None

# --------------------------
# Main script
# --------------------------
def main():
    print("SegNet Profiling Script - Complete Analysis with Activation Memory")
    print("=" * 80)
    print("Note: Running on CPU to avoid device compatibility issues")
    print("=" * 80)

    input_sizes = [
        (3, 640, 360),
        (3, 1280, 720),
        (3, 1360, 760),
        (3, 1600, 900),
        (3, 1920, 1080),
        (3, 2048, 1152),
        (3, 2560, 1440),
        (3, 3840, 2160),
    ]

    # immediate forward-pass test + latency+activation print
    print("\n=== Forward-pass sanity check + Latency + Activation (immediate) ===")
    header = f"{'Input':>16} | {'Status':>6} | {'Out shape':>18} | {'Latency(ms)':>12} | {'TotalAct(MB)':>14} | {'PeakAct(MB)':>12}"
    print(header)
    print("-" * len(header))
    device = torch.device('cpu')

    for inp in input_sizes:
        try:
            model = SegNet(input_channels=3, n_labels=21).to(device).eval()
            with torch.no_grad():
                out = model(torch.randn(1, *inp).to(device))
            # If forward OK, measure latency and estimated activation
            try:
                lat_ms, total_act_mb, peak_act_mb = measure_latency_and_activation(model, inp, device=device)
                status = "OK"
                out_shape = tuple(out.shape)
                lat_str = f"{lat_ms:12.2f}"
                total_act_str = f"{total_act_mb:14.2f}"
                peak_act_str = f"{peak_act_mb:12.2f}"
            except Exception as me:
                status = "OK"
                out_shape = tuple(out.shape)
                lat_str = f"{'ERR':>12}"
                total_act_str = f"{'ERR':>14}"
                peak_act_str = f"{'ERR':>12}"
                print(f"Warning: measurement failed for {inp}: {me}")

            print(f"{str(inp):>16} | {status:>6} | {str(out_shape):>18} | {lat_str} | {total_act_str} | {peak_act_str}")
            sys.stdout.flush()

        except Exception as e:
            # Forward failed (likely OOM or shape error). Print error and continue.
            err_msg = str(e).replace("\n", " ")[:80]
            print(f"{str(inp):>16} | {'ERR':>6} | {err_msg:>18} | {'N/A':>12} | {'N/A':>14} | {'N/A':>12}")
            sys.stdout.flush()
            continue

    print("\nAnalysis complete. Note: Activation sizes are estimators (float32).")

if __name__ == "__main__":
    main()
