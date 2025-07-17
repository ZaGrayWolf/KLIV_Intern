#!/usr/bin/env python3
"""
JetNet Profiling Script:
Calculates Theoretical FLOPs, PTFLOPS, Parameters, Latency, and Activation Size
for the JetNet semantic segmentation model across various input resolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Optional profiling libraries
try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    PTFLOPS_AVAILABLE = False
    print("Warning: 'ptflops' is not installed. PTFLOPS measurements will be skipped.")

# ------------------------------------------------------------------------------
# 1. Model Definition (JetBlock and JetNet)
# ------------------------------------------------------------------------------

class JetBlock(nn.Module):
    """
    The custom residual block for the JetNet model.
    It includes two 3x3 convolutions with a skip connection.
    The first convolution can be dilated.
    """
    def __init__(self, in_channels, out_channels, dilation=1):
        super(JetBlock, self).__init__()
        # The first convolution uses the specified dilation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # The second convolution has a standard dilation of 1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # Element-wise addition for the residual connection
        out += residual
        out = self.relu(out)
        return out

class JetNet(nn.Module):
    """
    The main JetNet model for semantic segmentation.
    An encoder-style network with a final classifier and upsampling.
    """
    def __init__(self, num_classes=21):
        super(JetNet, self).__init__()
        # Initial downsampling layer (stride=2)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Sequence of standard JetBlocks
        self.layer2 = nn.Sequential(
            JetBlock(32, 32),
            JetBlock(32, 32)
        )
        # Downsampling (stride=2) followed by dilated JetBlocks
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            JetBlock(64, 64, dilation=2),
            JetBlock(64, 64, dilation=2)
        )
        # Downsampling (stride=2) followed by more dilated JetBlocks
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            JetBlock(128, 128, dilation=4),
            JetBlock(128, 128, dilation=4)
        )
        # Final 1x1 convolution acts as the classifier
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        # Upsample the output to the original input size (total stride = 2*2*2 = 8)
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
        return x

# ------------------------------------------------------------------------------
# 2. Activation Tracking Utilities
# ------------------------------------------------------------------------------

class ActivationTrackerMixin:
    """A mixin to add activation size tracking to a model."""
    def reset_activation_bytes(self):
        self.activation_bytes = 0

    def add_activation(self, tensor):
        # Accumulate size in bytes (float32 is 4 bytes)
        if hasattr(self, 'activation_bytes'):
            self.activation_bytes += tensor.numel() * 4

class JetNetWithActivation(ActivationTrackerMixin, JetNet):
    """Extends JetNet to track activation sizes by overriding the forward pass."""
    def forward(self, x):
        self.reset_activation_bytes()
        self.add_activation(x)
        
        # Track activations after each major block
        x = self.layer1(x); self.add_activation(x)
        x = self.layer2(x); self.add_activation(x)
        x = self.layer3(x); self.add_activation(x)
        x = self.layer4(x); self.add_activation(x)
        x = self.classifier(x); self.add_activation(x)
        
        # Track final upsampled output
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
        self.add_activation(x)
        return x

# ------------------------------------------------------------------------------
# 3. Profiling Functions
# ------------------------------------------------------------------------------

def _conv_flops(c_in, c_out, h, w, k, stride=1, padding=0, dilation=1):
    """Calculates FLOPs and output shape for a Conv2d layer."""
    h_out = (h + 2 * padding - dilation * (k - 1) - 1) // stride + 1
    w_out = (w + 2 * padding - dilation * (k - 1) - 1) // stride + 1
    # FLOPs = 2 * Cin * Cout * K^2 * Hout * Wout (MACs * 2)
    flops = 2 * c_in * c_out * (k**2) * h_out * w_out
    return flops, h_out, w_out

def _bn_relu_flops(c, h, w):
    """Approximates FLOPs for BatchNorm and ReLU."""
    # BatchNorm: 4 FLOPs per element (2 mul, 2 add)
    # ReLU: 1 FLOP per element
    return c * h * w * 5

def _jetblock_flops(c_in, c_out, h, w, dilation=1):
    """Calculates FLOPs for a JetBlock."""
    # Conv1 + BN + ReLU
    flops1, h1, w1 = _conv_flops(c_in, c_out, h, w, 3, padding=dilation, dilation=dilation)
    flops1 += _bn_relu_flops(c_out, h1, w1)
    # Conv2 + BN
    flops2, h2, w2 = _conv_flops(c_out, c_out, h1, w1, 3, padding=1)
    flops2 += _bn_relu_flops(c_out, h2, w2) - (c_out * h2 * w2) # Subtract ReLU FLOPs
    # Residual add + final ReLU
    add_flops = c_out * h2 * w2
    relu_flops = c_out * h2 * w2
    total = flops1 + flops2 + add_flops + relu_flops
    return total, h2, w2

def calculate_theoretical_flops(input_size, num_classes=21):
    """Calculates the total theoretical FLOPs for the JetNet model."""
    c, h, w = input_size
    h_in, w_in = h, w
    total_flops = 0

    # Layer 1 (Conv + BN + ReLU)
    f, h, w = _conv_flops(c, 32, h, w, 3, stride=2, padding=1)
    total_flops += f + _bn_relu_flops(32, h, w)

    # Layer 2 (2 JetBlocks)
    for _ in range(2):
        f, h, w = _jetblock_flops(32, 32, h, w, dilation=1)
        total_flops += f

    # Layer 3 (Conv + BN + ReLU + 2 JetBlocks)
    f, h, w = _conv_flops(32, 64, h, w, 3, stride=2, padding=1)
    total_flops += f + _bn_relu_flops(64, h, w)
    for _ in range(2):
        f, h, w = _jetblock_flops(64, 64, h, w, dilation=2)
        total_flops += f

    # Layer 4 (Conv + BN + ReLU + 2 JetBlocks)
    f, h, w = _conv_flops(64, 128, h, w, 3, stride=2, padding=1)
    total_flops += f + _bn_relu_flops(128, h, w)
    for _ in range(2):
        f, h, w = _jetblock_flops(128, 128, h, w, dilation=4)
        total_flops += f

    # Classifier (1x1 Conv)
    f, h, w = _conv_flops(128, num_classes, h, w, 1)
    total_flops += f

    # Upsampling (Interpolate)
    # Bilinear interpolation is approx. 7 FLOPs per output element
    upsample_flops = 7 * num_classes * h_in * w_in
    total_flops += upsample_flops

    return total_flops

def count_parameters(model):
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_latency(model, input_size, runs=5, repeats=10):
    """Measures the average inference latency on CPU."""
    device = torch.device('cpu')
    model.to(device).eval()
    x = torch.randn(1, *input_size).to(device)
    
    # Warmup runs to stabilize measurements
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
            
    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            for _ in range(repeats):
                _ = model(x)
            end = time.perf_counter()
            times.append((end - start) * 1000 / repeats) # time in ms
            
    return sum(times) / len(times)

# ------------------------------------------------------------------------------
# 4. Main Execution
# ------------------------------------------------------------------------------

def main():
    """Main function to run the profiling script."""
    INPUT_RESOLUTIONS = [
        (3, 640, 360), (3, 1280, 720), (3, 1360, 760),
        (3, 1600, 900), (3, 1920, 1080), (3, 2048, 1152),
        (3, 2560, 1440), (3, 3840, 2160)
    ]
    
    NUM_CLASSES = 21 # Example: Pascal VOC
    model = JetNetWithActivation(num_classes=NUM_CLASSES)
    
    print("JetNet Profiling")
    print("="*120)
    header = (f"{'Input':>18} | {'Theo FLOPs (G)':>16} | {'PTFLOPS (G)':>12} | "
              f"{'Params (M)':>10} | {'Latency (ms)':>12} | {'Act. Size (MB)':>15}")
    print(header)
    print("-"*120)

    # Calculate parameters once, as they are constant
    params_m = count_parameters(model) / 1e6

    for inp in INPUT_RESOLUTIONS:
        # Theoretical FLOPs
        theo_flops_g = calculate_theoretical_flops(inp, NUM_CLASSES) / 1e9
        
        # PTFLOPS (if available)
        ptflops_g = "N/A"
        if PTFLOPS_AVAILABLE:
            try:
                # ptflops requires a clean model instance without the mixin
                clean_model = JetNet(num_classes=NUM_CLASSES)
                flops, _ = get_model_complexity_info(
                    clean_model, inp, as_strings=False, print_per_layer_stat=False, verbose=False)
                ptflops_g = f"{flops / 1e9:.2f}"
            except Exception as e:
                ptflops_g = "Error"
        
        # Latency
        try:
            latency_ms = measure_latency(model, inp)
        except Exception as e:
            latency_ms = "Error"

        # Activation Size
        try:
            with torch.no_grad():
                _ = model(torch.randn(1, *inp))
            act_size_mb = model.activation_bytes / (1024**2)
        except Exception as e:
            act_size_mb = "Error"
        
        # Print results for the current resolution
        print(f"{str(inp):>18} | {theo_flops_g:16.2f} | {str(ptflops_g):>12} | "
              f"{params_m:10.2f} | {latency_ms:12.2f} | {act_size_mb:15.2f}")

if __name__ == "__main__":
    main()
