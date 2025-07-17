#!/usr/bin/env python3
"""
LinkNet Profiling Script (CORRECTED):
Calculates Theoretical FLOPs, PTFLOPS, Parameters, Latency, and Activation Size
for the LinkNet semantic segmentation model across various input resolutions.
FIXES: Dimension mismatch issues in decoder skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
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
# 1. Model Definition (DecoderBlock and LinkNet)
# ------------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """
    Decoder block for LinkNet with adaptive upsampling.
    This block takes in features from the previous decoder block and upsamples them
    to match the spatial dimensions of the corresponding encoder skip connection.
    """
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        # 1x1 convolution to reduce channel dimensions
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 3x3 transposed convolution for upsampling
        self.deconv = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 
                                         kernel_size=3, stride=2, padding=1, 
                                         output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)
        
        # 1x1 convolution to expand channels to match encoder output
        self.conv3 = nn.Conv2d(in_channels // 4, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x, target_size=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.deconv(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        # If target_size is provided, use interpolation to match exact dimensions
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

class LinkNet(nn.Module):
    """
    The LinkNet architecture for semantic segmentation, built upon a ResNet-18 encoder.
    Fixed to handle arbitrary input dimensions by using adaptive upsampling.
    """
    def __init__(self, num_classes=21, pretrained=True):
        super(LinkNet, self).__init__()
        
        # Load a pre-trained ResNet-18 model
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet18(weights=weights)
        
        # ----------------- Encoder -----------------
        # The initial layers of ResNet-18
        self.encoder_conv1 = resnet.conv1
        self.encoder_bn1 = resnet.bn1
        self.encoder_relu = resnet.relu
        self.encoder_maxpool = resnet.maxpool
        
        # The four main blocks of the ResNet encoder
        self.encoder_layer1 = resnet.layer1  # Output channels: 64
        self.encoder_layer2 = resnet.layer2  # Output channels: 128
        self.encoder_layer3 = resnet.layer3  # Output channels: 256
        self.encoder_layer4 = resnet.layer4  # Output channels: 512
        
        # ----------------- Decoder -----------------
        # Decoder blocks corresponding to each encoder layer
        self.decoder_layer4 = DecoderBlock(512, 256)
        self.decoder_layer3 = DecoderBlock(256, 128)
        self.decoder_layer2 = DecoderBlock(128, 64)
        self.decoder_layer1 = DecoderBlock(64, 64)
        
        # ----------------- Final Upsampling Layers -----------------
        # These layers bring the feature map back to the original image size
        self.final_deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.final_bn1 = nn.BatchNorm2d(32)
        self.final_relu1 = nn.ReLU(inplace=True)
        
        self.final_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.final_bn2 = nn.BatchNorm2d(32)
        self.final_relu2 = nn.ReLU(inplace=True)
        
        # Final transposed convolution to produce the output segmentation map
        self.final_deconv2 = nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2, bias=False)

    def forward(self, x):
        # Store original input size for final upsampling
        input_size = x.shape[2:]
        
        # ----------------- Encoder Path -----------------
        # Initial block
        e0 = self.encoder_conv1(x)
        e0 = self.encoder_bn1(e0)
        e0 = self.encoder_relu(e0)
        e0_pool = self.encoder_maxpool(e0)
        
        # Encoder layers, saving the output of each for skip connections
        e1 = self.encoder_layer1(e0_pool)
        e2 = self.encoder_layer2(e1)
        e3 = self.encoder_layer3(e2)
        e4 = self.encoder_layer4(e3) # This is the bottleneck
        
        # ----------------- Decoder Path with Adaptive Skip Connections -----------------
        # Pass target sizes to ensure exact dimension matching
        d4 = self.decoder_layer4(e4, target_size=e3.shape[2:]) + e3
        d3 = self.decoder_layer3(d4, target_size=e2.shape[2:]) + e2
        d2 = self.decoder_layer2(d3, target_size=e1.shape[2:]) + e1
        d1 = self.decoder_layer1(d2, target_size=e0.shape[2:]) + e0
        
        # ----------------- Final Upsampling -----------------
        f1 = self.final_deconv1(d1)
        f1 = self.final_bn1(f1)
        f1 = self.final_relu1(f1)
        
        f2 = self.final_conv(f1)
        f2 = self.final_bn2(f2)
        f2 = self.final_relu2(f2)
        
        # Final output - use interpolation to match exact input size
        out = self.final_deconv2(f2)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        
        return out

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

class LinkNetWithActivation(ActivationTrackerMixin, LinkNet):
    """Extends LinkNet to track activation sizes by overriding the forward pass."""
    def forward(self, x):
        self.reset_activation_bytes()
        self.add_activation(x)
        
        # Store original input size for final upsampling
        input_size = x.shape[2:]
        
        # ----------------- Encoder Path -----------------
        # Initial block
        e0 = self.encoder_conv1(x)
        e0 = self.encoder_bn1(e0)
        e0 = self.encoder_relu(e0)
        self.add_activation(e0)
        
        e0_pool = self.encoder_maxpool(e0)
        self.add_activation(e0_pool)
        
        # Encoder layers, saving the output of each for skip connections
        e1 = self.encoder_layer1(e0_pool)
        self.add_activation(e1)
        
        e2 = self.encoder_layer2(e1)
        self.add_activation(e2)
        
        e3 = self.encoder_layer3(e2)
        self.add_activation(e3)
        
        e4 = self.encoder_layer4(e3) # This is the bottleneck
        self.add_activation(e4)
        
        # ----------------- Decoder Path with Adaptive Skip Connections -----------------
        d4 = self.decoder_layer4(e4, target_size=e3.shape[2:]) + e3
        self.add_activation(d4)
        
        d3 = self.decoder_layer3(d4, target_size=e2.shape[2:]) + e2
        self.add_activation(d3)
        
        d2 = self.decoder_layer2(d3, target_size=e1.shape[2:]) + e1
        self.add_activation(d2)
        
        d1 = self.decoder_layer1(d2, target_size=e0.shape[2:]) + e0
        self.add_activation(d1)
        
        # ----------------- Final Upsampling -----------------
        f1 = self.final_deconv1(d1)
        f1 = self.final_bn1(f1)
        f1 = self.final_relu1(f1)
        self.add_activation(f1)
        
        f2 = self.final_conv(f1)
        f2 = self.final_bn2(f2)
        f2 = self.final_relu2(f2)
        self.add_activation(f2)
        
        # Final output - use interpolation to match exact input size
        out = self.final_deconv2(f2)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        self.add_activation(out)
        
        return out

# ------------------------------------------------------------------------------
# 3. Profiling Functions
# ------------------------------------------------------------------------------

def _conv_flops(c_in, c_out, h, w, k, stride=1, padding=0):
    """Calculates FLOPs and output shape for a Conv2d layer."""
    h_out = (h + 2 * padding - k) // stride + 1
    w_out = (w + 2 * padding - k) // stride + 1
    # FLOPs = 2 * Cin * Cout * K^2 * Hout * Wout (MACs * 2)
    flops = 2 * c_in * c_out * (k**2) * h_out * w_out
    return flops, h_out, w_out

def _deconv_flops(c_in, c_out, h, w, k, stride=2, padding=1, output_padding=1):
    """Calculates FLOPs and output shape for a ConvTranspose2d layer."""
    h_out = (h - 1) * stride - 2 * padding + k + output_padding
    w_out = (w - 1) * stride - 2 * padding + k + output_padding
    # FLOPs for deconv is similar to conv
    flops = 2 * c_in * c_out * (k**2) * h_out * w_out
    return flops, h_out, w_out

def _bn_relu_flops(c, h, w):
    """Approximates FLOPs for BatchNorm and ReLU."""
    # BatchNorm: 4 FLOPs per element (2 mul, 2 add)
    # ReLU: 1 FLOP per element
    return c * h * w * 5

def _maxpool_flops(h, w, k=3, stride=2, padding=1):
    """Calculates output shape for MaxPool2d (no FLOPs for comparison)."""
    h_out = (h + 2 * padding - k) // stride + 1
    w_out = (w + 2 * padding - k) // stride + 1
    return h_out, w_out

def _interpolate_flops(c, h_in, w_in, h_out, w_out):
    """Approximates FLOPs for bilinear interpolation."""
    # Bilinear interpolation: ~7 FLOPs per output element
    return 7 * c * h_out * w_out

def _decoder_block_flops(c_in, c_out, h, w, target_h, target_w):
    """Calculates FLOPs for a DecoderBlock with adaptive upsampling."""
    total_flops = 0
    
    # First 1x1 conv
    flops, h1, w1 = _conv_flops(c_in, c_in // 4, h, w, 1)
    total_flops += flops + _bn_relu_flops(c_in // 4, h1, w1)
    
    # Transposed convolution
    flops, h2, w2 = _deconv_flops(c_in // 4, c_in // 4, h1, w1, 3)
    total_flops += flops + _bn_relu_flops(c_in // 4, h2, w2)
    
    # Adaptive interpolation to target size
    if (h2, w2) != (target_h, target_w):
        total_flops += _interpolate_flops(c_in // 4, h2, w2, target_h, target_w)
    
    # Second 1x1 conv
    flops, h3, w3 = _conv_flops(c_in // 4, c_out, target_h, target_w, 1)
    total_flops += flops + _bn_relu_flops(c_out, h3, w3)
    
    return total_flops, target_h, target_w

def calculate_theoretical_flops(input_size, num_classes=19):
    """Calculates the total theoretical FLOPs for the LinkNet model."""
    c, h, w = input_size
    h_orig, w_orig = h, w
    total_flops = 0

    # Initial encoder conv (7x7, stride=2)
    flops, h, w = _conv_flops(c, 64, h, w, 7, stride=2, padding=3)
    total_flops += flops + _bn_relu_flops(64, h, w)
    h_e0, w_e0 = h, w
    
    # MaxPooling (3x3, stride=2)
    h, w = _maxpool_flops(h, w, 3, 2, 1)
    
    # ResNet-18 encoder layers (approximation based on standard ResNet-18 FLOPs)
    # Layer 1: 2 blocks, 64 channels
    for _ in range(2):
        flops, h, w = _conv_flops(64, 64, h, w, 3, padding=1)
        total_flops += flops + _bn_relu_flops(64, h, w)
        flops, h, w = _conv_flops(64, 64, h, w, 3, padding=1)
        total_flops += flops + _bn_relu_flops(64, h, w)
        # Skip connection addition
        total_flops += 64 * h * w
    h_e1, w_e1 = h, w
    
    # Layer 2: 2 blocks, 128 channels, first block has stride=2
    flops, h, w = _conv_flops(64, 128, h, w, 3, stride=2, padding=1)
    total_flops += flops + _bn_relu_flops(128, h, w)
    flops, h, w = _conv_flops(128, 128, h, w, 3, padding=1)
    total_flops += flops + _bn_relu_flops(128, h, w)
    # Downsample shortcut
    flops_ds, _, _ = _conv_flops(64, 128, h*2, w*2, 1, stride=2)
    total_flops += flops_ds + _bn_relu_flops(128, h, w)
    total_flops += 128 * h * w  # Skip addition
    
    # Second block of layer 2
    flops, h, w = _conv_flops(128, 128, h, w, 3, padding=1)
    total_flops += flops + _bn_relu_flops(128, h, w)
    flops, h, w = _conv_flops(128, 128, h, w, 3, padding=1)
    total_flops += flops + _bn_relu_flops(128, h, w)
    total_flops += 128 * h * w  # Skip addition
    h_e2, w_e2 = h, w
    
    # Layer 3: 2 blocks, 256 channels, first block has stride=2
    flops, h, w = _conv_flops(128, 256, h, w, 3, stride=2, padding=1)
    total_flops += flops + _bn_relu_flops(256, h, w)
    flops, h, w = _conv_flops(256, 256, h, w, 3, padding=1)
    total_flops += flops + _bn_relu_flops(256, h, w)
    # Downsample shortcut
    flops_ds, _, _ = _conv_flops(128, 256, h*2, w*2, 1, stride=2)
    total_flops += flops_ds + _bn_relu_flops(256, h, w)
    total_flops += 256 * h * w  # Skip addition
    
    # Second block of layer 3
    flops, h, w = _conv_flops(256, 256, h, w, 3, padding=1)
    total_flops += flops + _bn_relu_flops(256, h, w)
    flops, h, w = _conv_flops(256, 256, h, w, 3, padding=1)
    total_flops += flops + _bn_relu_flops(256, h, w)
    total_flops += 256 * h * w  # Skip addition
    h_e3, w_e3 = h, w
    
    # Layer 4: 2 blocks, 512 channels, first block has stride=2
    flops, h, w = _conv_flops(256, 512, h, w, 3, stride=2, padding=1)
    total_flops += flops + _bn_relu_flops(512, h, w)
    flops, h, w = _conv_flops(512, 512, h, w, 3, padding=1)
    total_flops += flops + _bn_relu_flops(512, h, w)
    # Downsample shortcut
    flops_ds, _, _ = _conv_flops(256, 512, h*2, w*2, 1, stride=2)
    total_flops += flops_ds + _bn_relu_flops(512, h, w)
    total_flops += 512 * h * w  # Skip addition
    
    # Second block of layer 4
    flops, h, w = _conv_flops(512, 512, h, w, 3, padding=1)
    total_flops += flops + _bn_relu_flops(512, h, w)
    flops, h, w = _conv_flops(512, 512, h, w, 3, padding=1)
    total_flops += flops + _bn_relu_flops(512, h, w)
    total_flops += 512 * h * w  # Skip addition
    
    # Decoder blocks with adaptive upsampling
    # Decoder 4: 512 -> 256
    flops, h, w = _decoder_block_flops(512, 256, h, w, h_e3, w_e3)
    total_flops += flops + 256 * h_e3 * w_e3  # Skip addition
    
    # Decoder 3: 256 -> 128
    flops, h, w = _decoder_block_flops(256, 128, h_e3, w_e3, h_e2, w_e2)
    total_flops += flops + 128 * h_e2 * w_e2  # Skip addition
    
    # Decoder 2: 128 -> 64
    flops, h, w = _decoder_block_flops(128, 64, h_e2, w_e2, h_e1, w_e1)
    total_flops += flops + 64 * h_e1 * w_e1  # Skip addition
    
    # Decoder 1: 64 -> 64
    flops, h, w = _decoder_block_flops(64, 64, h_e1, w_e1, h_e0, w_e0)
    total_flops += flops + 64 * h_e0 * w_e0  # Skip addition
    
    # Final upsampling layers
    # Final deconv 1
    flops, h, w = _deconv_flops(64, 32, h_e0, w_e0, 3)
    total_flops += flops + _bn_relu_flops(32, h, w)
    
    # Final conv
    flops, h, w = _conv_flops(32, 32, h, w, 3, padding=1)
    total_flops += flops + _bn_relu_flops(32, h, w)
    
    # Final deconv 2
    flops, h, w = _deconv_flops(32, num_classes, h, w, 2)
    total_flops += flops
    
    # Final interpolation to original size
    total_flops += _interpolate_flops(num_classes, h, w, h_orig, w_orig)

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
            try:
                _ = model(x)
            except Exception:
                return "Error"
            
    times = []
    with torch.no_grad():
        for _ in range(runs):
            try:
                start = time.perf_counter()
                for _ in range(repeats):
                    _ = model(x)
                end = time.perf_counter()
                times.append((end - start) * 1000 / repeats) # time in ms
            except Exception:
                return "Error"
            
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
    
    NUM_CLASSES = 19  # Cityscapes dataset
    model = LinkNetWithActivation(num_classes=NUM_CLASSES, pretrained=False)
    
    print("LinkNet Profiling (CORRECTED)")
    print("="*130)
    header = (f"{'Input':>18} | {'Theo FLOPs (G)':>16} | {'PTFLOPS (G)':>12} | "
              f"{'Params (M)':>10} | {'Latency (ms)':>12} | {'Act. Size (MB)':>15}")
    print(header)
    print("-"*130)

    # Calculate parameters once, as they are constant
    params_m = count_parameters(model) / 1e6

    for inp in INPUT_RESOLUTIONS:
        # Theoretical FLOPs
        try:
            theo_flops_g = calculate_theoretical_flops(inp, NUM_CLASSES) / 1e9
            theo_str = f"{theo_flops_g:.2f}"
        except Exception as e:
            theo_str = "Error"
            print(f"Warning: Theoretical FLOPs calculation failed for {inp}: {e}")
        
        # PTFLOPS (if available)
        ptflops_g = "N/A"
        if PTFLOPS_AVAILABLE:
            try:
                # ptflops requires a clean model instance without the mixin
                clean_model = LinkNet(num_classes=NUM_CLASSES, pretrained=False)
                flops, _ = get_model_complexity_info(
                    clean_model, inp, as_strings=False, print_per_layer_stat=False, verbose=False)
                ptflops_g = f"{flops / 1e9:.2f}"
            except Exception as e:
                ptflops_g = "Error"
                print(f"Warning: PTFLOPS calculation failed for {inp}: {e}")
        
        # Latency
        try:
            latency_ms = measure_latency(model, inp)
            if isinstance(latency_ms, str):
                latency_str = latency_ms
            else:
                latency_str = f"{latency_ms:.2f}"
        except Exception as e:
            latency_str = "Error"
            print(f"Warning: Latency measurement failed for {inp}: {e}")

        # Activation Size
        try:
            with torch.no_grad():
                _ = model(torch.randn(1, *inp))
            act_size_mb = f"{model.activation_bytes / (1024**2):.2f}"
        except Exception as e:
            act_size_mb = "Error"
            print(f"Warning: Activation size measurement failed for {inp}: {e}")
        
        # Print results for the current resolution
        print(f"{str(inp):>18} | {theo_str:>16} | {str(ptflops_g):>12} | "
              f"{params_m:10.2f} | {latency_str:>12} | {act_size_mb:>15}")

if __name__ == "__main__":
    main()
