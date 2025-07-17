#!/usr/bin/env python3
"""
PSPNet Profiling Script (FULLY CORRECTED):
Calculates Theoretical FLOPs, PTFLOPS, Parameters, Latency, and Activation Size
for the PSPNet semantic segmentation model across various input resolutions.
FIXES: Complete elimination of ptflops warnings and memory issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import warnings
import gc
import sys
from typing import Dict, Any, Tuple

# Completely suppress all warnings including ptflops warnings
warnings.filterwarnings("ignore")
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

# Optional profiling libraries
try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    PTFLOPS_AVAILABLE = False
    print("Warning: 'ptflops' is not installed. PTFLOPS measurements will be skipped.")

# ------------------------------------------------------------------------------
# 1. ResNet Backbone Implementation (Simplified for PSPNet)
# ------------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def resnet50(pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    return model

# ------------------------------------------------------------------------------
# 2. PSPNet Model Components
# ------------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, bias=False):
        super(ConvBlock, self).__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.conv(x)
        return out

def upsample(input, size=None, scale_factor=None, align_corners=False):
    out = F.interpolate(input, size=size, scale_factor=scale_factor, mode='bilinear', align_corners=align_corners)
    return out

class PyramidPooling(nn.Module):
    def __init__(self, in_channels):
        super(PyramidPooling, self).__init__()
        self.pooling_size = [1, 2, 3, 6]
        self.channels = in_channels // 4

        self.pool1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pooling_size[0]),
            ConvBlock(in_channels, self.channels, kernel_size=1),
        )

        self.pool2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pooling_size[1]),
            ConvBlock(in_channels, self.channels, kernel_size=1),
        )

        self.pool3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pooling_size[2]),
            ConvBlock(in_channels, self.channels, kernel_size=1),
        )

        self.pool4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pooling_size[3]),
            ConvBlock(in_channels, self.channels, kernel_size=1),
        )

    def forward(self, x):
        out1 = self.pool1(x)
        out1 = upsample(out1, size=x.size()[-2:])

        out2 = self.pool2(x)
        out2 = upsample(out2, size=x.size()[-2:])

        out3 = self.pool3(x)
        out3 = upsample(out3, size=x.size()[-2:])

        out4 = self.pool4(x)
        out4 = upsample(out4, size=x.size()[-2:])

        out = torch.cat([x, out1, out2, out3, out4], dim=1)
        return out

# ------------------------------------------------------------------------------
# 3. PSPNet Model (Simplified for Profiling)
# ------------------------------------------------------------------------------

class PSPNetSimplified(nn.Module):
    def __init__(self, n_classes=21):
        super(PSPNetSimplified, self).__init__()
        self.out_channels = 2048

        self.backbone = resnet50(pretrained=False)
        self.stem = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool
        )
        self.block1 = self.backbone.layer1
        self.block2 = self.backbone.layer2
        self.block3 = self.backbone.layer3
        self.block4 = self.backbone.layer4

        self.depth = self.out_channels // 4
        self.pyramid_pooling = PyramidPooling(self.out_channels)

        self.decoder = nn.Sequential(
            ConvBlock(self.out_channels * 2, self.depth, kernel_size=3),
            nn.Dropout(0.1),
            nn.Conv2d(self.depth, n_classes, kernel_size=1),
        )

        self.aux = nn.Sequential(
            ConvBlock(self.out_channels // 2, self.depth // 2, kernel_size=3),
            nn.Dropout(0.1),
            nn.Conv2d(self.depth // 2, n_classes, kernel_size=1),
        )

    def forward(self, x):
        original_size = x.size()[-2:]
        
        out = self.stem(x)
        out1 = self.block1(out)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        aux_out = self.aux(out3)
        aux_out = upsample(aux_out, size=original_size, align_corners=True)
        out4 = self.block4(out3)

        out = self.pyramid_pooling(out4)
        out = self.decoder(out)
        out = upsample(out, size=original_size, align_corners=True)

        return out

# ------------------------------------------------------------------------------
# 4. Activation Tracking Utilities
# ------------------------------------------------------------------------------

class ActivationTrackerMixin:
    """A mixin to add activation size tracking to a model."""
    def reset_activation_bytes(self):
        self.activation_bytes = 0

    def add_activation(self, tensor):
        # Accumulate size in bytes (float32 is 4 bytes)
        if hasattr(self, 'activation_bytes'):
            self.activation_bytes += tensor.numel() * 4

class PSPNetWithActivation(ActivationTrackerMixin, PSPNetSimplified):
    """Extends PSPNet to track activation sizes by overriding the forward pass."""
    def forward(self, x):
        self.reset_activation_bytes()
        self.add_activation(x)
        
        original_size = x.size()[-2:]
        
        out = self.stem(x)
        self.add_activation(out)
        
        out1 = self.block1(out)
        self.add_activation(out1)
        
        out2 = self.block2(out1)
        self.add_activation(out2)
        
        out3 = self.block3(out2)
        self.add_activation(out3)
        
        aux_out = self.aux(out3)
        self.add_activation(aux_out)
        aux_out = upsample(aux_out, size=original_size, align_corners=True)
        self.add_activation(aux_out)
        
        out4 = self.block4(out3)
        self.add_activation(out4)

        out = self.pyramid_pooling(out4)
        self.add_activation(out)
        
        out = self.decoder(out)
        self.add_activation(out)
        
        out = upsample(out, size=original_size, align_corners=True)
        self.add_activation(out)

        return out

# ------------------------------------------------------------------------------
# 5. Comprehensive Model Cleanup (FULLY CORRECTED VERSION)
# ------------------------------------------------------------------------------

def completely_clean_model(model):
    """
    Completely removes all ptflops and profiling-related attributes from a model.
    This is the most comprehensive cleanup function.
    """
    def deep_clean_module(module):
        # List of all possible attributes that ptflops or other profilers might add
        ptflops_attrs = [
            '__flops__', '__params__', '__macc__', '__conv_flops__', '__bn_flops__',
            '__relu_flops__', '__pool_flops__', '__linear_flops__', '__upsample_flops__',
            '__activation_flops__', '__batch_counter__', '__flops_handle__', '__params_handle__'
        ]
        
        for attr in ptflops_attrs:
            if hasattr(module, attr):
                try:
                    delattr(module, attr)
                except (AttributeError, RuntimeError):
                    pass
        
        # Clean all child modules recursively
        for child in module.children():
            deep_clean_module(child)
        
        # Clean named modules
        for name, child in module.named_modules():
            if child != module:  # Avoid infinite recursion
                deep_clean_module(child)
    
    deep_clean_module(model)
    return model

def create_completely_fresh_model(n_classes=21):
    """
    Creates a completely fresh model instance with immediate cleanup.
    """
    # Force garbage collection before creating new model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create new model
    model = PSPNetSimplified(n_classes=n_classes)
    
    # Immediately clean it
    model = completely_clean_model(model)
    
    return model

def force_cleanup_everything():
    """
    Forces comprehensive cleanup of memory and cache.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ------------------------------------------------------------------------------
# 6. Profiling Functions
# ------------------------------------------------------------------------------

def calculate_theoretical_flops(input_size, num_classes=21):
    """Calculates the total theoretical FLOPs for the PSPNet model."""
    c, h, w = input_size
    h_orig, w_orig = h, w
    total_flops = 0

    # Simplified calculation focusing on major components
    # Initial conv and stem
    total_flops += 2 * c * 64 * (7**2) * (h//2) * (w//2)  # Conv1
    total_flops += 64 * (h//2) * (w//2) * 5  # BN + ReLU
    
    # ResNet50 backbone (approximation)
    # This is a simplified calculation of the backbone FLOPs
    current_h, current_w = h//4, w//4  # After stem + maxpool
    
    # Layer 1: 3 bottleneck blocks, channels: 64->256
    for _ in range(3):
        total_flops += 2 * 64 * 64 * current_h * current_w  # 1x1 conv
        total_flops += 2 * 64 * 64 * 9 * current_h * current_w  # 3x3 conv
        total_flops += 2 * 64 * 256 * current_h * current_w  # 1x1 conv
        total_flops += 256 * current_h * current_w * 15  # BNs + ReLUs + residual
    
    # Layer 2: 4 bottleneck blocks, channels: 256->512, first stride=2
    current_h, current_w = current_h//2, current_w//2
    for i in range(4):
        in_ch = 256 if i == 0 else 512
        total_flops += 2 * in_ch * 128 * current_h * current_w  # 1x1 conv
        total_flops += 2 * 128 * 128 * 9 * current_h * current_w  # 3x3 conv
        total_flops += 2 * 128 * 512 * current_h * current_w  # 1x1 conv
        total_flops += 512 * current_h * current_w * 15  # BNs + ReLUs + residual
    
    # Layer 3: 6 bottleneck blocks, channels: 512->1024, first stride=2
    current_h, current_w = current_h//2, current_w//2
    for i in range(6):
        in_ch = 512 if i == 0 else 1024
        total_flops += 2 * in_ch * 256 * current_h * current_w  # 1x1 conv
        total_flops += 2 * 256 * 256 * 9 * current_h * current_w  # 3x3 conv
        total_flops += 2 * 256 * 1024 * current_h * current_w  # 1x1 conv
        total_flops += 1024 * current_h * current_w * 15  # BNs + ReLUs + residual
    
    # Layer 4: 3 bottleneck blocks, channels: 1024->2048, dilated
    for i in range(3):
        in_ch = 1024 if i == 0 else 2048
        total_flops += 2 * in_ch * 512 * current_h * current_w  # 1x1 conv
        total_flops += 2 * 512 * 512 * 9 * current_h * current_w  # 3x3 conv (dilated)
        total_flops += 2 * 512 * 2048 * current_h * current_w  # 1x1 conv
        total_flops += 2048 * current_h * current_w * 15  # BNs + ReLUs + residual
    
    # Pyramid pooling (approximation)
    pyramid_flops = 0
    for pool_size in [1, 2, 3, 6]:
        # Pooling + 1x1 conv + upsampling
        pyramid_flops += 2048 * pool_size * pool_size  # Pooling
        pyramid_flops += 2 * 2048 * 512 * pool_size * pool_size  # 1x1 conv
        pyramid_flops += 7 * 512 * current_h * current_w  # Bilinear upsampling
    total_flops += pyramid_flops
    
    # Decoder
    total_flops += 2 * (2048*2) * 512 * 9 * current_h * current_w  # Decoder conv
    total_flops += 2 * 512 * num_classes * current_h * current_w  # Final conv
    
    # Final upsampling
    total_flops += 7 * num_classes * h_orig * w_orig
    
    # Auxiliary branch (approximation)
    total_flops += 2 * 1024 * 256 * 9 * current_h * current_w  # Aux conv
    total_flops += 2 * 256 * num_classes * current_h * current_w  # Aux final
    total_flops += 7 * num_classes * h_orig * w_orig  # Aux upsampling

    return total_flops

def count_parameters(model):
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_latency(model, input_size, runs=3, repeats=5):
    """Measures the average inference latency on CPU with reduced iterations."""
    device = torch.device('cpu')
    model.to(device).eval()
    x = torch.randn(1, *input_size).to(device)
    
    # Reduced warmup runs
    with torch.no_grad():
        for _ in range(2):
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
# 7. Main Execution (FULLY CORRECTED VERSION)
# ------------------------------------------------------------------------------

def main():
    """Main function to run the profiling script with complete error handling."""
    INPUT_RESOLUTIONS = [
        (3, 640, 360), (3, 1280, 720), (3, 1360, 760),
        (3, 1600, 900), (3, 1920, 1080), (3, 2048, 1152),
        (3, 2560, 1440), (3, 3840, 2160)
    ]
    
    NUM_CLASSES = 21  # Pascal VOC
    
    print("PSPNet Profiling (FULLY CORRECTED - No Warnings)")
    print("="*130)
    header = (f"{'Input':>18} | {'Theo FLOPs (G)':>16} | {'PTFLOPS (G)':>12} | "
              f"{'Params (M)':>10} | {'Latency (ms)':>12} | {'Act. Size (MB)':>15}")
    print(header)
    print("-"*130)

    # Create model for activation tracking and parameter counting only once
    model_for_activation = PSPNetWithActivation(n_classes=NUM_CLASSES)
    params_m = count_parameters(model_for_activation) / 1e6

    for inp in INPUT_RESOLUTIONS:
        # Force cleanup before each iteration
        force_cleanup_everything()
        
        # Theoretical FLOPs
        try:
            theo_flops_g = calculate_theoretical_flops(inp, NUM_CLASSES) / 1e9
            theo_str = f"{theo_flops_g:.2f}"
        except Exception as e:
            theo_str = "Error"
            print(f"Warning: Theoretical FLOPs calculation failed for {inp}: {e}")
        
        # PTFLOPS (if available) - COMPLETELY ISOLATED EXECUTION
        ptflops_g = "N/A"
        if PTFLOPS_AVAILABLE:
            try:
                # Suppress all output during ptflops execution
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                
                with open(os.devnull, 'w') as devnull:
                    sys.stdout = devnull
                    sys.stderr = devnull
                    
                    # Create completely fresh model in isolated scope
                    isolated_model = create_completely_fresh_model(n_classes=NUM_CLASSES)
                    
                    # Run ptflops with complete output suppression
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        flops, _ = get_model_complexity_info(
                            isolated_model, inp, as_strings=False, 
                            print_per_layer_stat=False, verbose=False)
                        ptflops_g = f"{flops / 1e9:.2f}"
                    
                    # Complete cleanup
                    completely_clean_model(isolated_model)
                    del isolated_model
                
                # Restore stdout/stderr
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                
                force_cleanup_everything()
                
            except Exception as e:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                ptflops_g = "Error"
        
        # Latency - simplified measurement
        try:
            latency_ms = measure_latency(model_for_activation, inp)
            if isinstance(latency_ms, str):
                latency_str = latency_ms
            else:
                latency_str = f"{latency_ms:.2f}"
        except Exception as e:
            latency_str = "Error"

        # Activation Size
        try:
            with torch.no_grad():
                _ = model_for_activation(torch.randn(1, *inp))
            act_size_mb = f"{model_for_activation.activation_bytes / (1024**2):.2f}"
        except Exception as e:
            act_size_mb = "Error"
        
        # Print results for the current resolution
        print(f"{str(inp):>18} | {theo_str:>16} | {str(ptflops_g):>12} | "
              f"{params_m:10.2f} | {latency_str:>12} | {act_size_mb:>15}")

    print("\nProfiling completed successfully!")

if __name__ == "__main__":
    main()
