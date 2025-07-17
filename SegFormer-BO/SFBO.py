#!/usr/bin/env python3
"""
SegFormer-B0 Profiling Script (FULLY CORRECTED):
Calculates Theoretical FLOPs, PTFLOPS, Parameters, Latency, and Activation Size
for the SegFormer-B0 semantic segmentation model across various input resolutions.
FIXES: All tensor dimension mismatch issues and forward pass errors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import warnings
import gc
import os
import sys
import math
from typing import List, Tuple

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

# Optional profiling libraries
try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    PTFLOPS_AVAILABLE = False
    print("Warning: 'ptflops' is not installed. PTFLOPS measurements will be skipped.")

# ------------------------------------------------------------------------------
# 1. SegFormer-B0 Model Implementation (CORRECTED)
# ------------------------------------------------------------------------------

class OverlapPatchMerging(nn.Module):
    """
    Overlap Patch Merging layer with proper dimension handling.
    """
    def __init__(self, in_channels: int, out_channels: int, patch_size: int, stride: int):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=stride, 
                              padding=patch_size // 2, bias=False)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        # Input: B, C, H, W
        B, C, H, W = x.shape
        
        # Apply convolution
        x = self.conv(x)  # B, C_out, H', W'
        _, C_out, H_new, W_new = x.shape
        
        # Flatten spatial dimensions: B, C_out, H', W' -> B, H'*W', C_out
        x = x.flatten(2).transpose(1, 2)  # B, H'*W', C_out
        
        # Apply layer normalization
        x = self.norm(x)
        
        return x, H_new, W_new

class EfficientSelfAttention(nn.Module):
    """
    Efficient Self-Attention module with sequence reduction (CORRECTED).
    """
    def __init__(self, channels: int, num_heads: int, reduction_ratio: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.reduction_ratio = reduction_ratio
        
        self.q = nn.Linear(channels, channels, bias=True)
        self.kv = nn.Linear(channels, channels * 2, bias=True)
        self.proj = nn.Linear(channels, channels)
        
        self.sr = None
        self.norm = None
        if reduction_ratio > 1:
            self.sr = nn.Conv2d(channels, channels, kernel_size=reduction_ratio, stride=reduction_ratio)
            self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        
        # Generate Q
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Generate K, V with optional sequence reduction
        if self.sr is not None and self.reduction_ratio > 1:
            # Reshape for spatial reduction
            x_spatial = x.transpose(1, 2).reshape(B, C, H, W)
            x_reduced = self.sr(x_spatial)  # B, C, H/R, W/R
            _, _, H_r, W_r = x_reduced.shape
            x_reduced = x_reduced.reshape(B, C, -1).transpose(1, 2)  # B, H_r*W_r, C
            x_reduced = self.norm(x_reduced)
            x_kv = x_reduced
        else:
            x_kv = x
            
        kv = self.kv(x_kv).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # B, num_heads, N_kv, head_dim
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class MixFFN(nn.Module):
    """
    Mix-FeedForward Network with depth-wise convolutions (CORRECTED).
    """
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.dwconv = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=True, groups=hidden_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_channels, in_channels)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        
        # First linear layer
        x = self.fc1(x)
        C_hidden = x.shape[-1]
        
        # Reshape for depth-wise convolution
        x = x.transpose(1, 2).reshape(B, C_hidden, H, W)
        x = self.dwconv(x)
        x = x.reshape(B, C_hidden, -1).transpose(1, 2)
        
        # Activation and second linear layer
        x = self.act(x)
        x = self.fc2(x)
        return x

class SegFormerEncoderBlock(nn.Module):
    """
    A single block of the SegFormer encoder (CORRECTED).
    """
    def __init__(self, channels: int, num_heads: int, mlp_ratio: int = 4, reduction_ratio: int = 1):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = EfficientSelfAttention(channels, num_heads, reduction_ratio)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = MixFFN(channels, channels * mlp_ratio)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x), H, W)
        # FFN with residual connection
        x = x + self.ffn(self.norm2(x), H, W)
        return x

class MiTEncoder(nn.Module):
    """
    The Mix Transformer (MiT) encoder backbone (CORRECTED).
    """
    def __init__(self, in_channels: int, widths: List[int], depths: List[int], 
                 num_heads: List[int], reduction_ratios: List[int], 
                 patch_sizes: List[int], strides: List[int], mlp_ratio: int):
        super().__init__()
        
        self.num_stages = len(depths)
        self.patch_merging_layers = nn.ModuleList()
        self.transformer_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(self.num_stages):
            # Patch merging layer
            in_ch = in_channels if i == 0 else widths[i-1]
            patch_merge = OverlapPatchMerging(
                in_channels=in_ch,
                out_channels=widths[i],
                patch_size=patch_sizes[i],
                stride=strides[i]
            )
            self.patch_merging_layers.append(patch_merge)
            
            # Transformer blocks for this stage
            transformer_blocks = nn.ModuleList([
                SegFormerEncoderBlock(widths[i], num_heads[i], mlp_ratio, reduction_ratios[i])
                for _ in range(depths[i])
            ])
            self.transformer_layers.append(transformer_blocks)
            
            # Layer normalization
            self.norms.append(nn.LayerNorm(widths[i]))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        
        for i in range(self.num_stages):
            # Apply patch merging
            x, H, W = self.patch_merging_layers[i](x)
            
            # Apply transformer blocks
            for block in self.transformer_layers[i]:
                x = block(x, H, W)
            
            # Apply layer normalization
            x = self.norms[i](x)
            
            # Reshape back to spatial format: B, N, C -> B, C, H, W
            B, N, C = x.shape
            x_spatial = x.transpose(1, 2).reshape(B, C, H, W)
            features.append(x_spatial)
            
            # Prepare for next stage (x remains in sequence format)
            x = x_spatial
            
        return features

class SegFormerDecoder(nn.Module):
    """
    Lightweight All-MLP Decoder head (CORRECTED).
    """
    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        self.mlps = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1, bias=False) for in_ch in in_channels_list
        ])

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # Get the target size from the first (highest resolution) feature map
        target_size = features[0].shape[2:]
        
        outs = []
        for i, (feature, mlp) in enumerate(zip(features, self.mlps)):
            x = mlp(feature)
            # Upsample to target size if necessary
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            outs.append(x)
        
        # Concatenate all features
        return torch.cat(outs, dim=1)

class SegFormer(nn.Module):
    """
    The complete SegFormer model for semantic segmentation (CORRECTED).
    """
    def __init__(self, in_channels: int, widths: List[int], depths: List[int],
                 num_heads: List[int], reduction_ratios: List[int],
                 patch_sizes: List[int], strides: List[int], mlp_ratio: int,
                 decoder_channels: int, num_classes: int):
        super().__init__()
        self.encoder = MiTEncoder(
            in_channels, widths, depths, num_heads, reduction_ratios,
            patch_sizes, strides, mlp_ratio
        )
        self.decoder = SegFormerDecoder(widths, decoder_channels)
        self.head = nn.Conv2d(decoder_channels * len(widths), num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[-2:]
        
        # Encode
        features = self.encoder(x)
        
        # Decode
        x = self.decoder(features)
        x = self.head(x)
        
        # Upsample to original input size
        if x.shape[2:] != input_shape:
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        return x

def segformer_b0(num_classes: int = 19):
    """
    Instantiates the SegFormer-B0 model with its specific configuration.
    """
    return SegFormer(
        in_channels=3,
        widths=[32, 64, 160, 256],      # C1, C2, C3, C4
        depths=[2, 2, 2, 2],            # L1, L2, L3, L4
        num_heads=[1, 2, 5, 8],         # H1, H2, H3, H4
        reduction_ratios=[8, 4, 2, 1],  # R1, R2, R3, R4
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        mlp_ratio=4,
        decoder_channels=256,
        num_classes=num_classes
    )

# ------------------------------------------------------------------------------
# 2. Activation Tracking Utilities (CORRECTED)
# ------------------------------------------------------------------------------

class ActivationTrackerMixin:
    """A mixin to add activation size tracking to a model."""
    def reset_activation_bytes(self):
        self.activation_bytes = 0

    def add_activation(self, tensor):
        if hasattr(self, 'activation_bytes') and tensor is not None:
            self.activation_bytes += tensor.numel() * 4

class SegFormerWithActivation(ActivationTrackerMixin, SegFormer):
    """Extends SegFormer to track activation sizes."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.reset_activation_bytes()
        self.add_activation(x)
        
        input_shape = x.shape[-2:]
        
        # Encode with activation tracking
        features = []
        current_x = x
        
        for i in range(self.encoder.num_stages):
            # Patch merging
            current_x, H, W = self.encoder.patch_merging_layers[i](current_x)
            self.add_activation(current_x)
            
            # Transformer blocks
            for block in self.encoder.transformer_layers[i]:
                current_x = block(current_x, H, W)
                self.add_activation(current_x)
            
            # Layer norm
            current_x = self.encoder.norms[i](current_x)
            self.add_activation(current_x)
            
            # Reshape to spatial
            B, N, C = current_x.shape
            x_spatial = current_x.transpose(1, 2).reshape(B, C, H, W)
            features.append(x_spatial)
            self.add_activation(x_spatial)
            
            current_x = x_spatial
        
        # Decode
        x = self.decoder(features)
        self.add_activation(x)
        
        x = self.head(x)
        self.add_activation(x)
        
        # Final upsample
        if x.shape[2:] != input_shape:
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            self.add_activation(x)
        
        return x

def segformer_b0_with_activation(num_classes: int = 19):
    """Instantiates SegFormer-B0 with activation tracking."""
    return SegFormerWithActivation(
        in_channels=3,
        widths=[32, 64, 160, 256],
        depths=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        reduction_ratios=[8, 4, 2, 1],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        mlp_ratio=4,
        decoder_channels=256,
        num_classes=num_classes
    )

# ------------------------------------------------------------------------------
# 3. Model Cleanup and Profiling Functions
# ------------------------------------------------------------------------------

def completely_clean_model(model):
    """Removes all ptflops attributes from a model."""
    def deep_clean_module(module):
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
        
        for child in module.children():
            deep_clean_module(child)
    
    deep_clean_module(model)
    return model

def create_fresh_model(num_classes=19):
    """Creates a fresh model instance for ptflops."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model = segformer_b0(num_classes=num_classes)
    return completely_clean_model(model)

def force_cleanup_everything():
    """Forces memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def calculate_theoretical_flops(input_size, num_classes=19):
    """Calculates theoretical FLOPs for SegFormer-B0."""
    c, h, w = input_size
    total_flops = 0
    
    # SegFormer-B0 configuration
    widths = [32, 64, 160, 256]
    depths = [2, 2, 2, 2]
    num_heads = [1, 2, 5, 8]
    reduction_ratios = [8, 4, 2, 1]
    patch_sizes = [7, 3, 3, 3]
    strides = [4, 2, 2, 2]
    decoder_channels = 256
    
    current_h, current_w = h, w
    
    # Stage-wise computation
    for stage_idx in range(4):
        # Patch merging
        in_ch = c if stage_idx == 0 else widths[stage_idx - 1]
        out_ch = widths[stage_idx]
        patch_size = patch_sizes[stage_idx]
        stride = strides[stage_idx]
        
        # Update dimensions
        current_h = math.ceil(current_h / stride)
        current_w = math.ceil(current_w / stride)
        
        # Patch merging conv FLOPs
        total_flops += 2 * in_ch * out_ch * (patch_size ** 2) * current_h * current_w
        
        # Transformer blocks
        for _ in range(depths[stage_idx]):
            seq_len = current_h * current_w
            seq_len_kv = seq_len // (reduction_ratios[stage_idx] ** 2) if reduction_ratios[stage_idx] > 1 else seq_len
            
            # Self-attention FLOPs (simplified)
            total_flops += 4 * seq_len * out_ch * out_ch  # Q, K, V projections + output
            total_flops += 2 * num_heads[stage_idx] * seq_len * seq_len_kv * (out_ch // num_heads[stage_idx])
            
            # MixFFN FLOPs
            hidden_ch = out_ch * 4
            total_flops += 2 * seq_len * out_ch * hidden_ch  # FC1
            total_flops += hidden_ch * 9 * current_h * current_w  # DW Conv
            total_flops += 2 * seq_len * hidden_ch * out_ch  # FC2
    
    # Decoder FLOPs (simplified)
    for width in widths:
        total_flops += 2 * width * decoder_channels * (h // 4) * (w // 4)
    
    # Final head
    total_flops += 2 * (decoder_channels * len(widths)) * num_classes * (h // 4) * (w // 4)
    
    # Final upsampling
    total_flops += 7 * num_classes * h * w
    
    return total_flops

def count_parameters(model):
    """Counts trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_latency(model, input_size, runs=3, repeats=3):
    """Measures average inference latency."""
    try:
        device = torch.device('cpu')
        model.to(device).eval()
        x = torch.randn(1, *input_size).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(2):
                _ = model(x)
        
        times = []
        with torch.no_grad():
            for _ in range(runs):
                start = time.perf_counter()
                for _ in range(repeats):
                    _ = model(x)
                end = time.perf_counter()
                times.append((end - start) * 1000 / repeats)
        
        return sum(times) / len(times)
    
    except Exception as e:
        return "Error"

# ------------------------------------------------------------------------------
# 4. Main Execution
# ------------------------------------------------------------------------------

def main():
    """Main profiling function."""
    INPUT_RESOLUTIONS = [
         (3, 3840, 2160)
    ]
    
    NUM_CLASSES = 19
    
    print("SegFormer-B0 Profiling (FULLY CORRECTED)")
    print("="*130)
    header = (f"{'Input':>18} | {'Theo FLOPs (G)':>16} | {'PTFLOPS (G)':>12} | "
              f"{'Params (M)':>10} | {'Latency (ms)':>12} | {'Act. Size (MB)':>15}")
    print(header)
    print("-"*130)

    # Create model for main measurements
    try:
        model_for_activation = segformer_b0_with_activation(num_classes=NUM_CLASSES)
        params_m = count_parameters(model_for_activation) / 1e6
    except Exception as e:
        print(f"Error creating model: {e}")
        return

    for inp in INPUT_RESOLUTIONS:
        force_cleanup_everything()
        
        # Theoretical FLOPs
        try:
            theo_flops_g = calculate_theoretical_flops(inp, NUM_CLASSES) / 1e9
            theo_str = f"{theo_flops_g:.2f}"
        except Exception as e:
            theo_str = "Error"
        
        # PTFLOPS
        ptflops_g = "N/A"
        if PTFLOPS_AVAILABLE:
            try:
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                
                with open(os.devnull, 'w') as devnull:
                    sys.stdout = devnull
                    sys.stderr = devnull
                    
                    isolated_model = create_fresh_model(num_classes=NUM_CLASSES)
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        flops, _ = get_model_complexity_info(
                            isolated_model, inp, as_strings=False, 
                            print_per_layer_stat=False, verbose=False)
                        ptflops_g = f"{flops / 1e9:.2f}"
                    
                    del isolated_model
                
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                force_cleanup_everything()
                
            except Exception as e:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                ptflops_g = "Error"
        
        # Latency
        try:
            latency_ms = measure_latency(model_for_activation, inp)
            latency_str = f"{latency_ms:.2f}" if isinstance(latency_ms, (int, float)) else latency_ms
        except Exception as e:
            latency_str = "Error"

        # Activation Size
        try:
            with torch.no_grad():
                _ = model_for_activation(torch.randn(1, *inp))
            act_size_mb = f"{model_for_activation.activation_bytes / (1024**2):.2f}"
        except Exception as e:
            act_size_mb = "Error"
        
        # Print results
        print(f"{str(inp):>18} | {theo_str:>16} | {str(ptflops_g):>12} | "
              f"{params_m:10.2f} | {latency_str:>12} | {act_size_mb:>15}")

    print("\nSegFormer-B0 profiling completed successfully!")

if __name__ == "__main__":
    main()
