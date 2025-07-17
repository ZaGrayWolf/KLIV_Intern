#!/usr/bin/env python3
"""
FCN-ResNet50 Profiling Script – Robust Theoretical FLOPs
- Now includes convolutions (2*MACs), batchnorm (4 FLOPs/elem), activations/adds (1 FLOP/elem), and interpolation (7 FLOPs/elem).
- Measures THOP, PTFLOPs, parameter count, activation memory, and CPU latency.
"""

import torch
import torch.nn.functional as F
from torchvision.models.segmentation import fcn_resnet50
from thop import profile
from ptflops import get_model_complexity_info
import time

def build_model(num_classes=21):
    # Initializes the FCN-ResNet50 model with default pre-trained weights
    # and sets it to evaluation mode on the CPU.
    # aux_loss=True is important as it includes the auxiliary head in the model.
    return fcn_resnet50(weights='DEFAULT', num_classes=num_classes, aux_loss=True).cpu().eval()

def calc_conv_flops(ci, co, kh, kw, ho, wo, groups=1):
    # Calculates the total FLOPs for a standard Conv+BN+ReLU block.
    # It adheres to the 2*MACs convention for convolutions.
    macs = ci * co * kh * kw * ho * wo / groups
    # Total FLOPs = Convolution (2*MACs) + BatchNorm (4 FLOPs/elem) + ReLU (1 FLOP/elem)
    flops = 2 * macs + 5 * co * ho * wo
    return flops

def calc_add_flops(ch, h, w):
    # Calculates FLOPs for an element-wise addition.
    return ch * h * w

def calc_interp_flops(ch, h_out, w_out):
    # Estimates FLOPs for bilinear interpolation (approx. 7 FLOPs per output element).
    return 7 * ch * h_out * w_out

def _calc_bottleneck_flops(c_in, c_out, h_in, w_in, stride):
    """
    Helper function to calculate the FLOPs for a single ResNet bottleneck block.
    This includes the three main convolutions and the residual path (with projection if needed).
    """
    flops = 0
    h_out, w_out = h_in // stride, w_in // stride
    c_mid = c_out // 4

    # --- Main Path ---
    flops += calc_conv_flops(c_in, c_mid, 1, 1, h_out, w_out)
    flops += calc_conv_flops(c_mid, c_mid, 3, 3, h_out, w_out)
    macs = c_mid * c_out * 1 * 1 * h_out * w_out
    flops += 2 * macs + 4 * c_out * h_out * w_out # Conv + BN (final ReLU is after add)

    # --- Shortcut Path ---
    if stride > 1 or c_in != c_out:
        macs_shortcut = c_in * c_out * 1 * 1 * h_out * w_out
        flops += 2 * macs_shortcut + 4 * c_out * h_out * w_out # Conv + BN on shortcut

    # --- Final Add and ReLU ---
    flops += calc_add_flops(c_out, h_out, w_out)
    return flops, c_out, h_out, w_out

def theoretical_flops_fcn50(input_res):
    """
    Calculates the theoretical FLOPs for an FCN-ResNet50 model.
    This rewritten function accurately models the architecture, including:
    1. Dilated convolutions in the backbone, preserving spatial resolution.
    2. The auxiliary classifier head which runs in parallel to the main head.
    """
    C, H, W = input_res
    num_classes = 21
    flops = 0

    # --- Backbone: Dilated ResNet-50 ---
    # Layer 0: Initial Conv (s=2) and MaxPool (s=2) -> H/4, W/4
    ho, wo = H // 2, W // 2
    flops += calc_conv_flops(C, 64, 7, 7, ho, wo)
    h, w, c = ho // 2, wo // 2, 64

    # Layer 1: 3 blocks, s=1. Resolution remains H/4, W/4
    f, c, h, w = _calc_bottleneck_flops(c, 256, h, w, stride=1)
    flops += f
    for _ in range(2):
        f, c, h, w = _calc_bottleneck_flops(c, 256, h, w, stride=1)
        flops += f

    # Layer 2: 4 blocks, s=2 for the first block. -> H/8, W/8
    f, c, h, w = _calc_bottleneck_flops(c, 512, h, w, stride=2)
    flops += f
    for _ in range(3):
        f, c, h, w = _calc_bottleneck_flops(c, 512, h, w, stride=1)
        flops += f

    # Layer 3: 6 blocks, s=1 (stride is replaced with dilation). Resolution remains H/8, W/8
    f, c, h, w = _calc_bottleneck_flops(c, 1024, h, w, stride=1)
    flops += f
    for _ in range(5):
        f, c, h, w = _calc_bottleneck_flops(c, 1024, h, w, stride=1)
        flops += f
    
    # Store the output of Layer 3 for the Auxiliary Head
    c_aux, h_aux, w_aux = c, h, w

    # Layer 4: 3 blocks, s=1 (stride is replaced with dilation). Resolution remains H/8, W/8
    f, c, h, w = _calc_bottleneck_flops(c, 2048, h, w, stride=1)
    flops += f
    for _ in range(2):
        f, c, h, w = _calc_bottleneck_flops(c, 2048, h, w, stride=1)
        flops += f

    # --- Main Classifier Head (FCNHead on Layer 4 output) ---
    # FCNHead has an intermediate conv (2048 -> 512) and a final conv (512 -> 21)
    flops += calc_conv_flops(c, 512, 3, 3, h, w)
    flops += 2 * (512 * num_classes * 1 * 1 * h * w) # Final 1x1 conv (MACs * 2)

    # --- Auxiliary Classifier Head (FCNHead on Layer 3 output) ---
    # This head is always active during training and evaluation
    # Intermediate conv (1024 -> 256) and a final conv (256 -> 21)
    flops += calc_conv_flops(c_aux, 256, 3, 3, h_aux, w_aux)
    flops += 2 * (256 * num_classes * 1 * 1 * h_aux * w_aux) # Final 1x1 conv (MACs * 2)

    # Final Upsampling of the main output back to the original resolution
    flops += calc_interp_flops(num_classes, H, W)

    return flops

# Function to calculate activation memory for FCN-ResNet50
def calculate_activation_memory(input_size, dtype="float32"):
    """
    Calculate activation memory for FCN-ResNet50 model with dilated convolutions.
    
    Args:
        input_size: Tuple of (C, H, W) for input tensor
        dtype: Data type (float32 or float16)
        
    Returns:
        dict: Dictionary containing activation memory information
    """
    C, H, W = input_size
    
    # Set bytes per element based on dtype
    if dtype == "float32":
        bytes_per_element = 4
    elif dtype == "float16":
        bytes_per_element = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    # Store activation sizes for each stage
    activations = []
    
    # Initial convolution (7x7, stride 2) and MaxPool (stride 2)
    h, w = H // 2, W // 2  # After conv1
    activations.append(64 * h * w)  # conv1 output
    
    h, w = h // 2, w // 2  # After maxpool
    activations.append(64 * h * w)  # maxpool output
    
    # Layer 1 (3 bottleneck blocks) - no stride, output is 256 channels
    activations.append(256 * h * w)  # layer1 output
    
    # Layer 2 (4 bottleneck blocks) - first block has stride 2
    h, w = h // 2, w // 2
    activations.append(512 * h * w)  # layer2 output
    
    # Layer 3 (6 bottleneck blocks) - dilated convs, no spatial reduction
    # In FCN, layer3 uses dilation=2 instead of stride
    activations.append(1024 * h * w)  # layer3 output
    
    # Layer 4 (3 bottleneck blocks) - dilated convs, no spatial reduction
    # In FCN, layer4 uses dilation=4 instead of stride
    activations.append(2048 * h * w)  # layer4 output
    
    # FCN head - classifier
    activations.append(512 * h * w)   # Intermediate conv output
    activations.append(21 * h * w)    # Classifier output (before upsampling)
    
    # Auxiliary head (from layer3)
    activations.append(256 * h * w)   # Auxiliary intermediate conv
    activations.append(21 * h * w)    # Auxiliary classifier output
    
    # Final upsampled output
    activations.append(21 * H * W)    # Final output
    
    # Calculate total activation memory (forward pass)
    total_activation = sum(activations) * bytes_per_element
    
    # Peak activation memory (largest intermediate tensor)
    peak_activation = max(activations) * bytes_per_element
    
    # Convert to MB
    total_activation_mb = total_activation / (1024 * 1024)
    peak_activation_mb = peak_activation / (1024 * 1024)
    
    return {
        'total_activation_memory': total_activation_mb,
        'peak_activation_memory': peak_activation_mb
    }

def measure_thop(model, inp):
    x = torch.randn(1, *inp)
    flops, params = profile(model, inputs=(x,), verbose=False)
    return flops, params

def measure_ptflops(model, inp):
    # ptflops returns MACs, so we multiply by 2 to get FLOPs.
    macs, params = get_model_complexity_info(model, inp, as_strings=False,
                                              print_per_layer_stat=False, verbose=False)
    return 2 * macs, params

def measure_latency(model, inp):
    x = torch.randn(1, *inp)
    with torch.no_grad():
        # Warmup runs
        for _ in range(5): model(x)
        times = []
        # Timed runs
        for _ in range(10):
            t0 = time.perf_counter()
            model(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000) # milliseconds
    return sum(times) / len(times)

if __name__ == "__main__":
    model = build_model()
    resolutions = [
        (3,640,360),(3,1280,720),(3,1360,760),(3,1600,900),
        (3,1920,1080),(3,2048,1152),(3,2560,1440),(3,3840,2160)
    ]

   # print(f"{'Input':>14} | {'Theo FLOPs (G)':>14} | {'THOP (G)':>10} | {'PTFLOPS (G)':>12} | {'Params (M)':>10} | {'Act Mem (MB)':>12} | {'Peak Mem (MB)':>12} | {'Latency(ms)':>12}")
   #print("-" * 120)
    for inp in resolutions:
        # Theoretical FLOPs
        theo=0;
       # theo = theoretical_flops_fcn50(inp) / 1e9
        th_f=0;
        # THOP
      #  th_f, th_p = measure_thop(model, inp)
        pt_f=0
        pt_p=0        
        # PTFLOPs
      #  pt_f, pt_p = measure_ptflops(model, inp)
        th_f=0
        th_p=0
        # Activation Memory
        mem_info = calculate_activation_memory(inp)
        print(inp)
        print(mem_info)
        # Latency
       # lt = measure_latency(model, inp)
        lt=0
        
      #  print(f"{str(inp):>14} | {theo:14.2f} | {th_f/1e9:10.2f} | {pt_f/1e9:12.2f} | {th_p/1e6:10.2f} | {mem_info['total_activation_memory']:12.2f} | {mem_info['peak_activation_memory']:12.2f} | {lt:12.2f}")
