#!/usr/bin/env python3
"""
ENet Profiling Script - Complete Analysis with Corrected Latency Measurement
- Resolves TypeError in latency function call.
- Fixes latency showing 0 by increasing workload per timing measurement.
- Includes full profiling for activation memory, THOP, PTFLOPs, and all other metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import warnings
warnings.filterwarnings("ignore")

# Try to import profiling libraries with graceful fallbacks
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    print("Warning: 'thop' is not installed. THOP measurements will be skipped. Install with 'pip install thop'.")
    THOP_AVAILABLE = False

try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    print("Warning: 'ptflops' is not installed. PTFLOPs measurements will be skipped. Install with 'pip install ptflops'.")
    PTFLOPS_AVAILABLE = False

# 1. ENet Model Definition (from user's file)

class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, relu=True):
        super().__init__()
        activation = nn.ReLU if relu else nn.PReLU
        self.main_branch = nn.Conv2d(in_channels, out_channels - 3, 3, 2, 1, bias=bias)
        self.ext_branch = nn.MaxPool2d(3, 2, 1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.out_activation = activation()
    def forward(self, x):
        main = self.main_branch(x); ext = self.ext_branch(x)
        out = torch.cat((main, ext), 1); out = self.batch_norm(out)
        return self.out_activation(out)

class RegularBottleneck(nn.Module):
    def __init__(self, channels, internal_ratio=4, kernel_size=3, padding=0,
                 dilation=1, asymmetric=False, dropout_prob=0, bias=False, relu=True):
        super().__init__()
        internal_channels = channels // internal_ratio
        activation = nn.ReLU if relu else nn.PReLU
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(channels, internal_channels, 1, 1, bias=bias),
            nn.BatchNorm2d(internal_channels), activation())
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, (kernel_size, 1), 1, (padding, 0), dilation=dilation, bias=bias),
                nn.BatchNorm2d(internal_channels), activation(),
                nn.Conv2d(internal_channels, internal_channels, (1, kernel_size), 1, (0, padding), dilation=dilation, bias=bias),
                nn.BatchNorm2d(internal_channels), activation())
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size, 1, padding, dilation=dilation, bias=bias),
                nn.BatchNorm2d(internal_channels), activation())
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, channels, 1, 1, bias=bias),
            nn.BatchNorm2d(channels), activation())
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = activation()
    def forward(self, x):
        main = x
        ext = self.ext_conv1(x); ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext); ext = self.ext_regul(ext)
        out = main + ext
        return self.out_activation(out)

class DownsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4,
                 return_indices=False, dropout_prob=0, bias=False, relu=True):
        super().__init__()
        self.return_indices = return_indices
        internal_channels = in_channels // internal_ratio
        activation = nn.ReLU if relu else nn.PReLU
        self.main_max1 = nn.MaxPool2d(2, 2, return_indices=return_indices)
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, 2, 2, bias=bias),
            nn.BatchNorm2d(internal_channels), activation())
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels, 3, 1, 1, bias=bias),
            nn.BatchNorm2d(internal_channels), activation())
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, 1, 1, bias=bias),
            nn.BatchNorm2d(out_channels), activation())
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = activation()
    def forward(self, x):
        if self.return_indices: main, idx = self.main_max1(x)
        else: main, idx = self.main_max1(x), None
        ext = self.ext_conv1(x); ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext); ext = self.ext_regul(ext)
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w, device=ext.device)
        main = torch.cat((main, padding), 1)
        out = main + ext
        return self.out_activation(out), idx

class UpsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4,
                 dropout_prob=0, bias=False, relu=True):
        super().__init__()
        internal_channels = in_channels // internal_ratio
        activation = nn.ReLU if relu else nn.PReLU
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=bias),
            nn.BatchNorm2d(out_channels))
        self.main_unpool1 = nn.MaxUnpool2d(2)
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, 1, bias=bias),
            nn.BatchNorm2d(internal_channels), activation())
        self.ext_tconv1 = nn.ConvTranspose2d(internal_channels, internal_channels, 2, 2, bias=bias)
        self.ext_tconv1_bnorm = nn.BatchNorm2d(internal_channels)
        self.ext_tconv1_activation = activation()
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, 1, bias=bias),
            nn.BatchNorm2d(out_channels))
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = activation()
    def forward(self, x, max_indices, output_size):
        main = self.main_conv1(x)
        main = self.main_unpool1(main, max_indices, output_size=output_size)
        ext = self.ext_conv1(x); ext = self.ext_tconv1(ext, output_size=output_size)
        ext = self.ext_tconv1_bnorm(ext); ext = self.ext_tconv1_activation(ext)
        ext = self.ext_conv2(ext); ext = self.ext_regul(ext)
        out = main + ext
        return self.out_activation(out)

class ENet(nn.Module):
    def __init__(self, num_classes, encoder_relu=False, decoder_relu=True):
        super().__init__()
        self.initial_block = InitialBlock(3, 16, relu=encoder_relu)
        self.downsample1_0 = DownsamplingBottleneck(16, 64, return_indices=True, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.downsample2_0 = DownsamplingBottleneck(64, 128, return_indices=True, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_1 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_2 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_4 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_5 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_6 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_7 = RegularBottleneck(128, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_8 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)
        self.regular3_0 = self.regular2_1
        self.dilated3_1 = self.dilated2_2
        self.asymmetric3_2 = self.asymmetric2_3
        self.dilated3_3 = self.dilated2_4
        self.regular3_4 = self.regular2_5
        self.dilated3_5 = self.dilated2_6
        self.asymmetric3_6 = self.asymmetric2_7
        self.dilated3_7 = self.dilated2_8
        self.upsample4_0 = UpsamplingBottleneck(128, 64, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_1 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.upsample5_0 = UpsamplingBottleneck(64, 16, dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = RegularBottleneck(16, padding=1, dropout_prob=0.01, relu=decoder_relu)
        self.transposed_conv = nn.ConvTranspose2d(16, num_classes, 3, 2, 1, 1, bias=False)
    def forward(self, x):
        s0_size = x.size()
        x = self.initial_block(x)
        s1_size = x.size(); x, idx1 = self.downsample1_0(x)
        x = self.regular1_1(x); x = self.regular1_2(x); x = self.regular1_3(x); x = self.regular1_4(x)
        s2_size = x.size(); x, idx2 = self.downsample2_0(x)
        x = self.regular2_1(x); x = self.dilated2_2(x); x = self.asymmetric2_3(x); x = self.dilated2_4(x)
        x = self.regular2_5(x); x = self.dilated2_6(x); x = self.asymmetric2_7(x); x = self.dilated2_8(x)
        x = self.regular3_0(x); x = self.dilated3_1(x); x = self.asymmetric3_2(x); x = self.dilated3_3(x)
        x = self.regular3_4(x); x = self.dilated3_5(x); x = self.asymmetric3_6(x); x = self.dilated3_7(x)
        x = self.upsample4_0(x, idx2, output_size=s2_size)
        x = self.regular4_1(x); x = self.regular4_2(x)
        x = self.upsample5_0(x, idx1, output_size=s1_size)
        x = self.regular5_1(x)
        x = self.transposed_conv(x, output_size=s0_size)
        return x

# 2. Corrected Theoretical FLOPs Calculation
def _calc_conv_flops(in_ch, out_ch, k, s, p, h, w, dil=1, kernel_shape=None):
    if kernel_shape is None: kernel_shape = (k, k)
    kh, kw = kernel_shape
    eff_kh = kh + (kh - 1) * (dil - 1); eff_kw = kw + (kw - 1) * (dil - 1)
    out_h = (h + 2 * p - eff_kh) // s + 1; out_w = (w + 2 * p - eff_kw) // s + 1
    flops = 2 * in_ch * out_ch * kh * kw * out_h * out_w # MACs * 2
    return flops, (out_ch, out_h, out_w)
def _calc_asym_flops(in_ch, out_ch, k, p, h, w, dil=1):
    f1, s1 = _calc_conv_flops(in_ch, out_ch, k, 1, p, h, w, dil, (k, 1))
    f2, s2 = _calc_conv_flops(s1[0], out_ch, k, 1, p, s1[1], s1[2], dil, (1, k))
    return f1 + f2, s2
def _calc_deconv_flops(in_ch, out_ch, k, s, p, h, w, op=0):
    out_h = (h - 1) * s - 2 * p + k + op; out_w = (w - 1) * s - 2 * p + k + op
    flops = 2 * in_ch * out_ch * k * k * h * w # MACs * 2
    return flops, (out_ch, out_h, out_w)

def calculate_enet_theoretical_flops(inp, num_classes=21):
    flops = 0; current_shape = inp # (C, H, W)
    
    # InitialBlock
    # main_branch: Conv2d(in_channels, out_channels - 3, k=3, s=2, p=1)
    f, shape_c = _calc_conv_flops(current_shape[0], 13, 3, 2, 1, current_shape[1], current_shape[2])
    flops += f; current_shape = (16, shape_c[1], shape_c[2])
    # Add non-conv FLOPs for InitialBlock: BN, ReLU (16 channels)
    flops += current_shape[0] * current_shape[1] * current_shape[2] * (4 + 1) # BN (approx 4) + ReLU (1)
    
    # Stage 1 Downsample
    # downsample1_0 (DownsamplingBottleneck): (16, 64)
    # ext_conv1: Conv2d(16, 4, k=2, s=2, p=0)
    f1, s1 = _calc_conv_flops(current_shape[0], 4, 2, 2, 0, current_shape[1], current_shape[2])
    # ext_conv2: Conv2d(4, 4, k=3, s=1, p=1)
    f2, s2 = _calc_conv_flops(s1[0], 4, 3, 1, 1, s1[1], s1[2])
    # ext_conv3: Conv2d(4, 64, k=1, s=1, p=0)
    f3, s3 = _calc_conv_flops(s2[0], 64, 1, 1, 0, s2[1], s2[2])
    flops += f1 + f2 + f3; current_shape = s3
    # Add non-conv FLOPs for DownsamplingBottleneck: BN (3x), ReLU (3x), Add (1x)
    flops += current_shape[0] * current_shape[1] * current_shape[2] * (3*4 + 3*1 + 1)
    
    # 4x RegularBottleneck (64 channels)
    for _ in range(4):
        # ext_conv1: Conv2d(64, 16, k=1, s=1, p=0)
        f1_rb, s1_rb = _calc_conv_flops(current_shape[0], 16, 1, 1, 0, current_shape[1], current_shape[2])
        # ext_conv2: Conv2d(16, 16, k=3, s=1, p=1)
        f2_rb, s2_rb = _calc_conv_flops(s1_rb[0], 16, 3, 1, 1, s1_rb[1], s1_rb[2])
        # ext_conv3: Conv2d(16, 64, k=1, s=1, p=0)
        f3_rb, s3_rb = _calc_conv_flops(s2_rb[0], 64, 1, 1, 0, s2_rb[1], s2_rb[2])
        flops += f1_rb + f2_rb + f3_rb
        # Add non-conv FLOPs for RegularBottleneck: BN (3x), ReLU (3x), Add (1x)
        flops += current_shape[0] * current_shape[1] * current_shape[2] * (3*4 + 3*1 + 1)
    
    # Stage 2 Downsample
    # downsample2_0 (DownsamplingBottleneck): (64, 128)
    # ext_conv1: Conv2d(64, 16, k=2, s=2, p=0)
    f1, s1 = _calc_conv_flops(current_shape[0], 16, 2, 2, 0, current_shape[1], current_shape[2])
    # ext_conv2: Conv2d(16, 16, k=3, s=1, p=1)
    f2, s2 = _calc_conv_flops(s1[0], 16, 3, 1, 1, s1[1], s1[2])
    # ext_conv3: Conv2d(16, 128, k=1, s=1, p=0)
    f3, s3 = _calc_conv_flops(s2[0], 128, 1, 1, 0, s2[1], s2[2])
    flops += f1 + f2 + f3; current_shape = s3
    # Add non-conv FLOPs for DownsamplingBottleneck: BN (3x), ReLU (3x), Add (1x)
    flops += current_shape[0] * current_shape[1] * current_shape[2] * (3*4 + 3*1 + 1)
    
    # Stage 2 & 3 Bottlenecks (16 total)
    # internal_channels=32 (128//4)
    bottleneck_cfgs = [
        (1,1), (2,2), (5,2,'asymm'), (4,4), (1,1), (8,8), (5,2,'asymm'), (16,16)
    ]
    for _ in range(2): # Stages 2 and 3
        for p1, p2, *rest in bottleneck_cfgs:
            # ext_conv1: Conv2d(128, 32, k=1, s=1, p=0)
            f1, s1 = _calc_conv_flops(current_shape[0], 32, 1, 1, 0, current_shape[1], current_shape[2])
            if rest: # asymmetric
                f_main, s_main = _calc_asym_flops(s1[0], 32, p1, p2, s1[1], s1[2])
            else: # regular or dilated
                f_main, s_main = _calc_conv_flops(s1[0], 32, 3, 1, p2, s1[1], s1[2], p1)
            # ext_conv3: Conv2d(32, 128, k=1, s=1, p=0)
            f3, s3 = _calc_conv_flops(s_main[0], 128, 1, 1, 0, s_main[1], s_main[2])
            flops += f1 + f_main + f3
            # Add non-conv FLOPs for RegularBottleneck: BN (3x), ReLU (3x), Add (1x)
            flops += current_shape[0] * current_shape[1] * current_shape[2] * (3*4 + 3*1 + 1)
    
    # Stage 4 Upsample
    # upsample4_0 (UpsamplingBottleneck): (128, 64)
    # main_conv1: Conv2d(128, 64, k=1)
    f1, s1 = _calc_conv_flops(current_shape[0], 64, 1, 1, 0, current_shape[1], current_shape[2])
    # ext_conv1: Conv2d(128, 32, k=1)
    f2, s2 = _calc_conv_flops(current_shape[0], 32, 1, 1, 0, current_shape[1], current_shape[2])
    # ext_tconv1: ConvTranspose2d(32, 32, k=2, s=2)
    f3, s3 = _calc_deconv_flops(s2[0], 32, 2, 2, 0, s2[1], s2[2], 1)
    # ext_conv2: Conv2d(32, 64, k=1)
    f4, s4 = _calc_conv_flops(s3[0], 64, 1, 1, 0, s3[1], s3[2])
    flops += f1 + f2 + f3 + f4; current_shape = s4
    # Add non-conv FLOPs for UpsamplingBottleneck: BN (4x), ReLU (4x), Add (1x)
    flops += current_shape[0] * current_shape[1] * current_shape[2] * (4*4 + 4*1 + 1)
    
    # 2x RegularBottleneck (64 channels)
    for _ in range(2):
        f1_rb, s1_rb = _calc_conv_flops(current_shape[0], 16, 1, 1, 0, current_shape[1], current_shape[2])
        f2_rb, s2_rb = _calc_conv_flops(s1_rb[0], 16, 3, 1, 1, s1_rb[1], s1_rb[2])
        f3_rb, s3_rb = _calc_conv_flops(s2_rb[0], 64, 1, 1, 0, s2_rb[1], s2_rb[2])
        flops += f1_rb + f2_rb + f3_rb
        # Add non-conv FLOPs for RegularBottleneck: BN (3x), ReLU (3x), Add (1x)
        flops += current_shape[0] * current_shape[1] * current_shape[2] * (3*4 + 3*1 + 1)
    
    # Stage 5 Upsample
    # upsample5_0 (UpsamplingBottleneck): (64, 16)
    # main_conv1: Conv2d(64, 16, k=1)
    f1, s1 = _calc_conv_flops(current_shape[0], 16, 1, 1, 0, current_shape[1], current_shape[2])
    # ext_conv1: Conv2d(64, 4, k=1)
    f2, s2 = _calc_conv_flops(current_shape[0], 4, 1, 1, 0, current_shape[1], current_shape[2])
    # ext_tconv1: ConvTranspose2d(4, 4, k=2, s=2)
    f3, s3 = _calc_deconv_flops(s2[0], 4, 2, 2, 0, s2[1], s2[2], 1)
    # ext_conv2: Conv2d(4, 16, k=1)
    f4, s4 = _calc_conv_flops(s3[0], 16, 1, 1, 0, s3[1], s3[2])
    flops += f1 + f2 + f3 + f4; current_shape = s4
    # Add non-conv FLOPs for UpsamplingBottleneck: BN (4x), ReLU (4x), Add (1x)
    flops += current_shape[0] * current_shape[1] * current_shape[2] * (4*4 + 4*1 + 1)
    
    # 1x RegularBottleneck (16 channels)
    f1_rb, s1_rb = _calc_conv_flops(current_shape[0], 4, 1, 1, 0, current_shape[1], current_shape[2])
    f2_rb, s2_rb = _calc_conv_flops(s1_rb[0], 4, 3, 1, 1, s1_rb[1], s1_rb[2])
    f3_rb, s3_rb = _calc_conv_flops(s2_rb[0], 16, 1, 1, 0, s2_rb[1], s2_rb[2])
    flops += f1_rb + f2_rb + f3_rb
    # Add non-conv FLOPs for RegularBottleneck: BN (3x), ReLU (3x), Add (1x)
    flops += current_shape[0] * current_shape[1] * current_shape[2] * (3*4 + 3*1 + 1)
    
    # Final Transposed Conv
    f_final, _ = _calc_deconv_flops(current_shape[0], num_classes, 3, 2, 1, current_shape[1], current_shape[2], 1)
    flops += f_final
    return flops

# 3. Activation Memory and Latency Functions (unchanged from your file)
def get_model_parameters(model):
    return sum(p.numel() for p in model.parameters())
def calculate_enet_activation_memory(input_shape):
    total_mb, max_mb, shape = 0.0, 0.0, input_shape
    stage_cfgs = [(16, 2), (64, 2), (128, 2), (128, 1), (64, 0.5), (16, 0.5)]
    for ch, scale in stage_cfgs:
        if scale == 2: shape = (ch, shape[1]//2, shape[2]//2)
        elif scale == 0.5: shape = (ch, shape[1]*2, shape[2]*2)
        else: shape = (ch, shape[1], shape[2])
        mem = (shape[0] * shape[1] * shape[2] * 4) / (1024**2)
        total_mb += mem; max_mb = max(max_mb, mem)
    final_mem = (21 * input_shape[1] * input_shape[2] * 4) / (1024**2)
    total_mb += final_mem; max_mb = max(max_mb, final_mem)
    return total_mb, max_mb
def run_thop_measurement(sizes, num_classes):
    if not THOP_AVAILABLE:
        print("THOP not available. Skipping THOP measurements.")
        return
    print("THOP (PyTorch-OpCounter) Measurements:")
    print(f"{'Input Size':>20} | {'FLOPs (G)':>12} | {'MACs (G)':>12} | {'Parameters (M)':>15}")
    print("-" * 70)
    for inp in sizes:
        model = ENet(num_classes=21).cpu().eval()
        x = torch.randn(1, *inp)
        flops, params = profile(model, inputs=(x,), verbose=False)
        print(f"{inp[0]}x{inp[1]}x{inp[2]:<8} | {flops/1e9:11.2f} | {flops/(2*1e9):11.2f} | {params/1e6:14.2f}")

def run_ptflops_measurement(sizes, num_classes): # Added PTFLOPS
    if not PTFLOPS_AVAILABLE:
        print("PTFLOPS not available. Skipping PTFLOPS measurements.")
        return
    print("\nPTFLOPS Measurements:")
    print(f"{'Input Size':>20} | {'FLOPs (G)':>12} | {'Parameters (M)':>15}")
    print("-" * 70)
    for inp in sizes:
        try:
            model = ENet(num_classes=21).cpu().eval()
            # PTFLOPs expects input_shape (C, H, W)
            flops, params = get_model_complexity_info(model, (inp[0], inp[1], inp[2]), as_strings=False, print_per_layer_stat=False)
            print(f"{inp[0]}x{inp[1]}x{inp[2]:<8} | {flops/1e9:11.2f} | {params/1e6:14.2f}")
        except Exception as e:
            print(f"{inp[0]}x{inp[1]}x{inp[2]:<8} | {'Error':>11} | {'Error'}")
            print(f"  > PTFLOPS failed for {inp}: {e}")

def measure_latency_all_inputs(sizes, num_classes):
    print("LATENCY MEASUREMENTS (CPU) - ALL INPUT SIZES")
    print(f"{'Input Size':>20} | {'Avg Latency (ms)':>18}")
    print("-" * 42)
    device = torch.device('cpu')
    for inp in sizes:
        try: # Added try-except for robustness in latency measurements
            model = ENet(num_classes=21).to(device).eval()
            x = torch.randn(1, *inp).to(device)
            with torch.no_grad():
                for _ in range(3): model(x)
                # Corrected latency measurement: perform multiple inferences inside timing loop
                num_inferences_per_run = 10 # Increased workload per measurement
                times = []
                for _ in range(5): # Average over 5 runs
                    start_time = time.perf_counter()
                    for _ in range(num_inferences_per_run):
                        _ = model(x)
                    end_time = time.perf_counter()
                    # Calculate average time per inference for this run
                    times.append((end_time - start_time) * 1000 / num_inferences_per_run) 
            print(f"{inp[0]}x{inp[1]}x{inp[2]:<8} | {sum(times)/len(times):17.2f}")
        except Exception as e:
            print(f"{inp[0]}x{inp[1]}x{inp[2]:<8} | Error: {str(e)[:20]}")

# 4. Main Execution
if __name__ == "__main__":
    print("ENet Profiling Script - Complete Analysis (FIXED)")
    print("="*60)
    NUM_CLASSES = 21
    input_sizes = [(3, 640, 360), (3, 1280, 720), (3, 1360, 760),
                   (3, 1600, 900), (3, 1920, 1080), (3, 2048, 1152),
                   (3, 2560, 1440), (3, 3840, 2160)]
    model = ENet(num_classes=NUM_CLASSES)
    total_params = get_model_parameters(model)
    
    print("THEORETICAL ANALYSIS FOR ENET (CORRECTED)")
    print(f"{'Input Size':>15} | {'FLOPs (G)':>10} | {'Params (M)':>10} | {'Total Act (MB)':>14} | {'Peak Act (MB)':>13}")
    print("-" * 80)
    for inp in input_sizes:
        theoretical_flops = calculate_enet_theoretical_flops(inp, num_classes=NUM_CLASSES)
        total_act, peak_act = calculate_enet_activation_memory(inp)
        print(f"{inp[0]}x{inp[1]}x{inp[2]:<5} | {theoretical_flops/1e9:10.2f} | {total_params/1e6:10.2f} | {total_act:14.2f} | {peak_act:13.2f}")
    print("\n")
    run_thop_measurement(input_sizes, num_classes=NUM_CLASSES)
    print("\n")
    run_ptflops_measurement(input_sizes, num_classes=NUM_CLASSES) # Call PTFLOPs
    print("\n")
    measure_latency_all_inputs(input_sizes, num_classes=NUM_CLASSES)
