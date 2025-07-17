#!/usr/bin/env python3
"""
ERFNet Profiling Script – FINAL FIXED VERSION
- Resolves the persistent BatchNorm channel mismatch error in InitialBlock.
- Implements the correct channel split for the InitialBlock.
- Profiles theoretical FLOPs, THOP measurements, parameters,
  activation memory, and latency on CPU for compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import warnings
warnings.filterwarnings("ignore")

# Try to import profiling libraries with graceful fallbacks
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

# 1. Building Blocks

class ConvBNAct(nn.Sequential):
    """Convolution -> BatchNorm -> ReLU"""
    def __init__(self, in_c, out_c, k, s=1, ph=0, pw=0, dil=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, stride=s, padding=(ph, pw), dilation=dil, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

class DeConvBNAct(nn.Sequential):
    """Transposed Convolution -> BatchNorm -> ReLU"""
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.ConvTranspose2d(in_c, out_c, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

class InitialBlock(nn.Module):
    """Initial downsampling block with CORRECT BatchNorm channel calculation."""
    def __init__(self, in_c, out_c):
        super().__init__()
        # CRITICAL FIX: The conv branch outputs (out_c - in_c) channels.
        # This way, (out_c - in_c) from conv + in_c from pool = out_c channels for BN.
        self.conv = nn.Conv2d(in_c, out_c - in_c, 3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.bn   = nn.BatchNorm2d(out_c) # BatchNorm now correctly receives 'out_c' channels
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c = self.conv(x)
        p = self.pool(x)
        out = torch.cat([c, p], dim=1)
        return self.relu(self.bn(out))

class NonBt1DBlock(nn.Module):
    """Non-bottleneck 1D block with residual connection"""
    def __init__(self, ch, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(ch, ch, (3,1), ph=1, pw=0),
            ConvBNAct(ch, ch, (1,3), ph=0, pw=1),
            ConvBNAct(ch, ch, (3,1), ph=dilation, pw=0, dil=dilation),
            ConvBNAct(ch, ch, (1,3), ph=0, pw=dilation, dil=dilation)
        )
        self.bnact = nn.Sequential(nn.BatchNorm2d(ch), nn.ReLU(True))

    def forward(self, x):
        res = x
        x = self.conv(x)
        return self.bnact(x + res)

# 2. ERFNet Model

class ERFNet(nn.Module):
    def __init__(self, num_class=1):
        super().__init__()
        # InitialBlock now correctly calculates BN channels based on in_c and out_c
        self.l1  = InitialBlock(3, 16) 
        self.l2  = InitialBlock(16, 64)
        self.l3  = nn.Sequential(*[NonBt1DBlock(64) for _ in range(5)])
        self.l8  = InitialBlock(64, 128)
        self.l9  = nn.Sequential(*[NonBt1DBlock(128, d) for d in [2,4,8,16,2,4,8,16]])
        self.l17 = DeConvBNAct(128, 64)
        self.l18 = nn.Sequential(*[NonBt1DBlock(64) for _ in range(2)])
        self.l20 = DeConvBNAct(64, 16)
        self.l21 = nn.Sequential(*[NonBt1DBlock(16) for _ in range(2)])
        self.l23 = DeConvBNAct(16, num_class)

    def forward(self, x):
        x = self.l1(x); x = self.l2(x)
        x = self.l3(x); x = self.l8(x)
        x = self.l9(x); x = self.l17(x)
        x = self.l18(x); x = self.l20(x)
        x = self.l21(x); return self.l23(x)

# 3. Theoretical FLOPs

def conv_flops(shape, out_ch, k, s=1, ph=0, pw=0, dil=1):
    in_ch, h, w = shape
    kh = k if isinstance(k, int) else k[0]
    kw = k if isinstance(k, int) else k[1]
    eff_kh = kh + (kh - 1) * (dil - 1)
    eff_kw = kw + (kw - 1) * (dil - 1)
    oh = (h + 2 * ph - eff_kh) // s + 1
    ow = (w + 2 * pw - eff_kw) // s + 1
    fl = in_ch * out_ch * kh * kw * oh * ow * 2
    return fl, (out_ch, oh, ow)

def deconv_flops(shape, out_ch, k, s=1, p=0, op=0):
    in_ch, h, w = shape
    oh = (h-1)*s - 2*p + k + op
    ow = (w-1)*s - 2*p + k + op
    fl = in_ch * out_ch * k * k * h * w * 2
    return fl, (out_ch, oh, ow)

def theoretical_flops(inp, num_class=1): # FIXED: Added num_class parameter
    fl, shape = 0, inp
    
    # Initial downsamplers
    # l1: InitialBlock(3, 16)
    f_conv, s_conv = conv_flops(shape, 16 - 3, 3, 2, 1, 1) # conv branch FLOPs
    fl += f_conv
    shape = (16, s_conv[1], s_conv[2]) # Final output shape of InitialBlock
    
    # l2: InitialBlock(16, 64)
    f_conv, s_conv = conv_flops(shape, 64 - 16, 3, 2, 1, 1) # conv branch FLOPs
    fl += f_conv
    shape = (64, s_conv[1], s_conv[2]) # Final output shape of InitialBlock
    
    # l8: InitialBlock(64, 128)
    f_conv, s_conv = conv_flops(shape, 128 - 64, 3, 2, 1, 1) # conv branch FLOPs
    fl += f_conv
    shape = (128, s_conv[1], s_conv[2]) # Final output shape of InitialBlock

    # Non-bottleneck sequences
    cfg = [(64, 5, [1]*5), (128, 8, [2,4,8,16,2,4,8,16]), (64, 2, [1]*2), (16, 2, [1]*2)]
    
    for out_ch, cnt, dils in cfg:
        # Before entering the sequential block, update the current shape to match
        # the input to this block. The 'shape' variable is maintained globally
        # in this theoretical function.
        
        # These factorized convs have internal channels 'ch' where ch=64 or 128 or 16
        # The outputs are also 'ch'.
        
        for d in dils:
            # NonBt1DBlock: conv -> bnact (x+res)
            # Internal conv block: 4 conv layers
            
            # Conv1: ConvBNAct(ch, ch, (3,1), ph=1, pw=0)
            f1, s1 = conv_flops(shape, shape[0], (3,1), 1, 1, 0)
            
            # Conv2: ConvBNAct(s1, ch, (1,3), ph=0, pw=1)
            f2, s2 = conv_flops(s1, shape[0], (1,3), 1, 0, 1)
            
            # Conv3: ConvBNAct(s2, ch, (3,1), ph=dilation, pw=0, dil=dilation)
            f3, s3 = conv_flops(s2, shape[0], (3,1), 1, d, 0, d)
            
            # Conv4: ConvBNAct(s3, ch, (1,3), ph=0, pw=dilation, dil=dilation)
            f4, s4 = conv_flops(s3, shape[0], (1,3), 1, 0, d, d)
            
            fl += f1+f2+f3+f4
            # Shape remains the same after NonBt1DBlock (ch, H, W)
            shape = (shape[0], s4[1], s4[2])


    # Deconvolutional stages
    # l17: DeConvBNAct(128, 64)
    f_deconv, shape = deconv_flops(shape, 64, 3, 2, 1, 1); fl += f_deconv
    
    # l18-19: NonBt1DBlock (64) x 2
    for _ in range(2):
        d=1 # Default dilation
        f1,s1 = conv_flops(shape, shape[0], (3,1), 1, 1, 0)
        f2,s2 = conv_flops(s1, shape[0], (1,3), 1, 0, 1)
        f3,s3 = conv_flops(s2, shape[0], (3,1), 1, d, 0, d)
        f4,s4 = conv_flops(s3, shape[0], (1,3), 1, 0, d, d)
        fl += f1+f2+f3+f4
        shape = (shape[0], s4[1], s4[2]) # Shape remains constant within NonBt1DBlock

    # l20: DeConvBNAct(64, 16)
    f_deconv, shape = deconv_flops(shape, 16, 3, 2, 1, 1); fl += f_deconv
    
    # l21-22: NonBt1DBlock (16) x 2
    for _ in range(2):
        d=1 # Default dilation
        f1,s1 = conv_flops(shape, shape[0], (3,1), 1, 1, 0)
        f2,s2 = conv_flops(s1, shape[0], (1,3), 1, 0, 1)
        f3,s3 = conv_flops(s2, shape[0], (3,1), 1, d, 0, d)
        f4,s4 = conv_flops(s3, shape[0], (1,3), 1, 0, d, d)
        fl += f1+f2+f3+f4
        shape = (shape[0], s4[1], s4[2]) # Shape remains constant within NonBt1DBlock
    
    # l23: DeConvBNAct(16, num_class) - final output
    f_deconv, _ = deconv_flops(shape, num_class, 3, 2, 1, 1); fl += f_deconv
    
    return fl

# 4. Activation Memory

def activation_memory(inp):
    total = peak = 0.0
    c, h, w = inp
    
    # Trace channels and spatial dimensions accurately through the model's forward pass
    current_h, current_w = h, w
    
    # InitialBlock (l1): (3,H,W) -> (16, H/2, W/2)
    current_h, current_w = current_h // 2, current_w // 2
    current_c = 16
    mem = current_c * current_h * current_w * 4 / 1024**2
    total += mem; peak = max(peak, mem)
    
    # l2: InitialBlock(16, 64) -> (64, H/4, W/4)
    current_h, current_w = current_h // 2, current_w // 2
    current_c = 64
    mem = current_c * current_h * current_w * 4 / 1024**2
    total += mem; peak = max(peak, mem)
    
    # l3-7: NonBt1DBlock (64) x 5 -> (64, H/4, W/4)
    for _ in range(5):
        mem = current_c * current_h * current_w * 4 / 1024**2
        total += mem; peak = max(peak, mem)
    
    # l8: InitialBlock(64, 128) -> (128, H/8, W/8)
    current_h, current_w = current_h // 2, current_w // 2
    current_c = 128
    mem = current_c * current_h * current_w * 4 / 1024**2
    total += mem; peak = max(peak, mem)
    
    # l9-16: NonBt1DBlock (128) x 8 -> (128, H/8, W/8)
    for _ in range(8):
        mem = current_c * current_h * current_w * 4 / 1024**2
        total += mem; peak = max(peak, mem)
        
    # l17: DeConvBNAct(128, 64) -> (64, H/4, W/4) (upsamples)
    current_h, current_w = current_h * 2, current_w * 2
    current_c = 64
    mem = current_c * current_h * current_w * 4 / 1024**2
    total += mem; peak = max(peak, mem)
    
    # l18-19: NonBt1DBlock (64) x 2 -> (64, H/4, W/4)
    for _ in range(2):
        mem = current_c * current_h * current_w * 4 / 1024**2
        total += mem; peak = max(peak, mem)
        
    # l20: DeConvBNAct(64, 16) -> (16, H/2, W/2) (upsamples)
    current_h, current_w = current_h * 2, current_w * 2
    current_c = 16
    mem = current_c * current_h * current_w * 4 / 1024**2
    total += mem; peak = max(peak, mem)
    
    # l21-22: NonBt1DBlock (16) x 2 -> (16, H/2, W/2)
    for _ in range(2):
        mem = current_c * current_h * current_w * 4 / 1024**2
        total += mem; peak = max(peak, mem)
        
    # l23: DeConvBNAct(16, num_class) -> (num_class, H, W) (final upsample)
    current_h, current_w = current_h * 2, current_w * 2
    final_num_class = 1 # Assuming num_class=1 for default profiling
    mem = final_num_class * current_h * current_w * 4 / 1024**2
    total += mem; peak = max(peak, mem)

    return total, peak

# 5. THOP Profiling

def thop_profile(sizes, num_class=1):
    print("THOP Profiling:")
    print(f"{'Input':>12}|{'FLOPs(G)':>10}|{'MACs(G)':>10}|{'Params(M)':>10}")
    for inp in sizes:
        model = ERFNet(num_class).cpu().eval()
        x = torch.randn(1, *inp)
        fl, params = profile(model, inputs=(x,), verbose=False)
        print(f"{str(inp):>12}|{fl/1e9:10.2f}|{(fl/2)/1e9:10.2f}|{params/1e6:10.2f}")

# 6. Latency Measurement

def measure_latency(sizes, num_class=1):
    print("Latency (ms):")
    print(f"{'Input':>12}|{'Avg':>8}")
    for inp in sizes:
        model = ERFNet(num_class).cpu().eval()
        x = torch.randn(1,*inp)
        for _ in range(2): model(x)
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            model(x)
            t1 = time.perf_counter()
            ts.append((t1 - t0) * 1000)
        print(f"{str(inp):>12}|{sum(ts)/len(ts):8.2f}")

# 7. Main Execution

if __name__ == "__main__":
    sizes = [(3, 640, 360),
             (3, 1280, 720),
             (3, 1360, 760),
             (3, 1600, 900),
             (3, 1920, 1080),
             (3, 2048, 1152),
             (3, 2560, 1440),
             (3, 3840, 2160)
]
    NUM_CLASSES = 1

    print("Theoretical FLOPs:")
    for inp in sizes:
        print(f"{inp}: {theoretical_flops(inp, num_class=NUM_CLASSES)/1e9:.2f} G")

    print("\nActivation Memory:")
    for inp in sizes:
        tot, pk = activation_memory(inp)
        print(f"{inp}: Total {tot:.2f} MB, Peak {pk:.2f} MB")

    print()
    thop_profile(sizes, num_class=NUM_CLASSES)

    print()
    measure_latency(sizes, num_class=NUM_CLASSES)
