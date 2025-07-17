#!/usr/bin/env python3
"""
RTFormer-Slim Profiling Script:
Calculates Theoretical FLOPs, PTFLOPS (MACs), Parameters, Latency, and Activation Size
for the RTFormer-Slim semantic segmentation model across various input resolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import warnings

# Optional profiling library
try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    PTFLOPS_AVAILABLE = False
    print("Warning: 'ptflops' not installed; PTFLOPS will be skipped.")

# ------------------------------------------------------------------------------
# 1. GPU-Friendly Attention (GFA)
# ------------------------------------------------------------------------------

class GFA(nn.Module):
    def __init__(self, dim, reduction=1):
        super().__init__()
        self.reduction = reduction
        if reduction > 1:
            self.sr = nn.Conv2d(dim, dim, reduction, reduction)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
        self.q   = nn.Linear(dim, dim)
        self.kv  = nn.Linear(dim, dim * 2)
        self.proj= nn.Linear(dim, dim)
        self.scale = (dim) ** -0.5

    def forward(self, x, H, W):
        B, N, C = x.shape
        # Query
        q = self.q(x).reshape(B, N, C).unsqueeze(1)  # B,1,N,C
        # Key/Value
        if self.sr:
            x2 = x.transpose(1,2).view(B, C, H, W)
            x2 = self.sr(x2).view(B, C, -1).transpose(1,2)
            x2 = self.norm(x2)
        else:
            x2 = x
        kv = self.kv(x2).reshape(B, -1, 2, C).permute(2,0,1,3)
        k, v = kv[0], kv[1]  # B,N_kv,C

        # Attention
        attn = (q @ k.transpose(-2,-1)) * self.scale  # B,1,N,N_kv
        attn = attn.softmax(dim=-1)
        out = (attn @ v).squeeze(1)  # B,N,C

        return self.proj(out)

# ------------------------------------------------------------------------------
# 2. RTFormer Transformer Block
# ------------------------------------------------------------------------------

class RTFormerBlock(nn.Module):
    def __init__(self, dim, reduction):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = GFA(dim, reduction)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x))
        return x

# ------------------------------------------------------------------------------
# 3. Overlap Patch Merging
# ------------------------------------------------------------------------------

class OverlapPatchMerging(nn.Module):
    def __init__(self, in_ch, out_ch, patch_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, patch_size, stride, patch_size//2, bias=False)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)                               # B, C2, H2, W2
        B, C2, H2, W2 = x.shape
        x = x.flatten(2).transpose(1,2)                # B, H2*W2, C2
        x = self.norm(x)
        return x, H2, W2

# ------------------------------------------------------------------------------
# 4. Backbone: RTFormer-Slim
# ------------------------------------------------------------------------------

class RTFormerBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        widths    = [32, 64, 128, 256]
        depths    = [2, 2, 2, 2]
        reductions= [8, 4, 2, 1]
        patch_sz  = [3, 3, 3, 3]
        strides   = [1, 2, 2, 2]

        self.stages = nn.ModuleList()
        self.norms  = nn.ModuleList()
        in_ch = 3
        for w, d, r, ps, st in zip(widths, depths, reductions, patch_sz, strides):
            layers = [OverlapPatchMerging(in_ch, w, ps, st)]
            for _ in range(d):
                layers.append(RTFormerBlock(w, r))
            self.stages.append(nn.Sequential(*layers))
            self.norms.append(nn.LayerNorm(w))
            in_ch = w

    def forward(self, x):
        feats = []
        for stage, norm in zip(self.stages, self.norms):
            x, H, W = stage[0](x)
            for blk in stage[1:]:
                x = blk(x, H, W)
            x = norm(x)
            B, N, C = x.shape
            x_sp = x.transpose(1,2).view(B, C, H, W)
            feats.append(x_sp)
            x = x_sp
        return feats

# ------------------------------------------------------------------------------
# 5. Dual Attention Pyramid Pooling Module (DAPPM)
# ------------------------------------------------------------------------------

class DAPPM(nn.Module):
    def __init__(self, in_ch, mid_ch):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU()
        )
        self.scale2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU()
        )
        self.scale3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU()
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU()
        )
        self.process = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU()
        )
        self.compress = nn.Sequential(
            nn.Conv2d(mid_ch*5, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[2:]
        x0 = self.process(x)
        x1 = F.interpolate(self.scale1(x), size, mode='bilinear', align_corners=False)
        x2 = F.interpolate(self.scale2(x), size, mode='bilinear', align_corners=False)
        x3 = F.interpolate(self.scale3(x), size, mode='bilinear', align_corners=False)
        x4 = F.interpolate(self.scale4(x), size, mode='bilinear', align_corners=False)
        return self.compress(torch.cat([x0, x1, x2, x3, x4], 1))

# ------------------------------------------------------------------------------
# 6. Segmentation Head
# ------------------------------------------------------------------------------

class RTFormerHead(nn.Module):
    def __init__(self, in_chs=[32,64,128,256], mid_ch=128, num_classes=19):
        super().__init__()
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ic, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch), nn.ReLU()
            ) for ic in in_chs
        ])
        # FIX: DAPPM in_ch = mid_ch * len(in_chs)
        self.dappm = DAPPM(mid_ch * len(in_chs), mid_ch)
        self.final = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(mid_ch, num_classes, 1)
        )

    def forward(self, feats):
        size = feats[0].shape[2:]
        outs = []
        for mlp, f in zip(self.mlps, feats):
            y = mlp(f)
            if y.shape[2:] != size:
                y = F.interpolate(y, size, mode='bilinear', align_corners=False)
            outs.append(y)
        x = torch.cat(outs, 1)
        x = self.dappm(x)
        return self.final(x)

# ------------------------------------------------------------------------------
# 7. RTFormer-Slim Model
# ------------------------------------------------------------------------------

class RTFormerSlim(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        self.backbone = RTFormerBackbone()
        self.head     = RTFormerHead(num_classes=num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

# ------------------------------------------------------------------------------
# 8. Profiling Utilities
# ------------------------------------------------------------------------------

def _conv_flops(cin, cout, h, w, k, stride=1, pad=0):
    hout = (h + 2*pad - k)//stride + 1
    wout = (w + 2*pad - k)//stride + 1
    return 2 * cin * cout * k * k * hout * wout, hout, wout

def calculate_theoretical_flops(inp, num_classes=19):
    c,h,w = inp
    widths    = [32,64,128,256]
    depths    = [2,2,2,2]
    patch_sz  = [3,3,3,3]
    strides   = [1,2,2,2]
    reductions= [8,4,2,1]
    mlp_ratio = 4
    mid_ch    = 128

    fl = 0
    H, W = h, w
    # Backbone
    for i, w_i in enumerate(widths):
        inch = c if i==0 else widths[i-1]
        pfl, H, W = _conv_flops(inch, w_i, H, W, patch_sz[i], strides[i], patch_sz[i]//2)
        fl += pfl
        seq = H * W
        # Transformer blocks
        for _ in range(depths[i]):
            # QKV + proj
            fl += 4 * seq * w_i * w_i
            # Attention matmul (approx)
            fl += 2 * seq * seq * w_i
            # FFN
            hid = w_i * mlp_ratio
            fl += 2 * seq * w_i * hid
            fl += hid * 9 * H * W
            fl += 2 * seq * hid * w_i

    # Head mlps + DAPPM + final conv + upsample
    fh, fw = H, W
    # MLP projections + upsampling
    for w_i in widths:
        f, _, _ = _conv_flops(w_i, mid_ch, fh, fw, 1)
        fl += f
        fl += 7 * mid_ch * fh * fw

    # DAPPM pooling + convs + upsample
    # approximate as negligible relative

    # Final conv
    f, _, _ = _conv_flops(mid_ch, num_classes, fh, fw, 1)
    fl += f
    fl += 7 * num_classes * h * w

    return fl

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_latency(model, inp, runs=3, repeats=5):
    device = torch.device('cpu')
    model.to(device).eval()
    x = torch.randn(1, *inp).to(device)
    with torch.no_grad():
        for _ in range(2): model(x)
    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            for _ in range(repeats):
                model(x)
            end = time.perf_counter()
            times.append((end - start) * 1000 / repeats)
    return sum(times) / len(times)

class ActivationTracker(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.act = 0
    def reset(self):
        self.act = 0
    def add(self, out):
        if isinstance(out, torch.Tensor):
            self.act += out.numel() * 4
    def forward(self, x):
        self.reset()
        hooks = [m.register_forward_hook(lambda m, inp, out: self.add(out))
                 for m in self.net.modules()]
        y = self.net(x)
        for h in hooks: h.remove()
        return y

# ------------------------------------------------------------------------------
# 9. Main Execution
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    RESOLUTIONS = [
        (3, 640, 360), (3, 1280, 720), (3, 1360, 760),
        (3, 1600, 900), (3, 1920, 1080), (3, 2048, 1152),
        (3, 2560, 1440), (3, 3840, 2160)
    ]
    model = RTFormerSlim(num_classes=19)
    tracker = ActivationTracker(model)
    params_m = count_parameters(model) / 1e6

    print(f"{'Input':>18} | {'TheoFLOPs(G)':>12} | {'PTFLOPs(G)':>12} | "
          f"{'Params(M)':>10} | {'Latency(ms)':>12} | {'Act(MB)':>8}")
    print("-" * 85)

    for inp in RESOLUTIONS:
        theo = calculate_theoretical_flops(inp, 19) / 1e9
        if PTFLOPS_AVAILABLE:
            clean = RTFormerSlim(num_classes=19)
            flops, _ = get_model_complexity_info(
                clean, inp, as_strings=False,
                print_per_layer_stat=False, verbose=False)
            pt = flops / 1e9
        else:
            pt = "N/A"
        lat = measure_latency(model, inp)
        _ = tracker(torch.randn(1, *inp))
        act = tracker.act / (1024**2)

        print(f"{str(inp):>18} | {theo:12.2f} | {str(pt):>12} | {params_m:10.2f} | "
              f"{lat:12.2f} | {act:8.2f}")
