#!/usr/bin/env python3
"""
SegFormer-B5 Profiling Script:
Calculates Theoretical FLOPs, PTFLOPS (MACs), Parameters, Latency, and Activation Size
for the SegFormer-B5 model across various input resolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import warnings

try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    PTFLOPS_AVAILABLE = False
    print("Warning: 'ptflops' not installed; PTFLOPS will be skipped.")

# ------------------------------------------------------------------------------
# Model Definition
# ------------------------------------------------------------------------------

class OverlapPatchMerging(nn.Module):
    def __init__(self, in_ch, out_ch, patch_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, patch_size, stride, patch_size//2, bias=False)
        self.norm = nn.LayerNorm(out_ch)
    def forward(self, x):
        B,C,H,W = x.shape
        x = self.conv(x)
        _,C2,H2,W2 = x.shape
        x = x.flatten(2).transpose(1,2)
        x = self.norm(x)
        return x, H2, W2

class EfficientSelfAttention(nn.Module):
    def __init__(self, C, heads, reduction):
        super().__init__()
        self.heads = heads
        self.scale = (C//heads)**-0.5
        self.q = nn.Linear(C, C)
        self.kv = nn.Linear(C, 2*C)
        self.proj = nn.Linear(C, C)
        if reduction>1:
            self.sr = nn.Conv2d(C, C, reduction, reduction)
            self.norm = nn.LayerNorm(C)
        else:
            self.sr = None
    def forward(self, x, H, W):
        B,N,C = x.shape
        q = self.q(x).view(B,N,self.heads,C//self.heads).permute(0,2,1,3)
        if self.sr:
            x2 = x.permute(0,2,1).view(B,C,H,W)
            x2 = self.sr(x2).view(B,C,-1).permute(0,2,1)
            x2 = self.norm(x2)
        else:
            x2 = x
        kv = self.kv(x2).view(B,-1,2,self.heads,C//self.heads).permute(2,0,3,1,4)
        k,v = kv[0], kv[1]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        return self.proj(x)

class MixFFN(nn.Module):
    def __init__(self, C, hidden):
        super().__init__()
        self.fc1 = nn.Linear(C, hidden)
        self.dw  = nn.Conv2d(hidden, hidden, 3,1,1,groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, C)
    def forward(self, x, H, W):
        B,N,C = x.shape
        x = self.fc1(x)
        x = x.transpose(1,2).view(B,-1,H,W)
        x = self.dw(x)
        x = x.flatten(2).transpose(1,2)
        x = self.act(x)
        return self.fc2(x)

class SegFormerBlock(nn.Module):
    def __init__(self, C, heads, mlp_ratio, reduction):
        super().__init__()
        self.norm1 = nn.LayerNorm(C)
        self.attn  = EfficientSelfAttention(C, heads, reduction)
        self.norm2 = nn.LayerNorm(C)
        self.ffn   = MixFFN(C, int(C*mlp_ratio))
    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(   self.norm2(x), H, W)
        return x

class SegFormerB5(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        ws=[64,128,320,512]
        ds=[3,6,40,3]
        heads=[1,2,5,8]
        rr=[8,4,2,1]
        ps=[7,3,3,3]
        strides=[4,2,2,2]
        mlp_ratio=4
        dec_ch=768

        self.stages = nn.ModuleList()
        for i in range(4):
            layers = []
            layers.append(OverlapPatchMerging(
                in_ch = 3 if i==0 else ws[i-1],
                out_ch= ws[i],
                patch_size= ps[i],
                stride= strides[i]
            ))
            for _ in range(ds[i]):
                layers.append(SegFormerBlock(ws[i], heads[i], mlp_ratio, rr[i]))
            self.stages.append(nn.Sequential(*layers))

        self.norms = nn.ModuleList([nn.LayerNorm(w) for w in ws])
        self.decoder_mlps = nn.ModuleList([nn.Conv2d(w, dec_ch, 1, bias=False) for w in ws])
        self.head = nn.Conv2d(dec_ch*4, num_classes, 1)

    def forward(self, x):
        B,C,H,W = x.shape
        feats = []
        for i, stage in enumerate(self.stages):
            x, H2, W2 = stage[0](x)
            for blk in stage[1:]:
                x = blk(x, H2, W2)
            x = self.norms[i](x)
            # FIX: use reshape instead of view
            x_sp = x.transpose(1,2).reshape(B, H2, W2, -1).permute(0,3,1,2)
            feats.append(x_sp)
            x = x_sp

        outs = []
        tgt = feats[0].shape[2:]
        for mlp, feat in zip(self.decoder_mlps, feats):
            y = mlp(feat)
            if y.shape[2:] != tgt:
                y = F.interpolate(y, size=tgt, mode='bilinear', align_corners=False)
            outs.append(y)

        x = torch.cat(outs,1)
        x = self.head(x)
        if x.shape[2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x

# ------------------------------------------------------------------------------
# Profiling Utilities
# ------------------------------------------------------------------------------

def _conv_flops(cin, cout, h, w, k, stride=1, pad=0):
    hout = (h+2*pad-k)//stride + 1
    wout = (w+2*pad-k)//stride + 1
    return 2*cin*cout*k*k*hout*wout, hout, wout

def calculate_theoretical_flops(inp, num_classes=19):
    c,h,w = inp
    ws=[64,128,320,512]
    ds=[3,6,40,3]
    heads=[1,2,5,8]
    rr=[8,4,2,1]
    ps=[7,3,3,3]
    strides=[4,2,2,2]
    mlp_ratio=4
    dec_ch=768

    fl=0
    H,W=h,w
    for i in range(4):
        inch = c if i==0 else ws[i-1]
        outch= ws[i]
        p, H, W = _conv_flops(inch,outch,H,W,ps[i],strides[i],ps[i]//2); fl+=p
        for _ in range(ds[i]):
            seq=H*W
            fl += 2*seq*outch*outch*2             # Q,K,V & output proj
            fl += 2*heads[i]*seq*seq*(outch//heads[i])
            hid = outch*mlp_ratio
            fl += 2*seq*outch*hid
            fl += hid*9*H*W
            fl += 2*seq*hid*outch

    fh,fw = h//4,w//4
    for w_i in ws:
        fl += 2*w_i*dec_ch*fh*fw
        fl += 7*dec_ch*fh*fw
    fl += 2*(dec_ch*4)*num_classes*fh*fw
    fl += 7*num_classes*h*w
    return fl

def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def measure_latency(m,inp,runs=3,rep=5):
    dev=torch.device('cpu'); m.to(dev).eval()
    x=torch.randn(1,*inp).to(dev)
    with torch.no_grad():
        for _ in range(2): m(x)
    ts=[]
    with torch.no_grad():
        for _ in range(runs):
            s=time.perf_counter()
            for _ in range(rep): m(x)
            e=time.perf_counter(); ts.append((e-s)*1000/rep)
    return sum(ts)/len(ts)

class ActivationTracker(nn.Module):
    def __init__(self, net):
        super().__init__(); self.net=net; self.act_bytes=0
    def reset(self): self.act_bytes=0
    def add(self,o): 
        if isinstance(o,torch.Tensor): self.act_bytes+=o.numel()*4
    def forward(self,x):
        self.reset()
        hooks=[m.register_forward_hook(lambda m,inp,out: self.add(out)) for m in self.net.modules()]
        y=self.net(x)
        for h in hooks: h.remove()
        return y

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__=='__main__':
    inputs=[(3,640,360),(3,1280,720),(3,1360,760),(3,1600,900),
            (3,1920,1080),(3,2048,1152),(3,2560,1440),(3,3840,2160)]
    model=SegFormerB5(num_classes=19)
    tracker=ActivationTracker(model)
    params_m=count_parameters(model)/1e6

    print(f"{'Input':>18} | {'TheoFLOPs(G)':>12} | {'PTFLOPs(G)':>12} | {'Params(M)':>10} | {'Latency(ms)':>12} | {'Act(MB)':>8}")
    print("-"*85)
    for inp in inputs:
        theo=calculate_theoretical_flops(inp,19)/1e9
        if PTFLOPS_AVAILABLE:
            clean=SegFormerB5(19)
            fl,_=get_model_complexity_info(clean,inp,as_strings=False,print_per_layer_stat=False,verbose=False)
            pt=fl/1e9
        else:
            pt="N/A"
        lat=measure_latency(model,inp)
        _=tracker(torch.randn(1,*inp))
        act=tracker.act_bytes/1024**2
        print(f"{str(inp):>18} | {theo:12.2f} | {str(pt):>12} | {params_m:10.2f} | {lat:12.2f} | {act:8.2f}")
