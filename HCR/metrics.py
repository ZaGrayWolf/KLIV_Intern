#!/usr/bin/env python3
"""
HRNet-W48 + OCR profiling script (fixed config with DATASET.NUM_CLASSES)
=======================================================================
Produces a table with:

    • “Theoretical” FLOPs  (2 × ptflops MACs)
    • ptflops MACs
    • Trainable parameters (M)
    • CPU latency  (ms / image)
    • Activation-memory footprint (MB, fp32)

Test resolutions can be edited in INPUT_SIZES.
"""

import gc, time, warnings, sys, importlib
from pathlib import Path

import torch
import torch.nn as nn

# ---- PATCH numpy to provide np.int for legacy HRNet code ----
import numpy as np
if not hasattr(np, 'int'):
    np.int = int

# ────────────────────────────────────────────────────────────────────
# 0.  Load HRNet-W48 + OCR model
# ────────────────────────────────────────────────────────────────────
MODULE_FILE = "hrnet_w48_ocr_full.py"
if not Path(MODULE_FILE).is_file():
    sys.exit(f"\n‼️  Expected {MODULE_FILE} in the current directory – aborting.\n")

spec = importlib.util.spec_from_file_location("hrnet_w48_ocr", MODULE_FILE)
hrnet_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hrnet_module)

# --- Minimal, self-contained HRNet-W48 + OCR config object -----------
EXTRA = {
    "STAGE1": {
        "NUM_MODULES": 1, "NUM_BRANCHES": 1,
        "NUM_BLOCKS": [4], "NUM_CHANNELS": [64],
        "BLOCK": "BOTTLENECK", "FUSE_METHOD": "SUM"
    },
    "STAGE2": {
        "NUM_MODULES": 1, "NUM_BRANCHES": 2,
        "NUM_BLOCKS": [4, 4], "NUM_CHANNELS": [48, 96],
        "BLOCK": "BASIC", "FUSE_METHOD": "SUM"
    },
    "STAGE3": {
        "NUM_MODULES": 4, "NUM_BRANCHES": 3,
        "NUM_BLOCKS": [4, 4, 4], "NUM_CHANNELS": [48, 96, 192],
        "BLOCK": "BASIC", "FUSE_METHOD": "SUM"
    },
    "STAGE4": {
        "NUM_MODULES": 3, "NUM_BRANCHES": 4,
        "NUM_BLOCKS": [4, 4, 4, 4], "NUM_CHANNELS": [48, 96, 192, 384],
        "BLOCK": "BASIC", "FUSE_METHOD": "SUM"
    },
    "FINAL_CONV_KERNEL": 1,
    "WITH_HEAD": True,
    "OCR": {"MID_CHANNELS": 512, "KEY_CHANNELS": 256, "DROPOUT": 0.05},
}

from types import SimpleNamespace

class Config:
    class MODEL:
        EXTRA = EXTRA
        ALIGN_CORNERS = False
        OCR = SimpleNamespace(**EXTRA["OCR"])
    class DATASET:
        NUM_CLASSES = 19

config = Config()
model = hrnet_module.HighResolutionNet(config, num_classes=19)

# ────────────────────────────────────────────────────────────────────
# 1.  Optional ptflops
# ────────────────────────────────────────────────────────────────────
try:
    from ptflops import get_model_complexity_info
    PTFLOPS = True
except ImportError:
    warnings.warn("ptflops not found – MAC column will show N/A")
    PTFLOPS = False

# ────────────────────────────────────────────────────────────────────
# 2.  Helpers
# ────────────────────────────────────────────────────────────────────
def count_params(net: nn.Module) -> float:
    return sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6

class ActivationTracker(nn.Module):
    """Accumulate activation-memory (bytes) during a forward pass."""
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.bytes = 0
    def _hook(self, _m, _i, o):
        if torch.is_tensor(o):
            self.bytes += o.numel() * 4
        elif isinstance(o, (list, tuple)):
            for t in o:
                if torch.is_tensor(t):
                    self.bytes += t.numel() * 4
    def forward(self, x):
        self.bytes = 0
        handles = [m.register_forward_hook(self._hook) for m in self.net.modules()]
        with torch.no_grad():
            _ = self.net(x)
        for h in handles:
            h.remove()
        return self.bytes

@torch.no_grad()
def latency_ms(net: nn.Module, size, warm=1, runs=2, reps=1):
    dev = torch.device("cpu")
    net.to(dev).eval()
    dummy = torch.randn(1, *size, device=dev)
    for _ in range(warm):
        _ = net(dummy)
    t0 = time.perf_counter()
    for _ in range(runs * reps):
        _ = net(dummy)
    elapsed = time.perf_counter() - t0
    return (elapsed / (runs * reps)) * 1000

# ────────────────────────────────────────────────────────────────────
# 3.  Profiling loop
# ────────────────────────────────────────────────────────────────────
INPUT_SIZES = [
    (3, 640, 360),
    (3, 1280, 720),
    (3, 1360, 760),
    (3, 1600, 900),
    (3, 1920, 1080),
    (3, 2048, 1152),
    (3, 2560, 1440),
    (3, 3840, 2160),
]

total_params = count_params(model)

print("HRNet-W48 + OCR Profiling")
print(f"{'Input':>18} | {'TheoFLOPs(G)':>13} | {'MACs(G)':>10} | "
      f"{'Params(M)':>9} | {'Latency(ms)':>12} | {'Act(MB)':>8}")
print("-" * 82)

for size in INPUT_SIZES:
    c, h, w = size
    inp_str = f"({c}, {h}, {w})"
    # ptflops MACs
    macs = "N/A"
    if PTFLOPS:
        try:
            clean = hrnet_module.HighResolutionNet(config, num_classes=19)
            mac_val, _ = get_model_complexity_info(
                clean, size, print_per_layer_stat=False, as_strings=False, verbose=False
            )
            macs = mac_val / 1e9
            del clean
            torch.cuda.empty_cache()
        except Exception as e:
            macs = "Err"
            warnings.warn(f"ptflops failed @ {size}: {e}")

    theo = macs * 2 if isinstance(macs, (int, float)) else "N/A"

    # latency + activations
    try:
        tracker = ActivationTracker(model)
        act_mb = tracker(torch.randn(1, *size)) / (1024 ** 2)
        lat_ms = latency_ms(model, size)
    except RuntimeError as e:
        act_mb, lat_ms = "OOM", "OOM"
        warnings.warn(f"OOM @ {size}: {e}")

    # print row
    print(f"{inp_str:>18} | "
          f"{theo if isinstance(theo, str) else f'{theo:13.2f}'} | "
          f"{macs if isinstance(macs, str) else f'{macs:10.2f}'} | "
          f"{total_params:9.2f} | "
          f"{lat_ms if isinstance(lat_ms, str) else f'{lat_ms:12.2f}'} | "
          f"{act_mb if isinstance(act_mb, str) else f'{act_mb:8.2f}'}")

    gc.collect()
