#!/usr/bin/env python3
"""
ESPNet Profiling Script - Params, Latency (CPU), and Activation Memory (forward)
Author: adapted for user
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import warnings
from typing import List

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------
# 1) Basic building blocks
# ------------------------------------------------------------------------------
def conv1x1(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 1, stride, bias=False)


class Activation(nn.Module):
    def __init__(self, act_type='prelu'):
        super().__init__()
        self.act = nn.PReLU() if act_type == 'prelu' else nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_c, out_c, k, s=1, d=1, act_type='prelu'):
        padding = d * (k - 1) // 2
        super().__init__(
            nn.Conv2d(in_c, out_c, k, s, padding, dilation=d, bias=False),
            nn.BatchNorm2d(out_c),
            Activation(act_type)
        )


class DeConvBNAct(nn.Sequential):
    def __init__(self, in_c, out_c, act_type='prelu'):
        super().__init__(
            nn.ConvTranspose2d(in_c, out_c, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            Activation(act_type)
        )


# ------------------------------------------------------------------------------
# 2) ESPNet modules (same architecture as used before)
# ------------------------------------------------------------------------------
class ESPModule(nn.Module):
    def __init__(self, in_c, out_c, K=5, ks=3, stride=1, act_type='prelu'):
        super().__init__()
        self.K = K
        self.use_skip = (in_c == out_c) and (stride == 1)
        kn = out_c // K
        k1 = out_c - kn * (K - 1)
        self.is_perfect = (k1 == kn)

        if self.is_perfect:
            self.reduce = conv1x1(in_c, kn, stride)
        else:
            self.reduce = nn.ModuleList([conv1x1(in_c, k1, stride),
                                         conv1x1(in_c, kn, stride)])

        self.layers = nn.ModuleList()
        for i in range(K):
            ch = kn if self.is_perfect else (k1 if i == 0 else kn)
            self.layers.append(ConvBNAct(ch, ch, ks, 1, 2 ** i, act_type))

    def forward(self, x):
        res = x if self.use_skip else None
        if self.is_perfect:
            x_r = self.reduce(x)
            feats = [layer(x_r) for layer in self.layers]
            for i in range(1, len(feats)):
                feats[i] = feats[i] + feats[i - 1]
        else:
            x_r1 = self.reduce[0](x)
            x_rn = self.reduce[1](x)
            feats = [self.layers[0](x_r1)] + [layer(x_rn) for layer in self.layers[1:]]
            for i in range(2, len(feats)):
                feats[i] = feats[i] + feats[i - 1]

        out = torch.cat(feats, 1)
        if res is not None:
            out = out + res
        return out


class L2Block(nn.Module):
    def __init__(self, in_c, hid_c, alpha, act_type):
        super().__init__()
        self.down = ESPModule(in_c, hid_c, stride=2, act_type=act_type)
        self.layers = nn.Sequential(*[ESPModule(hid_c, hid_c, act_type=act_type)
                                      for _ in range(alpha)])

    def forward(self, x):
        x = self.down(x)
        x = self.layers(x)
        return x


class L3Block(nn.Module):
    def __init__(self, in_c, hid_c, out_c, alpha, act_type):
        super().__init__()
        self.down = ESPModule(in_c, hid_c, stride=2, act_type=act_type)
        self.layers = nn.Sequential(*[ESPModule(hid_c, hid_c, act_type=act_type)
                                      for _ in range(alpha)])
        self.out_conv = conv1x1(hid_c, out_c)

    def forward(self, x):
        x = self.down(x)
        x = self.layers(x)
        return self.out_conv(x)


class Decoder(nn.Module):
    def __init__(self, num_class, l1_c, l2_c, act_type='prelu'):
        super().__init__()
        self.up3 = DeConvBNAct(num_class, num_class, act_type)
        self.cat2 = ConvBNAct(l2_c, num_class, 1, act_type=act_type)
        self.conv2 = ESPModule(2 * num_class, num_class, act_type=act_type)
        self.up2 = DeConvBNAct(num_class, num_class, act_type)
        self.cat1 = ConvBNAct(l1_c, num_class, 1, act_type=act_type)
        self.conv1 = ESPModule(2 * num_class, num_class, act_type=act_type)
        self.up1 = DeConvBNAct(num_class, num_class, act_type)

    def forward(self, x, x_l1, x_l2):
        x = self.up3(x)

        # ensure shapes match for cat
        if x.shape[2:] != x_l2.shape[2:]:
            x_l2 = F.interpolate(x_l2, size=x.shape[2:], mode='bilinear', align_corners=False)
        x_l2 = self.cat2(x_l2)
        x = torch.cat([x, x_l2], 1)
        x = self.conv2(x)

        x = self.up2(x)
        if x.shape[2:] != x_l1.shape[2:]:
            x_l1 = F.interpolate(x_l1, size=x.shape[2:], mode='bilinear', align_corners=False)
        x_l1 = self.cat1(x_l1)
        x = torch.cat([x, x_l1], 1)
        x = self.conv1(x)

        return self.up1(x)


# ------------------------------------------------------------------------------
# 3) ESPNet model
# ------------------------------------------------------------------------------
class ESPNet(nn.Module):
    def __init__(self, num_class=21, n_channel=3, act_type='prelu'):
        super().__init__()
        self.l1 = ConvBNAct(n_channel, 16, 3, 2, act_type=act_type)
        self.l2 = L2Block(16, 64, alpha=2, act_type=act_type)
        self.l3 = L3Block(64, 128, num_class, alpha=8, act_type=act_type)
        self.dec = Decoder(num_class, l1_c=16, l2_c=64, act_type=act_type)

    def forward(self, x):
        inp = x
        x = self.l1(x)
        x_l1 = x
        x = self.l2(x)
        x_l2 = x
        x = self.l3(x)
        x = self.dec(x, x_l1, x_l2)
        x = F.interpolate(x, inp.shape[2:], mode='bilinear', align_corners=True)
        return x


# ------------------------------------------------------------------------------
# 4) Utilities: params, latency measurement, activation hooks
# ------------------------------------------------------------------------------

def get_model_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_latency(model: nn.Module, inp: tuple, runs: int = 3, repeat: int = 5) -> float:
    """
    CPU latency measurement in milliseconds. Model moved to cpu for measurement.
    We do warmups (2) then runs x repeat, return average per-inference time (ms).
    """
    device = torch.device('cpu')
    model.to(device).eval()
    x = torch.randn(1, *inp).to(device)
    with torch.no_grad():
        for _ in range(2):
            model(x)  # warmup
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        for _ in range(repeat):
            model(x)
        end = time.perf_counter()
        times.append((end - start) * 1000.0 / repeat)
    return float(sum(times) / len(times))


# ----- Activation measurement via forward hooks on leaf modules -----
class ActivationMeter:
    def __init__(self):
        self.bytes = 0
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def reset(self):
        self.bytes = 0

    def _hook_fn(self, module, inp, out):
        # out may be tensor, tuple, list, dict
        def _add_tensor(t):
            if isinstance(t, torch.Tensor):
                # count number of elements of the tensor
                # only count tensors that require grad or not - it's forward activation so include all
                self.bytes += t.numel() * 4  # float32 bytes
        if isinstance(out, torch.Tensor):
            _add_tensor(out)
        elif isinstance(out, (list, tuple)):
            for o in out:
                _add_tensor(o)
        elif isinstance(out, dict):
            for o in out.values():
                _add_tensor(o)

    def attach_hooks_to_leaf_modules(self, model: nn.Module):
        # Only attach to leaf modules (modules without children) to avoid double counting
        for module in model.modules():
            # skip the top-level Module itself? we want leaf modules only
            if len(list(module.children())) == 0:
                handle = module.register_forward_hook(self._hook_fn)
                self.handles.append(handle)

    def remove_hooks(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []


def measure_activation_bytes(model: nn.Module, inp: tuple) -> int:
    """
    Returns estimated activation bytes (sum of leaf-module outputs) for a single forward.
    Note: This is an estimate; peak memory differs and exact accounting is complex.
    """
    meter = ActivationMeter()
    meter.attach_hooks_to_leaf_modules(model)
    meter.reset()
    device = torch.device('cpu')
    model.to(device).eval()
    x = torch.randn(1, *inp).to(device)
    with torch.no_grad():
        try:
            _ = model(x)
        except Exception as e:
            # Remove hooks before raising
            meter.remove_hooks()
            raise e
    bytes_val = meter.bytes
    meter.remove_hooks()
    return int(bytes_val)


# ------------------------------------------------------------------------------
# 5) Main loop with full input list, printing params, activation size, latency
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    INPUT_RESOLUTIONS = [
        (3, 640, 360),
        (3, 1280, 720),
        (3, 1360, 760),
        (3, 1600, 900),
        (3, 1920, 1080),
        (3, 2048, 1152),
        (3, 2560, 1440),
        (3, 3840, 2160),
    ]

    model = ESPNet(num_class=21)

    print("ESPNet Profiling")
    print("=" * 100)
    print(f"{'Input':>20} | {'Params (M)':>10} | {'Act Size (MB)':>14} | {'Latency (ms)':>12}")
    print("-" * 100)

    params_m = get_model_parameters(model) / 1e6

    for inp in INPUT_RESOLUTIONS:
        try:
            # Activation bytes (single forward with hooks)
            act_bytes = measure_activation_bytes(model, inp)
            act_mb = act_bytes / (1024.0 ** 2)

            # Latency measured w/out hooks (so measurement isn't impacted)
            latency_ms = measure_latency(model, inp)

            print(f"{str(inp):>20} | {params_m:10.2f} | {act_mb:14.2f} | {latency_ms:12.2f}")
        except RuntimeError as re:
            print(f"{str(inp):>20} | ERROR during forward: {re}")
        except Exception as e:
            print(f"{str(inp):>20} | Unexpected error: {e}")
