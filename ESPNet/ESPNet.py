#!/usr/bin/env python3
"""
ESPNet Profiling Script - Theoretical FLOPs, THOP, PTFLOPs, Params, Latency, and Activation Memory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import warnings

warnings.filterwarnings("ignore")

# Optional profiling libraries
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: 'thop' is not installed. THOP measurements will be skipped.")

try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    PTFLOPS_AVAILABLE = False
    print("Warning: 'ptflops' is not installed. PTFLOPs measurements will be skipped.")

# ------------------------------------------------------------------------------
# 1. Basic Building Blocks and Activation Tracking Mixin
# ------------------------------------------------------------------------------
class ActivationTrackerMixin:
    def reset_activation_bytes(self):
        self.activation_bytes = 0

    def add_activation(self, tensor):
        # Number of elements × size of float32 (4 bytes)
        self.activation_bytes += tensor.numel() * 4

def conv1x1(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 1, stride, bias=False)

class Activation(nn.Module):
    def __init__(self, act_type='prelu'):
        super().__init__()
        self.act = nn.PReLU() if act_type == 'prelu' else nn.ReLU(inplace=True)
    def forward(self, x): return self.act(x)

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
# 2. ESPNet Modules
# ------------------------------------------------------------------------------
class ESPModule(nn.Module):
    def __init__(self, in_c, out_c, K=5, ks=3, stride=1, act_type='prelu'):
        super().__init__()
        self.K = K
        self.use_skip = in_c == out_c and stride == 1
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
            self.layers.append(ConvBNAct(ch, ch, ks, 1, 2**i, act_type))

    def forward(self, x):
        res = x if self.use_skip else None
        if self.is_perfect:
            x_r = self.reduce(x)
            feats = [layer(x_r) for layer in self.layers]
            for i in range(1, len(feats)): feats[i] = feats[i] + feats[i-1]
        else:
            x_r1 = self.reduce[0](x); x_rn = self.reduce[1](x)
            feats = [self.layers[0](x_r1)] + [layer(x_rn) for layer in self.layers[1:]]
            for i in range(2, len(feats)): feats[i] = feats[i] + feats[i-1]

        out = torch.cat(feats, 1)
        if res is not None: out += res
        return out

class L2Block(nn.Module):
    def __init__(self, in_c, hid_c, alpha, use_skip, reinforce, act_type):
        super().__init__()
        self.use_skip, self.reinforce = use_skip, reinforce
        ic = in_c + 3 if reinforce else in_c
        self.down = ESPModule(ic, hid_c, stride=2, act_type=act_type)
        self.layers = nn.Sequential(*[ESPModule(hid_c, hid_c, act_type=act_type)
                                      for _ in range(alpha)])

    def forward(self, x, x_input=None):
        x = self.down(x); skip = x if self.use_skip else None
        x = self.layers(x)
        if self.use_skip: x = torch.cat([x, skip], 1)
        if self.reinforce and x_input is not None:
            size = x.shape[2:]; q = F.interpolate(x_input, size,
                                                 mode='bilinear',
                                                 align_corners=False)
            x = torch.cat([x, q], 1)
        return x

class L3Block(nn.Module):
    def __init__(self, in_c, hid_c, out_c, alpha, use_skip, reinforce, use_decoder, act_type):
        super().__init__()
        self.use_skip, self.reinforce = use_skip, reinforce
        ic = in_c + 3 if reinforce else in_c
        self.down = ESPModule(ic, hid_c, stride=2, act_type=act_type)
        self.layers = nn.Sequential(*[ESPModule(hid_c, hid_c, act_type=act_type)
                                      for _ in range(alpha)])
        out_ch = hid_c * 2 if use_skip else hid_c
        self.out_conv = (ConvBNAct(out_ch, out_c, 1, act_type=act_type)
                         if use_decoder else conv1x1(out_ch, out_c))

    def forward(self, x):
        x = self.down(x); skip = x if self.use_skip else None
        x = self.layers(x)
        if self.use_skip: x = torch.cat([x, skip], 1)
        return self.out_conv(x)

class Decoder(nn.Module):
    def __init__(self, num_class, l1_c, l2_c, act_type='prelu'):
        super().__init__()
        self.up3 = DeConvBNAct(num_class, num_class, act_type)
        self.cat2 = ConvBNAct(l2_c, num_class, 1, act_type=act_type)
        self.conv2 = ESPModule(2*num_class, num_class, act_type=act_type)
        self.up2 = DeConvBNAct(num_class, num_class, act_type)
        self.cat1 = ConvBNAct(l1_c, num_class, 1, act_type=act_type)
        self.conv1 = ESPModule(2*num_class, num_class, act_type=act_type)
        self.up1 = DeConvBNAct(num_class, num_class, act_type)

    def forward(self, x, x_l1, x_l2):
        x = self.up3(x); x_l2 = self.cat2(x_l2)
        x = torch.cat([x, x_l2], 1); x = self.conv2(x)
        x = self.up2(x); x_l1 = self.cat1(x_l1)
        x = torch.cat([x, x_l1], 1); x = self.conv1(x)
        return self.up1(x)

# ------------------------------------------------------------------------------
# 3. ESPNet with Activation Tracking
# ------------------------------------------------------------------------------
class ESPNet(ActivationTrackerMixin, nn.Module):
    def __init__(self, num_class=1, n_channel=3, arch_type='espnet',
                 K=5, alpha2=2, alpha3=8, block_ch=[16,64,128],
                 act_type='prelu'):
        super().__init__()
        types = ['espnet','espnet-a','espnet-b','espnet-c']
        if arch_type not in types: raise ValueError("Unsupported arch")
        self.arch_type = arch_type # Store arch_type
        use_skip = arch_type in ['espnet','espnet-b','espnet-c']
        reinforce = arch_type in ['espnet','espnet-c']
        use_decoder = arch_type=='espnet'
        if arch_type=='espnet-a': block_ch[2]=block_ch[1]

        self.l1 = ConvBNAct(n_channel, block_ch[0], 3, 2, act_type=act_type)
        self.l2 = L2Block(block_ch[0], block_ch[1], alpha2,
                          use_skip, reinforce, act_type)
        l2_out_c = block_ch[1]*2 if use_skip else block_ch[1]
        if reinforce: l2_out_c += 3
        self.l3 = L3Block(l2_out_c, block_ch[2], num_class, alpha3,
                          use_skip, reinforce, use_decoder, act_type)
        self.dec = Decoder(num_class, l1_c=block_ch[0]+n_channel,
                           l2_c=l2_out_c, act_type=act_type) if use_decoder else None

    def forward(self, x):
        # Reset activation byte counter
        self.reset_activation_bytes()

        inp = x; self.add_activation(x)

        # Layer 1
        x = self.l1(x); self.add_activation(x)

        # L2 (with optional reinforcement)
        x_l1_for_decoder = None
        if self.l2.reinforce:
            size = x.shape[2:]; q = F.interpolate(inp, size,
                                                 mode='bilinear',
                                                 align_corners=False)
            x_cat = torch.cat([x, q], 1); self.add_activation(x_cat)
            x_l1_for_decoder = x_cat # Capture x_l1 here
            x = self.l2(x_cat, inp); self.add_activation(x)
        else:
            x = self.l2(x); self.add_activation(x)

        # Correctly capture L2 features for decoder (x_l2) before L3 changes its channels
        x_l2_for_decoder = None
        if self.dec:
            x_l2_for_decoder = x # x is the output of L2, which is correct for x_l2

        # L3 (with optional reinforcement)
        if self.l3.reinforce:
             size = x.shape[2:]; q = F.interpolate(inp, size,
                                                  mode='bilinear',
                                                  align_corners=False)
             x = torch.cat([x, q], 1); self.add_activation(x)

        x = self.l3(x); self.add_activation(x)

        # Decoder or final upsample
        if self.dec:
            x = self.dec(x, x_l1_for_decoder, x_l2_for_decoder); self.add_activation(x)
        else:
            x = F.interpolate(x, inp.shape[2:],
                              mode='bilinear',
                              align_corners=True)
            self.add_activation(x)
        return x

# ------------------------------------------------------------------------------
# 4. Profiling Utilities
# ------------------------------------------------------------------------------
NUM_CLASSES = 21

def _calc_conv_layer_flops(in_c, out_c, k, s, p, h, w, dil=1, groups=1):
    kh = k if isinstance(k, int) else k[0]
    kw = k if isinstance(k, int) else k[1]
    eff_kh = kh + (kh - 1) * (dil - 1); eff_kw = kw + (kw - 1) * (dil - 1)
    out_h = (h + 2 * p - eff_kh) // s + 1; out_w = (w + 2 * p - eff_kw) // s + 1

    flops_conv = 2 * out_c * out_h * out_w * (in_c / groups * kh * kw)
    flops_bn = 4 * out_c * out_h * out_w
    flops_act = 1 * out_c * out_h * out_w

    return flops_conv + flops_bn + flops_act, (out_c, out_h, out_w)

def _calc_deconv_layer_flops(in_c, out_c, k, s, p, h, w, op=0):
    # For deconv, output padding is often 1 to ensure output_size = (input_size - 1) * stride + kernel_size - 2 * padding + output_padding
    # Here, we assume the default op=1 from previous usage.
    out_h = (h - 1) * s - 2 * p + k + op
    out_w = (w - 1) * s - 2 * p + k + op

    flops_deconv = 2 * out_c * out_h * out_w * in_c * k * k
    flops_bn = 4 * out_c * out_h * out_w
    flops_act = 1 * out_c * out_h * out_w

    return flops_deconv + flops_bn + flops_act, (out_c, out_h, out_w)

def _calc_add_cat_flops(in_ch, h, w, type='add'):
    if type == 'add': return in_ch * h * w
    if type == 'cat': return 0
    return 0

def _calc_interpolate_flops(in_ch, h_in, w_in, h_out, w_out, mode='bilinear'):
    if mode == 'bilinear': return 7 * in_ch * h_out * w_out
    return 0

def _calc_esp_module_flops(in_c, out_c, h, w, stride, K):
    total_flops = 0
    use_skip = (in_c == out_c) and (stride == 1)

    kn = out_c // K
    k1 = out_c - kn * (K - 1)
    is_perfect = (k1 == kn)

    # 1. Reduce Layer
    if is_perfect:
        f, sh_red = _calc_conv_layer_flops(in_c, kn, 1, stride, 0, h, w)
        total_flops += f
        sh_layers_in_ch = [kn] * K
        h_dil, w_dil = sh_red[1], sh_red[2]
    else:
        f1, sh_red1 = _calc_conv_layer_flops(in_c, k1, 1, stride, 0, h, w)
        fn, sh_redn = _calc_conv_layer_flops(in_c, kn, 1, stride, 0, h, w)
        total_flops += f1 + fn
        sh_layers_in_ch = [k1] + [kn] * (K - 1)
        h_dil, w_dil = sh_red1[1], sh_red1[2]

    # 2. Parallel Dilated Convolutions and cascading sums
    for i in range(K):
        f, _ = _calc_conv_layer_flops(sh_layers_in_ch[i], sh_layers_in_ch[i], 3, 1, 2**i, h_dil, w_dil, dil=2**i)
        total_flops += f

    # Cascading Sums
    if is_perfect:
        total_flops += (K - 1) * _calc_add_cat_flops(kn, h_dil, w_dil, 'add')
    else:
        total_flops += (K - 2) * _calc_add_cat_flops(kn, h_dil, w_dil, 'add')

    # 3. Concatenation (0 FLOPs) and Skip Connection
    if use_skip:
        total_flops += _calc_add_cat_flops(out_c, h_dil, w_dil, 'add')

    final_shape = (out_c, h_dil, w_dil)
    return total_flops, final_shape

def calculate_espnet_theoretical_flops(inp):
    flops = 0
    K = 5
    alpha2 = 2
    alpha3 = 8
    block_ch = [16, 64, 128]
    n_channel = inp[0]
    
    # --- L1 Block ---
    f, shape_l1 = _calc_conv_layer_flops(n_channel, block_ch[0], 3, 2, 1, inp[1], inp[2])
    flops += f

    # --- Reinforce (for L2) ---
    f_interp1 = _calc_interpolate_flops(n_channel, inp[1], inp[2], shape_l1[1], shape_l1[2])
    flops += f_interp1
    l2_in_c = block_ch[0] + n_channel
    shape_l1_cat = (l2_in_c, shape_l1[1], shape_l1[2])
    
    shape_l1_for_decoder = shape_l1_cat
    
    # --- L2 Block ---
    f_l2_down, shape_l2_down = _calc_esp_module_flops(l2_in_c, block_ch[1], shape_l1_cat[1], shape_l1_cat[2], stride=2, K=K)
    flops += f_l2_down
    
    shape_l2_layers_out = shape_l2_down
    for _ in range(alpha2):
        f_esp, shape_l2_layers_out = _calc_esp_module_flops(shape_l2_layers_out[0], shape_l2_layers_out[0], shape_l2_layers_out[1], shape_l2_layers_out[2], stride=1, K=K)
        flops += f_esp
        
    l2_block_out_c = shape_l2_layers_out[0] + shape_l2_down[0]
    shape_l2_block_out = (l2_block_out_c, shape_l2_layers_out[1], shape_l2_layers_out[2])
    
    # --- Reinforce (for L3) ---
    f_interp2 = _calc_interpolate_flops(n_channel, inp[1], inp[2], shape_l2_block_out[1], shape_l2_block_out[2])
    flops += f_interp2
    l3_in_c = l2_block_out_c + n_channel
    shape_l2_reinforce = (l3_in_c, shape_l2_block_out[1], shape_l2_block_out[2])

    shape_l2_for_decoder = shape_l2_reinforce

    # --- L3 Block ---
    f_l3_down, shape_l3_down = _calc_esp_module_flops(l3_in_c, block_ch[2], shape_l2_reinforce[1], shape_l2_reinforce[2], stride=2, K=K)
    flops += f_l3_down
    
    shape_l3_layers_out = shape_l3_down
    for _ in range(alpha3):
        f_esp, shape_l3_layers_out = _calc_esp_module_flops(shape_l3_layers_out[0], shape_l3_layers_out[0], shape_l3_layers_out[1], shape_l3_layers_out[2], stride=1, K=K)
        flops += f_esp

    l3_block_out_c = shape_l3_layers_out[0] + shape_l3_down[0]
    shape_l3_block_pre_conv = (l3_block_out_c, shape_l3_layers_out[1], shape_l3_layers_out[2])
    
    f_l3_out, shape_l3_final = _calc_conv_layer_flops(l3_block_out_c, NUM_CLASSES, 1, 1, 0, shape_l3_block_pre_conv[1], shape_l3_block_pre_conv[2])
    flops += f_l3_out
    
    # --- Decoder ---
    current_shape = shape_l3_final
    
    f, shape_up3 = _calc_deconv_layer_flops(current_shape[0], NUM_CLASSES, 3, 2, 1, current_shape[1], current_shape[2], op=1)
    flops += f
    
    f, _ = _calc_conv_layer_flops(shape_l2_for_decoder[0], NUM_CLASSES, 1, 1, 0, shape_up3[1], shape_up3[2])
    flops += f
    
    shape_dec_conv2_in = (NUM_CLASSES * 2, shape_up3[1], shape_up3[2])
    f, shape_dec_conv2_out = _calc_esp_module_flops(shape_dec_conv2_in[0], NUM_CLASSES, shape_dec_conv2_in[1], shape_dec_conv2_in[2], stride=1, K=K)
    flops += f
    
    f, shape_up2 = _calc_deconv_layer_flops(shape_dec_conv2_out[0], NUM_CLASSES, 3, 2, 1, shape_dec_conv2_out[1], shape_dec_conv2_out[2], op=1)
    flops += f
    
    f, _ = _calc_conv_layer_flops(shape_l1_for_decoder[0], NUM_CLASSES, 1, 1, 0, shape_up2[1], shape_up2[2])
    flops += f
    
    shape_dec_conv1_in = (NUM_CLASSES * 2, shape_up2[1], shape_up2[2])
    f, shape_dec_conv1_out = _calc_esp_module_flops(shape_dec_conv1_in[0], NUM_CLASSES, shape_dec_conv1_in[1], shape_dec_conv1_in[2], stride=1, K=K)
    flops += f
    
    f, _ = _calc_deconv_layer_flops(shape_dec_conv1_out[0], NUM_CLASSES, 3, 2, 1, shape_dec_conv1_out[1], shape_dec_conv1_out[2], op=1)
    flops += f
    
    return flops

def get_model_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_latency(model, inp, runs=5, repeat=10):
    device = torch.device('cpu'); model.to(device).eval(); x = torch.randn(1, *inp).to(device)
    with torch.no_grad():
        for _ in range(3): model(x) # Warmup
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        for _ in range(repeat): model(x)
        end = time.perf_counter()
        times.append((end - start) * 1000 / repeat)
    return sum(times) / len(times)

# ------------------------------------------------------------------------------
# 5. Main Execution
# ------------------------------------------------------------------------------
def main():
    INPUT_RESOLUTIONS = [(3, 640, 360), (3, 1280, 720), (3, 1360, 760),
                         (3, 1600, 900), (3, 1920, 1080), (3, 2048, 1152),
                         (3, 2560, 1440), (3, 3840, 2160)]
    
    arch_type = 'espnet'
    model = ESPNet(num_class=NUM_CLASSES, arch_type=arch_type)

    print("ESPNet Profiling")
    print(f"Architecture: {arch_type.upper()}")
    print("="*120)
    print(f"{'Input':>18} | {'Theo FLOPs (G)':>14} | {'THOP (G)':>10} | {'PTFLOPS (G)':>12} | {'Params (M)':>10} | {'Latency (ms)':>12} | {'Act. Size (MB)':>15}")
    print("-"*120)

    for inp in INPUT_RESOLUTIONS:
        thop_val, pt_val, params, lat, theo_val, act_size_mb = "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
        
       

        # Activation Size
        try:
            with torch.no_grad():
                _ = model(torch.randn(1, *inp))
            act_size_mb = f"{model.activation_bytes / (1024**2):.2f}"
            print(f"Activation Size (MB) for {inp}: {act_size_mb}")
        except Exception as e:
            act_size_mb = "Error"
            print(f"Warning: Activation size measurement failed for {inp}: {e}")
            
            
        
        #print(f"{str(inp):>18} | {theo_val:>14} | {thop_val:>10} | {pt_val:>12} | {params:>10} | {lat:>12} | {act_size_mb:>15}")

if __name__ == "__main__":
    main()
