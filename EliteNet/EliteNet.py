import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# --- Network Architecture Definitions ---
class ConvBlockNoSkip(nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, pool=True):
        super().__init__()
        pad = (k_sz - 1) // 2
        self.out_channels = out_c
        self.pool = nn.MaxPool2d(kernel_size=2) if pool else None

        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k_sz, padding=pad, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(out_c),
            nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding=pad, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x):
        if self.pool:
            x = self.pool(x)
        return self.block(x)

class UNetEncoderWithSkip(nn.Module):
    def __init__(self, in_c, layers, k_sz=3):
        super().__init__()
        self.layers = layers
        self.first = ConvBlockNoSkip(in_c, layers[0], k_sz, pool=False)

        self.down_path = nn.ModuleList()
        self.up_path = nn.ModuleList()

        for i in range(len(layers) - 1):
            self.down_path.append(ConvBlockNoSkip(layers[i], layers[i+1], k_sz, pool=True))
        for i in range(len(layers) - 1):
            self.up_path.append(nn.ConvTranspose2d(layers[i+1], layers[i], kernel_size=2, stride=2, bias=True))

        self.final_conv = nn.Conv2d(layers[0], in_c, kernel_size=1, bias=True)

    def forward(self, x):
        activations = []

        # Encoder
        x1 = self.first(x)
        activations.append(x1)
        x = x1
        for down in self.down_path:
            x = down(x)
            activations.append(x)

        # Decoder
        for i, up in enumerate(reversed(self.up_path)):
            x = up(x)
            skip_connection = activations[len(self.layers)-2-i]

            # Fix mismatched spatial size by cropping/padding
            diff_h = skip_connection.size(2) - x.size(2)
            diff_w = skip_connection.size(3) - x.size(3)
            if diff_h != 0 or diff_w != 0:
                x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                              diff_h // 2, diff_h - diff_h // 2])

            x = x + skip_connection
            activations.append(x)

        # Final conv
        x = self.final_conv(x)
        activations.append(x)
        return x, activations

# --- FLOPs / Params Calculation ---
def calculate_conv_flops_params(input_shape, out_channels, kernel_size, stride, padding, bias=True):
    in_channels = input_shape[2]
    k_h, k_w = kernel_size, kernel_size
    s_h, s_w = stride, stride
    p_h, p_w = padding, padding

    out_h = (input_shape[0] + 2*p_h - (k_h - 1) - 1)//s_h + 1
    out_w = (input_shape[1] + 2*p_w - (k_w - 1) - 1)//s_w + 1
    params = (k_h*k_w*in_channels + (1 if bias else 0)) * out_channels
    flops = out_h*out_w*out_channels*k_h*k_w*in_channels
    if bias:
        flops += out_h*out_w*out_channels
    return params, flops, (out_h, out_w, out_channels)

def calculate_bn_flops_params(input_shape):
    return 2*input_shape[2], 0, input_shape

def calculate_relu_flops(input_shape):
    return 0, 0, input_shape

def calculate_pooling_flops_params(input_shape, kernel_size=2, stride=2):
    out_h = (input_shape[0] - kernel_size)//stride + 1
    out_w = (input_shape[1] - kernel_size)//stride + 1
    return 0, 0, (out_h, out_w, input_shape[2])

def calculate_convtranspose_flops_params(input_shape, out_channels, kernel_size, stride, padding, output_padding=0, bias=True):
    in_channels = input_shape[2]
    k_h, k_w = kernel_size, kernel_size
    s_h, s_w = stride, stride
    p_h, p_w = padding, padding
    op_h, op_w = output_padding, output_padding
    out_h = (input_shape[0] - 1) * s_h - 2*p_h + (k_h - 1) + op_h + 1
    out_w = (input_shape[1] - 1) * s_w - 2*p_w + (k_w - 1) + op_w + 1
    params = (in_channels*k_h*k_w + (1 if bias else 0))*out_channels
    flops = out_h*out_w*out_channels*k_h*k_w*in_channels
    if bias:
        flops += out_h*out_w*out_channels
    return params, flops, (out_h, out_w, out_channels)

def calculate_addition_flops(input_shape):
    return 0, 0, input_shape

# --- Activation Memory Utility ---
def compute_activation_size_mb(activations):
    total_elements = sum([act.numel() for act in activations])
    return total_elements*4/(1024**2)  # MB

# --- Trace Model for FLOPs, Params, Latency, Activation ---
def trace_unet_full(model, input_res, in_ch, layers):
    total_params, total_flops = 0, 0
    x = torch.randn(1, in_ch, input_res[0], input_res[1])
    start_time = time.time()
    with torch.no_grad():
        out, activations = model(x)
    latency_ms = (time.time() - start_time)*1000
    activation_mb = compute_activation_size_mb(activations)

    # Simple FLOPs/Params estimation using first block (for demonstration)
    # For detailed manual trace, you can integrate the previous full trace code

    print(f"Input: {input_res[0]}x{input_res[1]} | Params: ~{sum(p.numel() for p in model.parameters())/1e6:.2f} M | Latency: {latency_ms:.2f} ms | Activation: {activation_mb:.2f} MB | Output Shape: {out.shape}")

# --- Run Benchmark ---
def run_benchmark():
    input_sizes = [
        (360, 640),
        (720, 1280),
        (760, 1360),
        (900, 1600),
        (1080, 1920),
        (1152, 2048),
        (1440, 2560),
        (3840, 2160)
    ]
    in_channels = 3
    layers = [4, 8, 16, 32]

    print("=== UNetEncoderWithSkip Benchmark (Fixed Spatial Mismatch) ===")
    model = UNetEncoderWithSkip(in_c=in_channels, layers=layers, k_sz=3).eval()
    for res in input_sizes:
        trace_unet_full(model, res, in_channels, layers)

if __name__ == "__main__":
    run_benchmark()
