import torch
import torch.nn as nn
import time
import timeit # timeit is still useful for CPU-only, but less critical for GPU

# --- Define a simplified UNet Encoder in PyTorch ---
class SimpleUNetEncoder(nn.Module):
    def __init__(self, in_channels):
        super(SimpleUNetEncoder, self).__init__()

        # Encoder Block 1 (conv1 -> conv1 -> MaxPool)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder Block 2 (conv2 -> conv2 -> MaxPool)
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder Block 3 (conv3 -> conv3 -> MaxPool)
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder Block 4 (conv4 -> conv4) - Bottleneck in typical UNet
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        return e4 # This is the output of the encoder

# --- Helper function for FLOPs and Params (from previous code) ---
# Note: For actual PyTorch models, you'd typically use libraries like `thop` or `torchsummary`
# to get FLOPs/Params directly from the model object, which is more robust.
# However, for consistency with your previous theoretical calculations, I'll keep your helper functions
# as they represent the underlying math.
def calculate_conv_flops_params(input_shape, output_channels, kernel_size, stride, padding):
    if len(input_shape) != 3:
        raise ValueError(f"Input shape must be (H, W, C), but got {input_shape}")
    in_channels = input_shape[2]
    k_h, k_w = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    s_h, s_w = (stride, stride) if isinstance(stride, int) else stride
    p_h, p_w = (padding, padding) if isinstance(padding, int) else padding

    out_height = (input_shape[0] - k_h + 2 * p_h) // s_h + 1
    out_width = (input_shape[1] - k_w + 2 * p_w) // s_w + 1

    if out_height <= 0 or out_width <= 0:
         return 0, 0, (max(0, out_height), max(0, out_width), output_channels)

    params = (k_h * k_w * in_channels + 1) * output_channels # +1 for bias
    num_macs_per_output_element = k_h * k_w * in_channels
    mults = num_macs_per_output_element * out_height * out_width * output_channels
    adds = (num_macs_per_output_element - 1) * out_height * out_width * output_channels if num_macs_per_output_element > 0 else 0
    total_flops = mults + adds
    return params, total_flops, (out_height, out_width, output_channels)

def calculate_bn_flops_params(input_shape):
    if len(input_shape) != 3:
        raise ValueError(f"Input shape must be (H, W, C), but got {input_shape}")
    num_features = input_shape[2]
    params = 2 * num_features
    total_elements = input_shape[0] * input_shape[1] * input_shape[2]
    flops = 2 * total_elements # 2 ops per element (mult + add)
    return params, flops, input_shape

def calculate_relu_flops(input_shape):
    if len(input_shape) != 3:
        raise ValueError(f"Input shape must be (H, W, C), but got {input_shape}")
    total_elements = input_shape[0] * input_shape[1] * input_shape[2]
    flops = total_elements # 1 comparison/op per element
    return 0, flops, input_shape

def calculate_pooling_flops_params(input_shape, kernel_size, stride):
    if len(input_shape) != 3:
        raise ValueError(f"Input shape must be (H, W, C), but got {input_shape}")
    k_h, k_w = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    s_h, s_w = (stride, stride) if isinstance(stride, int) else stride
    out_height = input_shape[0] // s_h
    out_width = input_shape[1] // s_w
    return 0, 0, (out_height, out_width, input_shape[2])

def get_activation_size_mb(shape, bytes_per_element=4):
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    return (num_elements * bytes_per_element) / (1024 * 1024)

def get_unet_encoder_flops_params_theoretical(input_res):
    total_params = 0
    total_flops = 0
    total_activation_size_mb = 0 # This will sum up sizes of all intermediate activations

    # Initial input shape (H, W, C)
    curr_shape = (input_res[1], input_res[2], input_res[0])
    total_activation_size_mb += get_activation_size_mb(curr_shape)

    # Encoder Block 1
    # Conv1
    p, f, curr_shape = calculate_conv_flops_params(curr_shape, 64, 3, 1, 1)
    total_params += p; total_flops += f; total_activation_size_mb += get_activation_size_mb(curr_shape)
    # BN1
    p, f, curr_shape = calculate_bn_flops_params(curr_shape)
    total_params += p; total_flops += f; total_activation_size_mb += get_activation_size_mb(curr_shape)
    # ReLU1
    _, f, curr_shape = calculate_relu_flops(curr_shape)
    total_flops += f; total_activation_size_mb += get_activation_size_mb(curr_shape)
    # Conv2
    p, f, curr_shape = calculate_conv_flops_params(curr_shape, 64, 3, 1, 1)
    total_params += p; total_flops += f; total_activation_size_mb += get_activation_size_mb(curr_shape)
    # BN2
    p, f, curr_shape = calculate_bn_flops_params(curr_shape)
    total_params += p; total_flops += f; total_activation_size_mb += get_activation_size_mb(curr_shape)
    # ReLU2
    _, f, curr_shape = calculate_relu_flops(curr_shape)
    total_flops += f; total_activation_size_mb += get_activation_size_mb(curr_shape)
    # Pool1
    _, _, curr_shape = calculate_pooling_flops_params(curr_shape, 2, 2)
    total_activation_size_mb += get_activation_size_mb(curr_shape)

    # Encoder Block 2
    # Conv1
    p, f, curr_shape = calculate_conv_flops_params(curr_shape, 128, 3, 1, 1)
    total_params += p; total_flops += f; total_activation_size_mb += get_activation_size_mb(curr_shape)
    # BN1
    p, f, curr_shape = calculate_bn_flops_params(curr_shape)
    total_params += p; total_flops += f; total_activation_size_mb += get_activation_size_mb(curr_shape)
    # ReLU1
    _, f, curr_shape = calculate_relu_flops(curr_shape)
    total_flops += f; total_activation_size_mb += get_activation_size_mb(curr_shape)
    # Conv2
    p, f, curr_shape = calculate_conv_flops_params(curr_shape, 128, 3, 1, 1)
    total_params += p; total_flops += f; total_activation_size_mb += get_activation_size_mb(curr_shape)
    # BN2
    p, f, curr_shape = calculate_bn_flops_params(curr_shape)
    total_params += p; total_flops += f; total_activation_size_mb += get_activation_size_mb(curr_shape)
    # ReLU2
    _, f, curr_shape = calculate_relu_flops(curr_shape)
    total_flops += f; total_activation_size_mb += get_activation_size_mb(curr_shape)
    # Pool2
    _, _, curr_shape = calculate_pooling_flops_params(curr_shape, 2, 2)
    total_activation_size_mb += get_activation_size_mb(curr_shape)

    # Encoder Block 3
    # Conv1
    p, f, curr_shape = calculate_conv_flops_params(curr_shape, 256, 3, 1, 1)
    total_params += p; total_flops += f; total_activation_size_mb += get_activation_size_mb(curr_shape)
    # BN1
    p, f, curr_shape = calculate_bn_flops_params(curr_shape)
    total_params += p; total_flops += f; total_activation_size_mb += get_activation_size_mb(curr_shape)
    # ReLU1
    _, f, curr_shape = calculate_relu_flops(curr_shape)
    total_flops += f; total_activation_size_mb += get_activation_size_mb(curr_shape)
    # Conv2
    p, f, curr_shape = calculate_conv_flops_params(curr_shape, 256, 3, 1, 1)
    total_params += p; total_flops += f; total_activation_size_mb += get_activation_size_mb(curr_shape)
    # BN2
    p, f, curr_shape = calculate_bn_flops_params(curr_shape)
    total_params += p; total_flops += f; total_activation_size_mb += get_activation_size_mb(curr_shape)
    # ReLU2
    _, f, curr_shape = calculate_relu_flops(curr_shape)
    total_flops += f; total_activation_size_mb += get_activation_size_mb(curr_shape)
    # Pool3
    _, _, curr_shape = calculate_pooling_flops_params(curr_shape, 2, 2)
    total_activation_size_mb += get_activation_size_mb(curr_shape)

    # Encoder Block 4 (Bottleneck)
    # Conv1
    p, f, curr_shape = calculate_conv_flops_params(curr_shape, 512, 3, 1, 1)
    total_params += p; total_flops += f; total_activation_size_mb += get_activation_size_mb(curr_shape)
    # BN1
    p, f, curr_shape = calculate_bn_flops_params(curr_shape)
    total_params += p; total_flops += f; total_activation_size_mb += get_activation_size_mb(curr_shape)
    # ReLU1
    _, f, curr_shape = calculate_relu_flops(curr_shape)
    total_flops += f; total_activation_size_mb += get_activation_size_mb(curr_shape)
    # Conv2
    p, f, curr_shape = calculate_conv_flops_params(curr_shape, 512, 3, 1, 1)
    total_params += p; total_flops += f; total_activation_size_mb += get_activation_size_mb(curr_shape)
    # BN2
    p, f, curr_shape = calculate_bn_flops_params(curr_shape)
    total_params += p; total_flops += f; total_activation_size_mb += get_activation_size_mb(curr_shape)
    # ReLU2
    _, f, curr_shape = calculate_relu_flops(curr_shape)
    total_flops += f; total_activation_size_mb += get_activation_size_mb(curr_shape)

    return total_params, total_flops, total_activation_size_mb


# --- Setup for Latency Measurement ---
# Check for GPU availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

input_configs = [
    {'input_res': (3, 640, 360), 'batch_size': 1}, # C, H, W
    {'input_res': (3, 1280, 720), 'batch_size': 1},
    {'input_res': (3, 1360, 760), 'batch_size': 1},
    {'input_res': (3, 1600, 900), 'batch_size': 1},
    {'input_res': (3, 1920, 1080), 'batch_size': 1},
    {'input_res': (3, 2048, 1152), 'batch_size': 1},
    {'input_res': (3, 2560, 1440), 'batch_size': 1}
]

NUM_WARMUP_RUNS = 5
NUM_MEASUREMENT_RUNS = 30

print(f"{'Input Size (CxHxW)':>22} | {'Params (M)':>10} | {'FLOPs (B)':>10} | {'Activations (MB)':>18} | {'Latency (ms)':>15}")
print("-" * 90)

for config in input_configs:
    input_channels = config['input_res'][0]
    input_height = config['input_res'][1]
    input_width = config['input_res'][2]
    batch_size = config['batch_size']

    # Get theoretical metrics (these don't change based on batch size)
    params, flops, act_mem = get_unet_encoder_flops_params_theoretical(config['input_res'])

    # Initialize model and move to device
    model = SimpleUNetEncoder(input_channels).to(DEVICE)
    model.eval() # Set model to evaluation mode for inference

    # Create dummy input tensor
    dummy_input = torch.randn(batch_size, input_channels, input_height, input_width).to(DEVICE)

    # --- Warm-up runs ---
    for _ in range(NUM_WARMUP_RUNS):
        _ = model(dummy_input)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize() # Ensure GPU operations complete

    # --- Measurement runs ---
    latencies = []
    if DEVICE.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad(): # Disable gradient calculation for inference
        for _ in range(NUM_MEASUREMENT_RUNS):
            if DEVICE.type == 'cuda':
                start_event.record()
                _ = model(dummy_input)
                end_event.record()
                torch.cuda.synchronize() # Wait for the events to be recorded
                latency = start_event.elapsed_time(end_event) # milliseconds
            else: # CPU
                start_time = time.perf_counter()
                _ = model(dummy_input)
                end_time = time.perf_counter()
                latency = (end_time - start_time) * 1000 # Convert to milliseconds
            latencies.append(latency)

    average_latency = sum(latencies) / NUM_MEASUREMENT_RUNS

    print(f"{config['input_res'][0]}x{config['input_res'][1]}x{config['input_res'][2]:<5} | {params / 1e6:10.2f} | {flops / 1e9:10.2f} | {act_mem:18.2f} | {average_latency:14.5f}")