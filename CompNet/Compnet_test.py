import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# --- Model Definitions (as provided by you) ---

class encoder(nn.Module):
    def __init__(self, n_downconv = 3, in_chn = 3):
        super().__init__()
        self.n_downconv = n_downconv
        layer_list = [
            nn.Conv2d(in_channels=in_chn, out_channels=64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        ]
        for i in range(self.n_downconv):
            layer_list.extend([
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            ])
        layer_list.append(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1),
        )
        self.encoder = nn.Sequential(*layer_list)

    def forward(self, x):
        return torch.clamp(self.encoder(x), 0, 1)

class decoder(nn.Module):
    def __init__(self, n_upconv = 3, out_chn = 3):
        super().__init__()
        self.n_upconv = n_upconv
        layer_list = [
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        ]
        for i in range(self.n_upconv):
            layer_list.extend([
                nn.Conv2d(in_channels=64, out_channels=64*4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.PixelShuffle(2),
            ])
        layer_list.extend([
            nn.Conv2d(in_channels=64, out_channels=out_chn*4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2)
        ])
        self.decoder = nn.Sequential(*layer_list)

    def forward(self, x):
        return torch.clamp(self.decoder(x), 0, 1)

class autoencoder(nn.Module):
    def __init__(self, n_updownconv = 3, in_chn = 3, out_chn = 3):
        super().__init__()
        self.n_updownconv = n_updownconv
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.encoder = encoder(n_downconv = self.n_updownconv,in_chn=self.in_chn)
        self.decoder = decoder(n_upconv = self.n_updownconv, out_chn=self.out_chn)

    def forward(self, x):
        # These are for internal tracking within the model, not used by the metric function directly.
        # self.shape_input = list(x.shape) # Removed as not needed for this metric calculation
        x = self.encoder(x)
        # self.shape_latent = list(x.shape) # Removed as not needed for this metric calculation
        x = self.decoder(x)
        return x

# --- Metric Calculation Functions ---

def calculate_conv2d_flops(in_channels, out_channels, kernel_size, output_size_hw, stride_hw=(1,1)):
    """
    Calculate FLOPs for a Conv2d layer.
    FLOPs = 2 * C_in * K_h * K_w * C_out * O_h * O_w (multiplies + adds)
    This formula is for MACs. Each MAC is 2 FLOPs.
    Bias adds = O_h * O_w * C_out
    """
    output_h, output_w = output_size_hw
    # MACs = kernel_volume * output_elements
    macs_per_output_channel = (kernel_size * kernel_size * in_channels) * output_h * output_w
    total_macs = macs_per_output_channel * out_channels
    # Assuming 2 FLOPs per MAC (multiplication and addition)
    flops = 2 * total_macs
    # Add FLOPs for bias (if bias is true, each output element involves an addition)
    # PyTorch Conv2d has bias by default if not specified as False.
    bias_flops = output_h * output_w * out_channels
    flops += bias_flops
    return flops

def calculate_relu_flops(num_elements):
    """
    Calculate FLOPs for a ReLU activation.
    Each element involves one comparison/operation (max(0, x)).
    """
    return num_elements

def calculate_pixelshuffle_flops(num_elements_input_tensor_before_shuffle, upscale_factor):
    """
    Calculate FLOPs for PixelShuffle operation.
    PixelShuffle is primarily a data rearrangement operation and does not
    involve arithmetic operations in the traditional sense.
    Therefore, its arithmetic FLOPs are considered 0.
    """
    return 0

def calculate_conv2d_params(in_channels, out_channels, kernel_size, bias=True):
    """
    Calculate the number of parameters for a Conv2d layer.
    Parameters = (kernel_size * kernel_size * in_channels + 1 (for bias if present)) * out_channels
    """
    num_params = (kernel_size * kernel_size * in_channels) * out_channels
    if bias:
        num_params += out_channels # Add parameters for bias terms
    return num_params

def get_activation_size_mb(shape, bytes_per_element=4):
    """
    Calculates the memory size of a tensor in MB.
    Shape is expected to be (C, H, W) or (B, C, H, W).
    """
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    return (num_elements * bytes_per_element) / (1024 * 1024)


def calculate_model_metrics_and_latency(model, input_shape_chw, batch_size=1, device='cpu'):
    """
    Calculates FLOPs, total parameters, total forward pass memory,
    peak forward pass memory, and inference latency for a given PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.
        input_shape_chw (tuple): The shape of the input tensor (C, H, W)
                                 without the batch dimension.
        batch_size (int): The batch size for the input tensor.
        device (str): 'cpu' or 'cuda' to specify the device for measurement.

    Returns:
        tuple: (total_flops, total_params, total_forward_pass_memory_mb,
                peak_forward_pass_memory_mb, average_latency_ms)
    """
    total_flops = 0
    total_params = 0
    total_forward_pass_memory_bytes = 0 # Sum of memory for all intermediate activations
    peak_forward_pass_memory_bytes = 0   # Maximum memory used by any single intermediate activation

    # Create a dummy input tensor
    dummy_input = torch.randn(batch_size, *input_shape_chw).to(device)
    current_x = dummy_input

    # Add the memory of the input tensor itself to total and peak
    input_memory_bytes = dummy_input.numel() * dummy_input.element_size()
    total_forward_pass_memory_bytes += input_memory_bytes
    peak_forward_pass_memory_bytes = input_memory_bytes

    # Helper function to process layers within a Sequential module
    def process_sequential_module_for_metrics(sequential_module, current_x_tensor):
        nonlocal total_flops, total_params, total_forward_pass_memory_bytes, peak_forward_pass_memory_bytes

        for name, module in sequential_module.named_children():
            # Apply module to current_x_tensor to get output shape
            # Ensure model is in eval mode and no_grad for accurate tracing
            with torch.no_grad():
                prev_x_tensor = current_x_tensor # Store previous for some calculations (e.g., PixelShuffle input)
                current_x_tensor = module(current_x_tensor)

            # Calculate FLOPs and Params for the module
            if isinstance(module, nn.Conv2d):
                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_size = module.kernel_size[0] # Assuming square kernels
                output_h, output_w = current_x_tensor.shape[2], current_x_tensor.shape[3]
                
                flops = calculate_conv2d_flops(in_channels, out_channels, kernel_size, (output_h, output_w), module.stride)
                params = calculate_conv2d_params(in_channels, out_channels, kernel_size, bias=module.bias is not None)
                
                total_flops += flops
                total_params += params
            elif isinstance(module, nn.ReLU):
                flops = calculate_relu_flops(current_x_tensor.numel())
                total_flops += flops
            elif isinstance(module, nn.PixelShuffle):
                # PixelShuffle input's elements count before rearrangement
                flops = calculate_pixelshuffle_flops(prev_x_tensor.numel(), module.upscale_factor)
                total_flops += flops # This will add 0
            # Note: torch.clamp FLOPs are typically not counted as they are element-wise non-arithmetic.

            # Calculate memory for the output activation of this layer
            layer_output_memory_bytes = current_x_tensor.numel() * current_x_tensor.element_size()
            total_forward_pass_memory_bytes += layer_output_memory_bytes
            peak_forward_pass_memory_bytes = max(peak_forward_pass_memory_bytes, layer_output_memory_bytes)
        return current_x_tensor

    # Process encoder path
    current_x = process_sequential_module_for_metrics(model.encoder.encoder, current_x)
    
    # Process decoder path
    current_x = process_sequential_module_for_metrics(model.decoder.decoder, current_x)

    # Convert bytes to MB for readability
    total_forward_pass_memory_mb = total_forward_pass_memory_bytes / (1024 * 1024)
    peak_forward_pass_memory_mb = peak_forward_pass_memory_bytes / (1024 * 1024)

    # --- Latency Measurement ---
    NUM_WARMUP_RUNS = 10
    NUM_MEASUREMENT_RUNS = 100

    model.eval() # Set model to evaluation mode
    latencies = []

    # Warm-up runs
    for _ in range(NUM_WARMUP_RUNS):
        _ = model(dummy_input)
        if device == 'cuda':
            torch.cuda.synchronize()

    # Measurement runs
    with torch.no_grad():
        for _ in range(NUM_MEASUREMENT_RUNS):
            if device == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = model(dummy_input)
                end_event.record()
                torch.cuda.synchronize()
                latency = start_event.elapsed_time(end_event) # milliseconds
            else: # CPU
                start_time = time.perf_counter()
                _ = model(dummy_input)
                end_time = time.perf_counter()
                latency = (end_time - start_time) * 1000 # Convert to milliseconds
            latencies.append(latency)

    average_latency = sum(latencies) / NUM_MEASUREMENT_RUNS

    return total_flops, total_params, total_forward_pass_memory_mb, peak_forward_pass_memory_mb, average_latency


if __name__ == '__main__':
    n_updownconv = 3 # Number of downsampling/upsampling pairs after the initial conv
    in_chn = 3
    out_chn = 3

    # Check for GPU availability
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # List of input sizes (H, W) to test
    input_hw_sizes = [
        (360, 640),    # H, W (changed order to H, W to match typical usage in input_configs)
        (720, 1280),
        (760, 1360),
        (900, 1600),
        (1080, 1920),
        (1152, 2048),
        (1440, 2560),
        (2160, 3840),  # Corrected UHD/4K resolution (H, W)
    ]
    
    # Input configs for consistent printing format
    input_configs = [{'input_res': (in_chn, h, w), 'batch_size': 1} for h, w in input_hw_sizes]

    print(f"{'Input Size (CxHxW)':>22} | {'Params (M)':>10} | {'FLOPs (B)':>10} | {'Total Acts (MB)':>16} | {'Peak Act (MB)':>15} | {'Latency (ms)':>15}")
    print("-" * 120)

    for config in input_configs:
        input_channels = config['input_res'][0]
        input_height = config['input_res'][1]
        input_width = config['input_res'][2]
        batch_size = config['batch_size']

        # Initialize model for current configuration and move to device
        model = autoencoder(n_updownconv=n_updownconv, in_chn=input_channels, out_chn=out_chn).to(DEVICE)
        
        # Calculate metrics and latency
        flops, params, total_memory_mb, peak_memory_mb, avg_latency = \
            calculate_model_metrics_and_latency(model, config['input_res'], batch_size=batch_size, device=DEVICE)

        print(f"{input_channels}x{input_height}x{input_width:<5} | {params / 1e6:10.2f} | {flops / 1e9:10.2f} | {total_memory_mb:16.2f} | {peak_memory_mb:14.2f} | {avg_latency:14.5f}")