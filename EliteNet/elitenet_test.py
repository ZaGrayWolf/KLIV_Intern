import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import timeit # Useful for CPU timing, but less critical for GPU

# --- Network Architecture Definitions ---

class ConvBlockNoSkip(nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, pool=True):
        super(ConvBlockNoSkip, self).__init__()
        pad = (k_sz - 1) // 2
        self.out_channels = out_c # Store out_channels for calculation tracing
        block = []
        if pool:
            # Use ceil_mode=True for MaxPool2d to potentially simplify upsampling,
            # but dynamic cropping is still the most robust.
            self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=False) # Changed to False to match default, but keep in mind behavior.
        else:
            self.pool = False

        # First Conv -> ReLU -> BN
        block.append(nn.Conv2d(in_c, out_c, kernel_size=k_sz, padding=pad, bias=True))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_c))

        # Second Conv -> ReLU -> BN
        block.append(nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding=pad, bias=True))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_c))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        if self.pool:
            x = self.pool(x)
        out = self.block(x)
        return out


class UNetEncoderWithSkip(nn.Module):
    def __init__(self, in_c, layers, k_sz=3):
        """
        UNet encoder with skip connections using ConvTranspose2d and addition.
        Note: This is not a standard UNet decoder structure (which uses concatenation).
        """
        super(UNetEncoderWithSkip, self).__init__()
        self.first = ConvBlockNoSkip(in_c=in_c, out_c=layers[0], k_sz=k_sz, pool=False)
        self.layers = layers
        self.down_path = nn.ModuleList()
        self.up_path = nn.ModuleList()
        # Encoder Down Path
        for i in range(len(layers) - 1):
            block = ConvBlockNoSkip(in_c=layers[i], out_c=layers[i + 1], k_sz=k_sz, pool=True)
            self.down_path.append(block)
        # Decoder Up Path (using ConvTranspose2d for upsampling)
        for i in range(len(layers) - 1):
             # For robust upsampling in UNet-like architectures,
             # setting output_padding to 1 is often a good heuristic for cases
             # where a dimension might become one pixel smaller after pooling an odd dimension.
             # However, the subsequent cropping makes this less critical for correctness,
             # but it can optimize the transpose conv itself.
            self.up_path.append(nn.ConvTranspose2d(layers[i+1], layers[i], kernel_size=2, stride=2, bias=True, output_padding=0)) # Set back to 0 as cropping handles it.
            # If you *strictly* want output_padding to make it match the skip connection's size *before* cropping,
            # you'd need to calculate it. For simplicity, dynamic cropping is better.


        # Added a final conv layer after the upsampling path
        self.final_conv = nn.Conv2d(layers[0], in_c, kernel_size=1, bias=True)

    def forward(self, x):
        # Encoder
        x1 = self.first(x) # Output of the first block (no pool)
        feature_maps = [x1] # Store for skip connection

        x = x1 # Start processing down path from x1
        for i, down in enumerate(self.down_path):
            x = down(x) # down block includes pooling
            feature_maps.append(x) # Store output of each down block

        # Decoder
        x = feature_maps[-1] # Start decoder with the deepest feature map
        # Iterate through up_path in reverse order
        for i, up in enumerate(reversed(self.up_path)):
            x = up(x) # Upsample

            # Get the corresponding skip connection
            # The index is len(self.layers) - 2 - i (corresponds to skip_connection levels from shallowest to deepest in feature_maps list)
            skip_connection = feature_maps[len(self.layers) - 2 - i]

            # --- FIX: Dynamically crop the upsampled tensor (x) to match the skip_connection's spatial dimensions ---
            h_x, w_x = x.shape[2:]
            h_skip, w_skip = skip_connection.shape[2:]

            if h_x != h_skip or w_x != w_skip:
                # Calculate crop amounts
                # Assuming x is either same size or 1 pixel larger due to upsampling rules
                # The crop should remove (h_x - h_skip) and (w_x - w_skip) pixels
                # We want to crop equally from both sides (if possible)
                crop_h = h_x - h_skip
                crop_w = w_x - w_skip

                # Calculate start indices for slicing
                start_h = crop_h // 2
                start_w = crop_w // 2

                # Slice to perform the crop
                # [:, :, start_h : start_h + target_h, start_w : start_w + target_w]
                x = x[:, :, start_h:start_h + h_skip, start_w:start_w + w_skip]

            # Now, x and skip_connection should have identical spatial dimensions
            x = x + skip_connection # Element-wise addition

        # Final Convolution
        x = self.final_conv(x)
        return x

# --- Theoretical Calculation Functions (kept as is, but ensure logic aligns with model) ---

def calculate_conv_flops_params(input_shape, output_channels, kernel_size, stride, padding, bias=True):
    if len(input_shape) != 3: raise ValueError(f"Conv input shape must be (H, W, C), but got {input_shape}")
    in_channels = input_shape[2]
    k_h, k_w = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    s_h, s_w = (stride, stride) if isinstance(stride, int) else stride
    p_h, p_w = (padding, padding) if isinstance(padding, int) else padding

    out_height = (input_shape[0] + 2 * p_h - k_h) // s_h + 1
    out_width = (input_shape[1] + 2 * p_w - k_w) // s_w + 1

    if out_height <= 0 or out_width <= 0:
        return 0, 0, (max(0, out_height), max(0, out_width), output_channels)

    params = (k_h * k_w * in_channels + (1 if bias else 0)) * output_channels
    mults = (k_h * k_w * in_channels) * out_height * out_width * output_channels
    adds = (k_h * k_w * in_channels - 1) * out_height * out_width * output_channels if (k_h * k_w * in_channels) > 0 else 0
    bias_adds = out_height * out_width * output_channels if bias else 0
    total_flops = mults + adds + bias_adds
    return params, total_flops, (out_height, out_width, output_channels)

def calculate_bn_flops_params(input_shape):
    if len(input_shape) != 3: raise ValueError(f"BN input shape must be (H, W, C), but got {input_shape}")
    num_features = input_shape[2]
    params = 2 * num_features
    total_elements = input_shape[0] * input_shape[1] * input_shape[2]
    flops = 2 * total_elements
    return params, flops, input_shape

def calculate_relu_flops(input_shape):
    if len(input_shape) != 3: raise ValueError(f"ReLU input shape must be (H, W, C), but got {input_shape}")
    total_elements = input_shape[0] * input_shape[1] * input_shape[2]
    flops = total_elements
    return 0, flops, input_shape

def calculate_pooling_flops_params(input_shape, kernel_size, stride):
    if len(input_shape) != 3: raise ValueError(f"Pool input shape must be (H, W, C), but got {input_shape}")
    k_h, k_w = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    s_h, s_w = (stride, stride) if isinstance(stride, int) else stride
    out_height = (input_shape[0] - k_h) // s_h + 1
    out_width = (input_shape[1] - k_w) // s_w + 1
    return 0, 0, (out_height, out_width, input_shape[2])

def calculate_convtranspose_flops_params(input_shape, output_channels, kernel_size, stride, padding, output_padding, bias=True):
    if len(input_shape) != 3: raise ValueError(f"ConvTranspose input shape must be (H, W, C), but got {input_shape}")
    in_channels = input_shape[2]
    k_h, k_w = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    s_h, s_w = (stride, stride) if isinstance(stride, int) else stride
    p_h, p_w = (padding, padding) if isinstance(padding, int) else padding
    op_h, op_w = (output_padding, output_padding) if isinstance(output_padding, int) else output_padding

    out_height = (input_shape[0] - 1) * s_h - 2 * p_h + k_h + op_h
    out_width = (input_shape[1] - 1) * s_w - 2 * p_w + k_w + op_w

    if out_height <= 0 or out_width <= 0:
        return 0, 0, (max(0, out_height), max(0, out_width), output_channels)

    params = (in_channels * k_h * k_w + (1 if bias else 0)) * output_channels
    total_output_elements = out_height * out_width * output_channels
    macs = input_shape[0] * input_shape[1] * k_h * k_w * in_channels * output_channels
    total_flops = 2 * macs
    bias_adds = total_output_elements if bias else 0
    total_flops += bias_adds

    return params, total_flops, (out_height, out_width, output_channels)


def calculate_addition_flops(input_shape):
    if len(input_shape) != 3: raise ValueError(f"Addition input shape must be (H, W, C), but got {input_shape}")
    total_elements = input_shape[0] * input_shape[1] * input_shape[2]
    flops = total_elements
    return 0, flops, input_shape

def get_activation_size_mb(shape, bytes_per_element=4):
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    return (num_elements * bytes_per_element) / (1024 * 1024)


# --- Unified Theoretical Calculation and Activation Tracking ---
def get_model_metrics_theoretical(model_instance, input_res_chw, layers, k_sz=3):
    """
    Calculates total theoretical FLOPs, Parameters, and Activation Size for the UNetEncoderWithSkip architecture by tracing.
    input_res_chw: (C, H, W) tuple from the config.
    """
    total_params = 0
    total_flops = 0
    total_activation_sum_mb = 0
    peak_activation_mb = 0

    in_ch = input_res_chw[0]
    input_height = input_res_chw[1]
    input_width = input_res_chw[2]

    curr_shape_hwc = (input_height, input_width, in_ch)
    initial_input_activation_size = get_activation_size_mb(curr_shape_hwc)
    total_activation_sum_mb += initial_input_activation_size
    peak_activation_mb = max(peak_activation_mb, initial_input_activation_size)

    encoder_skip_shapes_hwc = []

    # --- Encoder Tracing ---
    conv_block_first = model_instance.first.block
    for module in conv_block_first:
        if isinstance(module, nn.Conv2d):
            p, f, curr_shape_hwc = calculate_conv_flops_params(
                curr_shape_hwc, module.out_channels, module.kernel_size[0], module.stride[0], module.padding[0], bias=module.bias is not None
            )
            total_params += p; total_flops += f
            current_act_size = get_activation_size_mb(curr_shape_hwc)
            total_activation_sum_mb += current_act_size
            peak_activation_mb = max(peak_activation_mb, current_act_size)
        elif isinstance(module, nn.ReLU):
            _, f, curr_shape_hwc = calculate_relu_flops(curr_shape_hwc)
            total_flops += f
            current_act_size = get_activation_size_mb(curr_shape_hwc)
            total_activation_sum_mb += current_act_size
            peak_activation_mb = max(peak_activation_mb, current_act_size)
        elif isinstance(module, nn.BatchNorm2d):
            p, f, curr_shape_hwc = calculate_bn_flops_params(curr_shape_hwc)
            total_params += p; total_flops += f
            current_act_size = get_activation_size_mb(curr_shape_hwc)
            total_activation_sum_mb += current_act_size
            peak_activation_mb = max(peak_activation_mb, current_act_size)
    encoder_skip_shapes_hwc.append(curr_shape_hwc)

    for i, down_block_module in enumerate(model_instance.down_path):
        pool_layer = down_block_module.pool
        # For theoretical calculation of pooling, we need to consider if it uses ceil_mode.
        # Your MaxPool2d in ConvBlockNoSkip is set to `ceil_mode=False` explicitly.
        _, _, curr_shape_hwc_after_pool = calculate_pooling_flops_params(
             curr_shape_hwc, pool_layer.kernel_size, pool_layer.stride
        )
        current_act_size = get_activation_size_mb(curr_shape_hwc_after_pool)
        total_activation_sum_mb += current_act_size
        peak_activation_mb = max(peak_activation_mb, current_act_size)
        curr_shape_hwc = curr_shape_hwc_after_pool

        conv_block_down = down_block_module.block
        for module in conv_block_down:
            if isinstance(module, nn.Conv2d):
                p, f, curr_shape_hwc = calculate_conv_flops_params(
                    curr_shape_hwc, module.out_channels, module.kernel_size[0], module.stride[0], module.padding[0], bias=module.bias is not None
                )
                total_params += p; total_flops += f
                current_act_size = get_activation_size_mb(curr_shape_hwc)
                total_activation_sum_mb += current_act_size
                peak_activation_mb = max(peak_activation_mb, current_act_size)
            elif isinstance(module, nn.ReLU):
                _, f, curr_shape_hwc = calculate_relu_flops(curr_shape_hwc)
                total_flops += f
                current_act_size = get_activation_size_mb(curr_shape_hwc)
                total_activation_sum_mb += current_act_size
                peak_activation_mb = max(peak_activation_mb, current_act_size)
            elif isinstance(module, nn.BatchNorm2d):
                p, f, curr_shape_hwc = calculate_bn_flops_params(curr_shape_hwc)
                total_params += p; total_flops += f
                current_act_size = get_activation_size_mb(curr_shape_hwc)
                total_activation_sum_mb += current_act_size
                peak_activation_mb = max(peak_activation_mb, current_act_size)
        encoder_skip_shapes_hwc.append(curr_shape_hwc)


    # --- Decoder Tracing ---
    curr_shape_hwc = encoder_skip_shapes_hwc[-1] # Start decoder with deepest feature map

    num_up_layers = len(model_instance.up_path)
    for i in range(num_up_layers):
        up_layer = model_instance.up_path[num_up_layers - 1 - i]
        skip_shape_hwc = encoder_skip_shapes_hwc[num_up_layers - 1 - i]

        p, f, curr_shape_hwc_after_up_transpose = calculate_convtranspose_flops_params(
            curr_shape_hwc, up_layer.out_channels, up_layer.kernel_size[0], up_layer.stride[0], up_layer.padding[0], up_layer.output_padding[0], bias=up_layer.bias is not None
        )
        total_params += p; total_flops += f
        current_act_size = get_activation_size_mb(curr_shape_hwc_after_up_transpose)
        total_activation_sum_mb += current_act_size
        peak_activation_mb = max(peak_activation_mb, current_act_size)
        
        # FIX: The theoretical calculation of the shape *after* ConvTranspose2d
        # must also account for the subsequent cropping that happens in the forward pass.
        # So, the shape for the addition and subsequent layers will be the *cropped* shape.
        
        # Apply the same cropping logic as in the forward pass to theoretical shapes
        h_x, w_x, c_x = curr_shape_hwc_after_up_transpose
        h_skip, w_skip, c_skip = skip_shape_hwc # Channels should match already

        if h_x != h_skip or w_x != w_skip:
            crop_h = h_x - h_skip
            crop_w = w_x - w_skip

            # Assuming the cropped tensor will match the skip connection's size
            curr_shape_hwc = (h_skip, w_skip, c_x)
            # Re-calculate activation size after "theoretical crop"
            current_act_size = get_activation_size_mb(curr_shape_hwc)
            total_activation_sum_mb += current_act_size # Account for the cropped activation
            peak_activation_mb = max(peak_activation_mb, current_act_size)
        else:
            curr_shape_hwc = curr_shape_hwc_after_up_transpose


        # Addition (Skip Connection)
        _, f, curr_shape_hwc = calculate_addition_flops(curr_shape_hwc) # Shape stays same after add
        total_flops += f
        current_act_size = get_activation_size_mb(curr_shape_hwc)
        total_activation_sum_mb += current_act_size
        peak_activation_mb = max(peak_activation_mb, current_act_size)


    # --- Final Convolution Layer ---
    final_conv_layer = model_instance.final_conv
    p, f, curr_shape_hwc = calculate_conv_flops_params(
        curr_shape_hwc, final_conv_layer.out_channels, final_conv_layer.kernel_size[0], final_conv_layer.stride[0], final_conv_layer.padding[0], bias=final_conv_layer.bias is not None
    )
    total_params += p; total_flops += f
    current_act_size = get_activation_size_mb(curr_shape_hwc)
    total_activation_sum_mb += current_act_size
    peak_activation_mb = max(peak_activation_mb, current_act_size)

    return total_params, total_flops, total_activation_sum_mb, peak_activation_mb


# --- Benchmarking Setup ---

# Check for GPU availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

input_configs = [
    {'input_res': (3, 640, 360), 'batch_size': 1}, # C, H, W
    {'input_res': (3, 1280, 720), 'batch_size': 1},
    {'input_res': (3, 1360, 760), 'batch_size': 1}, # This input likely caused the error
    {'input_res': (3, 1600, 900), 'batch_size': 1},
    {'input_res': (3, 1920, 1080), 'batch_size': 1},
    {'input_res': (3, 2048, 1152), 'batch_size': 1},
    {'input_res': (3, 2560, 1440), 'batch_size': 1},
    {'input_res': (3, 3840, 2160), 'batch_size': 1}

]

# Define the number of channels per level in the encoder
MODEL_LAYERS = [4, 8, 16, 32] # Example layer structure

NUM_WARMUP_RUNS = 5
NUM_MEASUREMENT_RUNS = 40

print(f"{'Input Size (CxHxW)':>22} | {'Params (M)':>10} | {'FLOPs (B)':>10} | {'Total Acts (MB)':>16} | {'Peak Act (MB)':>15} | {'Latency (ms)':>15}")
print("-" * 120)

for config in input_configs:
    input_channels = config['input_res'][0]
    input_height = config['input_res'][1]
    input_width = config['input_res'][2]
    batch_size = config['batch_size']

    # Initialize model for current configuration
    model = UNetEncoderWithSkip(in_c=input_channels, layers=MODEL_LAYERS, k_sz=3).to(DEVICE)
    model.eval() # Set model to evaluation mode

    # Create dummy input tensor
    dummy_input = torch.randn(batch_size, input_channels, input_height, input_width).to(DEVICE)

    # --- Get theoretical metrics ---
    params, flops, total_act_sum_mb, peak_act_mb = get_model_metrics_theoretical(
        model,
        input_res_chw=config['input_res'], # Pass the original (C, H, W) tuple here
        layers=MODEL_LAYERS,
        k_sz=3
    )

    # --- Warm-up runs for actual latency measurement ---
    for _ in range(NUM_WARMUP_RUNS):
        _ = model(dummy_input)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()

    # --- Measurement runs for actual latency ---
    latencies = []
    if DEVICE.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in range(NUM_MEASUREMENT_RUNS):
            if DEVICE.type == 'cuda':
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

    print(f"{config['input_res'][0]}x{config['input_res'][1]}x{config['input_res'][2]:<5} | {params / 1e6:10.2f} | {flops / 1e9:10.2f} | {total_act_sum_mb:16.2f} | {peak_act_mb:14.2f} | {average_latency:14.5f}")