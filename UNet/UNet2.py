import time
import timeit # Import timeit for accurate benchmarking

def calculate_flops_and_params(input_shape, output_channels, kernel_size, stride, padding=1):
    in_channels = input_shape[2]
    out_height = (input_shape[0] - kernel_size[0] + 2 * padding) // stride[0] + 1
    out_width = (input_shape[1] - kernel_size[1] + 2 * padding) // stride[1] + 1

    # Ensure valid output dimensions (prevent negative values from division)
    if out_height <= 0 or out_width <= 0:
        return 0, 0, (max(0, out_height), max(0, out_width), output_channels)

    # Parameters
    params = (kernel_size[0] * kernel_size[1] * in_channels + 1) * output_channels

    # FLOPs (using simplified common interpretation)
    # MACs = Multiplications + Additions. A single MAC is often counted as 2 FLOPs (1 mult, 1 add).
    # Here, you have mults and adds separately. Let's keep your original detailed breakdown.
    num_macs_per_output_element = kernel_size[0] * kernel_size[1] * in_channels

    mults = num_macs_per_output_element * out_height * out_width * output_channels
    # divs: Typically 0 for standard convolutions. If you meant batch norm-like divs, add it.
    # Given your original model, this isn't clear for conv. Let's assume it's part of your FLOPs definition.
    divs = 0 # Most convs don't have explicit divisions. Removed from calculation based on typical conv FLOPs.
             # If you intended this for something else (e.g., BN), it should be in its own function.

    add_subs = (num_macs_per_output_element - 1) * out_height * out_width * output_channels if num_macs_per_output_element > 0 else 0

    total_flops = mults + add_subs # Common FLOPs definition is MACs, or mults + adds.
                                # Your original code had "divs" which is unusual for conv. Let's stick to the common interpretation.
    output_shape = (out_height, out_width, output_channels)

    return params, total_flops, output_shape


def total_unet_encoder_flops_params(input_res):
    layers_config = [] # Store configurations for each effective layer
    curr_shape = (input_res[1], input_res[2], input_res[0])  # H x W x C (corrected from C,H,W to H,W,C for consistency)

    # enc1: 2 convs + MaxPool
    # First conv
    layers_config.append({'input_shape': curr_shape, 'output_channels': 64, 'kernel_size': (3, 3), 'stride': (1, 1), 'padding': 1})
    curr_shape = (curr_shape[0], curr_shape[1], 64) # Output shape after conv
    # Second conv
    layers_config.append({'input_shape': curr_shape, 'output_channels': 64, 'kernel_size': (3, 3), 'stride': (1, 1), 'padding': 1})
    curr_shape = (curr_shape[0], curr_shape[1], 64) # Output shape after conv
    # MaxPool
    curr_shape = (curr_shape[0] // 2, curr_shape[1] // 2, 64)

    # enc2: 2 convs + MaxPool
    # First conv
    layers_config.append({'input_shape': curr_shape, 'output_channels': 128, 'kernel_size': (3, 3), 'stride': (1, 1), 'padding': 1})
    curr_shape = (curr_shape[0], curr_shape[1], 128)
    # Second conv
    layers_config.append({'input_shape': curr_shape, 'output_channels': 128, 'kernel_size': (3, 3), 'stride': (1, 1), 'padding': 1})
    curr_shape = (curr_shape[0], curr_shape[1], 128)
    # MaxPool
    curr_shape = (curr_shape[0] // 2, curr_shape[1] // 2, 128)

    # enc3: 2 convs + MaxPool
    # First conv
    layers_config.append({'input_shape': curr_shape, 'output_channels': 256, 'kernel_size': (3, 3), 'stride': (1, 1), 'padding': 1})
    curr_shape = (curr_shape[0], curr_shape[1], 256)
    # Second conv
    layers_config.append({'input_shape': curr_shape, 'output_channels': 256, 'kernel_size': (3, 3), 'stride': (1, 1), 'padding': 1})
    curr_shape = (curr_shape[0], curr_shape[1], 256)
    # MaxPool
    curr_shape = (curr_shape[0] // 2, curr_shape[1] // 2, 256)

    # enc4: 2 convs (no final MaxPool for the typical UNet bottleneck)
    # First conv
    layers_config.append({'input_shape': curr_shape, 'output_channels': 512, 'kernel_size': (3, 3), 'stride': (1, 1), 'padding': 1})
    curr_shape = (curr_shape[0], curr_shape[1], 512)
    # Second conv
    layers_config.append({'input_shape': curr_shape, 'output_channels': 512, 'kernel_size': (3, 3), 'stride': (1, 1), 'padding': 1})
    # curr_shape = (curr_shape[0], curr_shape[1], 512) # This is the final output shape of the encoder part

    total_params = 0
    total_flops = 0
    total_activation_bytes = 0

    # Calculate for each layer
    current_activation_shape = (input_res[1], input_res[2], input_res[0]) # Start with input activation
    total_activation_bytes += current_activation_shape[0] * current_activation_shape[1] * current_activation_shape[2] * 4

    for layer_params in layers_config:
        params, flops, out_shape = calculate_flops_and_params(
            input_shape=layer_params['input_shape'],
            output_channels=layer_params['output_channels'],
            kernel_size=layer_params['kernel_size'],
            stride=layer_params['stride'],
            padding=layer_params['padding']
        )
        total_params += params
        total_flops += flops

        # Accumulate activation memory for the *output* of each layer
        # This is a common way to estimate peak memory or total memory used for activations during a forward pass.
        activation_mem = out_shape[0] * out_shape[1] * out_shape[2] * 4
        total_activation_bytes += activation_mem # Summing activation memory

        # Update the input shape for the next layer if it's not explicitly in layers_config
        # (This is implicitly handled by `layers_config` creation now, but important for general case)
        # For pooling layers, the flops are usually negligible, and params are 0.
        # MaxPool is handled by the curr_shape manipulation above.

    total_activation_MB = total_activation_bytes / (1024 ** 2) # Convert to MB

    return total_params, total_flops, total_activation_MB


# --- Run for all input sizes and calculate latency ----
input_sizes = [
    (3, 640, 360), # Channels, Height, Width
    (3, 1280, 720),
    (3, 1360, 760),
    (3, 1600, 900),
    (3, 1920, 1080),
    (3, 2048, 1152),
    (3, 2560, 1440)
]

print(f"{'Input Size (CxHxW)':>22} | {'Params (M)':>10} | {'FLOPs (B)':>10} | {'Activations (MB)':>18} | {'Latency (ms)':>15}")
print("-" * 90)

for inp in input_sizes:
    # Get the theoretical calculations first
    params, flops, act_mem = total_unet_encoder_flops_params(inp)

    # Use timeit to measure latency for the function call
    setup_code = f"""
from __main__ import total_unet_encoder_flops_params
input_tuple = {inp}
    """
    stmt = "total_unet_encoder_flops_params(input_tuple)"

    timer = timeit.Timer(stmt=stmt, setup=setup_code)
    # repeat=105, number=1: Run the statement once, repeat the entire process 105 times
    raw_times = timer.repeat(repeat=105, number=1)

    # Convert to milliseconds and sort to easily get the 'last 100' best readings
    times_ms = sorted([t * 1000 for t in raw_times])
    average_latency = sum(times_ms[5:]) / 100

    # Note: Using :.5f for latency to show more precision if numbers are very small.
    print(f"{inp[0]}x{inp[1]}x{inp[2]:<5} | {params / 1e6:10.2f} | {flops / 1e9:10.2f} | {act_mem:18.2f} | {average_latency:14.5f}")