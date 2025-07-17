import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Network Architecture Definitions ---

class ConvBlockNoSkip(nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, pool=True):
        super(ConvBlockNoSkip, self).__init__()
        pad = (k_sz - 1) // 2
        self.out_channels = out_c # Store out_channels for calculation tracing
        block = []
        if pool:
            self.pool = nn.MaxPool2d(kernel_size=2)
        else:
            self.pool = False

        # First Conv -> ReLU -> BN
        block.append(nn.Conv2d(in_c, out_c, kernel_size=k_sz, padding=pad, bias=True)) # Explicitly use bias=True for calculation
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_c))

        # Second Conv -> ReLU -> BN
        block.append(nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding=pad, bias=True)) # Explicitly use bias=True for calculation
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
            self.up_path.append(nn.ConvTranspose2d(layers[i+1], layers[i], kernel_size=2, stride=2, bias=True))

        # Final conv layer after the upsampling path
        self.final_conv = nn.Conv2d(layers[0], in_c, kernel_size=1, bias=True)

    def forward(self, x):
        # Encoder
        x1 = self.first(x)
        feature_maps = [x1]

        x = x1
        for i, down in enumerate(self.down_path):
            x = down(x)
            feature_maps.append(x)

        # Decoder
        x = feature_maps[-1]
        for i, up in enumerate(reversed(self.up_path)):
            x = up(x)
            skip_connection = feature_maps[len(self.layers) - 2 - i]
            x = x + skip_connection

        # Final Convolution
        x = self.final_conv(x)
        return x, feature_maps

# --- Theoretical Calculation Functions (Adjusted to count MACs for ptflops alignment) ---

def calculate_conv_flops_params(input_shape, output_channels, kernel_size, stride, padding, bias=True):
    """Calculates MACs and Parameters for a standard Conv2d layer."""
    if len(input_shape) != 3:
        raise ValueError(f"Conv input shape must be (H, W, C), but got {input_shape}")
    in_channels = input_shape[2]
    k_h, k_w = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    s_h, s_w = (stride, stride) if isinstance(stride, int) else stride
    p_h, p_w = (padding, padding) if isinstance(padding, int) else padding

    out_height = (input_shape[0] + 2 * p_h - (k_h - 1) - 1) // s_h + 1
    out_width = (input_shape[1] + 2 * p_w - (k_w - 1) - 1) // s_w + 1

    if out_height <= 0 or out_width <= 0:
         print(f"Warning: Conv input shape {input_shape}, kernel {kernel_size}, stride {stride}, padding {padding} resulted in non-positive output dimensions: ({out_height}, {out_width}). Returning 0 for this layer.")
         return 0, 0, (max(0, out_height), max(0, out_width), output_channels)

    # Parameters: (kH * kW * in_channels + bias) * out_channels
    params = (k_h * k_w * in_channels + (1 if bias else 0)) * output_channels

    # FLOPs as MACs (Multiply-Accumulate operations)
    # Each output element requires (kH * kW * in_channels) MACs.
    total_macs = (k_h * k_w * in_channels) * out_height * out_width * output_channels
    # ptflops often includes bias additions as part of the MACs count for conv layers.
    if bias:
        total_macs += out_height * out_width * output_channels # Additions for bias

    total_flops = total_macs # Manual count of FLOPs will represent MACs for alignment

    return params, total_flops, (out_height, out_width, output_channels)

def calculate_bn_flops_params(input_shape):
    """Calculates Parameters for BatchNorm2d. FLOPs are 0 for alignment with ptflops' MMac."""
    if len(input_shape) != 3:
        raise ValueError(f"BN input shape must be (H, W, C), but got {input_shape}")

    num_features = input_shape[2]
    params = 2 * num_features # gamma and beta are learnable

    flops = 0 # Set to 0 for alignment with ptflops' common conventions
    return params, flops, input_shape # Output shape is the same

def calculate_relu_flops(input_shape):
    """Calculates FLOPs for ReLU activation (set to 0 for alignment with ptflops' MMac)."""
    if len(input_shape) != 3:
        raise ValueError(f"ReLU input shape must be (H, W, C), but got {input_shape}")

    flops = 0 # Set to 0 for alignment with ptflops' common conventions
    return 0, flops, input_shape # No parameters, output shape is the same

def calculate_pooling_flops_params(input_shape, kernel_size, stride):
    """MaxPool has 0 parameters and 0 FLOPs for alignment with ptflops' MMac."""
    if len(input_shape) != 3:
        raise ValueError(f"Pool input shape must be (H, W, C), but got {input_shape}")

    k_h, k_w = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    s_h, s_w = (stride, stride) if isinstance(stride, int) else stride

    out_height = (input_shape[0] - k_h) // s_h + 1
    out_width = (input_shape[1] - k_w) // s_w + 1

    return 0, 0, (out_height, out_width, input_shape[2]) # 0 params, 0 flops, updated shape

def calculate_convtranspose_flops_params(input_shape, output_channels, kernel_size, stride, padding, output_padding, bias=True):
    """Calculates MACs and Parameters for ConvTranspose2d, adjusted for alignment with ptflops."""
    if len(input_shape) != 3:
        raise ValueError(f"ConvTranspose input shape must be (H, W, C), but got {input_shape}")
    in_channels = input_shape[2]
    k_h, k_w = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    s_h, s_w = (stride, stride) if isinstance(stride, int) else stride
    p_h, p_w = (padding, padding) if isinstance(padding, int) else padding
    op_h, op_w = (output_padding, output_padding) if isinstance(output_padding, int) else output_padding

    out_height = (input_shape[0] - 1) * s_h - 2 * p_h + (k_h - 1) + op_h + 1
    out_width = (input_shape[1] - 1) * s_w - 2 * p_w + (k_w - 1) + op_w + 1

    if out_height <= 0 or out_width <= 0:
         print(f"Warning: ConvTranspose input shape {input_shape}, kernel {kernel_size}, stride {stride}, padding {padding}, output_padding {output_padding} resulted in non-positive output dimensions: ({out_height}, {out_width}). Returning 0 for this layer.")
         return 0, 0, (max(0, out_height), max(0, out_width), output_channels)

    # Parameters: (in_channels * kernel_height * kernel_width + bias) * output_channels
    params = (in_channels * k_h * k_w + (1 if bias else 0)) * output_channels

    # FLOPs as MACs for ConvTranspose2d.
    # This is often calculated as the number of MACs if it were a regular convolution
    # from the output features back to the input features.
    total_macs = out_height * out_width * output_channels * k_h * k_w * in_channels
    if bias:
        total_macs += out_height * out_width * output_channels # Additions for bias

    total_flops = total_macs # Manual count of FLOPs will represent MACs for alignment

    return params, total_flops, (out_height, out_width, output_channels)

def calculate_addition_flops(input_shape):
    """Calculates FLOPs for element-wise addition (set to 0 for alignment with ptflops' MMac)."""
    if len(input_shape) != 3:
        raise ValueError(f"Addition input shape must be (H, W, C), but got {input_shape}")

    flops = 0 # Set to 0 for alignment with ptflops' common conventions
    return 0, flops, input_shape # No parameters, output shape is the same


def trace_unet_with_skip_flops_params(model, input_res, in_ch, out_ch, layers, k_sz=3):  # H x W
    """Calculates total theoretical FLOPs (as MACs) and Parameters for the UNetEncoderWithSkip architecture by tracing."""
    total_params = 0
    total_flops = 0 # Now represents total MACs

    curr_shape = (input_res[0], input_res[1], in_ch)
    print(f"Input shape (H, W, C): {curr_shape}")

    encoder_skip_shapes = []

    print("\n--- Encoder ---")

    # First block: model.first (ConvBlockNoSkip with pool=False)
    print("Block: model.first (no pool)")
    conv_block = model.first.block

    layer_names = ['Conv1', 'ReLU1', 'BN1', 'Conv2', 'ReLU2', 'BN2']
    for i, module in enumerate(conv_block):
        layer_name = layer_names[i]
        if isinstance(module, nn.Conv2d):
            params, flops, curr_shape = calculate_conv_flops_params(
                curr_shape, module.out_channels, module.kernel_size[0], module.stride[0], module.padding[0], bias=module.bias is not None
            )
            total_params += params
            total_flops += flops # Add MACs
            #print(f"  - {layer_name}: Output shape: {curr_shape}, Params: {params}, FLOPs (MACs): {flops}")
        elif isinstance(module, nn.ReLU):
            params, flops, curr_shape = calculate_relu_flops(curr_shape)
            total_flops += flops # Should be 0
          #  print(f"  - {layer_name}: Output shape: {curr_shape}, Params: {params}, FLOPs (MACs): {flops} (ignored)")
        elif isinstance(module, nn.BatchNorm2d):
            params, flops, curr_shape = calculate_bn_flops_params(curr_shape)
            total_params += params
            total_flops += flops # Should be 0
            #print(f"  - {layer_name}: Output shape: {curr_shape}, Params: {params}, FLOPs (MACs): {flops} (ignored)")

    encoder_skip_shapes.append(curr_shape)

    # Down Path blocks: model.down_path
    for i, down_block_module in enumerate(model.down_path):
        #print(f"Block: model.down_path[{i}] (pool=True)")
        pool_layer = down_block_module.pool
        conv_block = down_block_module.block

        # Max Pooling (FLOPs ignored)
        params, flops, curr_shape_after_pool = calculate_pooling_flops_params(
             curr_shape, pool_layer.kernel_size, pool_layer.stride
        )
        total_params += params
        total_flops += flops
       # print(f"  - MaxPool: Input shape: {curr_shape}, Output shape: {curr_shape_after_pool}, Params: {params}, FLOPs (MACs): {flops} (ignored)")
        curr_shape = curr_shape_after_pool

        # ConvBlockNoSkip after pooling
        layer_names = ['Conv1', 'ReLU1', 'BN1', 'Conv2', 'ReLU2', 'BN2']
        for j, module in enumerate(conv_block):
            layer_name = layer_names[j]
            if isinstance(module, nn.Conv2d):
                 params, flops, curr_shape = calculate_conv_flops_params(
                     curr_shape, module.out_channels, module.kernel_size[0], module.stride[0], module.padding[0], bias=module.bias is not None
                 )
                 total_params += params
                 total_flops += flops # Add MACs
                # print(f"  - {layer_name}: Output shape: {curr_shape}, Params: {params}, FLOPs (MACs): {flops}")
            elif isinstance(module, nn.ReLU):
                 params, flops, curr_shape = calculate_relu_flops(curr_shape)
                 total_flops += flops # Should be 0
              #   print(f"  - {layer_name}: Output shape: {curr_shape}, Params: {params}, FLOPs (MACs): {flops} (ignored)")
            elif isinstance(module, nn.BatchNorm2d):
                 params, flops, curr_shape = calculate_bn_flops_params(curr_shape)
                 total_params += params
                 total_flops += flops # Should be 0
             #    print(f"  - {layer_name}: Output shape: {curr_shape}, Params: {params}, FLOPs (MACs): {flops} (ignored)")

        encoder_skip_shapes.append(curr_shape)


    print("\n--- Decoder ---")

    curr_shape = encoder_skip_shapes[-1]

    num_up_layers = len(model.up_path)
    for i in range(num_up_layers):
        up_layer = model.up_path[num_up_layers - 1 - i]
        skip_shape = encoder_skip_shapes[num_up_layers - 1 - i]

        print(f"Up Block: model.up_path[{num_up_layers - 1 - i}]")

        # ConvTranspose2d (Upsampling)
        params, flops, curr_shape_after_up = calculate_convtranspose_flops_params(
            curr_shape, up_layer.out_channels, up_layer.kernel_size[0], up_layer.stride[0], up_layer.padding[0], up_layer.output_padding[0], bias=up_layer.bias is not None
        )
        total_params += params
        total_flops += flops # Add MACs
        # Sanity check for spatial size matching skip connection
        if curr_shape_after_up[0] != skip_shape[0] or curr_shape_after_up[1] != skip_shape[1]:
            print(f"Warning: Upsample output shape {curr_shape_after_up} does not match skip shape {skip_shape} spatially for addition! Adjusting output shape for subsequent operations.")
            curr_shape_after_up = (skip_shape[0], skip_shape[1], curr_shape_after_up[2])

     #   print(f"  - ConvTranspose2d: Input shape: {curr_shape}, Output shape: {curr_shape_after_up}, Params: {params}, FLOPs (MACs): {flops}")
        curr_shape = curr_shape_after_up

        # Addition (Skip Connection) - FLOPs ignored for MACs count
        if curr_shape != skip_shape:
             print(f"Error: Final shape mismatch {curr_shape} vs {skip_shape} for addition!")
             addition_shape = curr_shape
        else:
             addition_shape = curr_shape

        params, flops, curr_shape_after_add = calculate_addition_flops(addition_shape)
        total_flops += flops # Should be 0
       # print(f"  - Addition (Skip): Input shape: {addition_shape}, Output shape: {curr_shape_after_add}, Params: {params}, FLOPs (MACs): {flops} (ignored)")
        curr_shape = curr_shape_after_add


    print("\n--- Final Convolution ---")
    final_conv_layer = model.final_conv
    params, flops, curr_shape = calculate_conv_flops_params(
        curr_shape, final_conv_layer.out_channels, final_conv_layer.kernel_size[0], final_conv_layer.stride[0], final_conv_layer.padding[0], bias=final_conv_layer.bias is not None
    )
    total_params += params
    total_flops += flops # Add MACs
  #  print(f"Layer final_conv: Input shape: {curr_shape}, Output shape: {curr_shape}, Params: {params}, FLOPs (MACs): {flops}")


    print("\n--- Summary ---")
    print(f"Input Resolution (HxW): {input_res}")
    print(f"Input Channels: {in_ch}")
    print(f"Output Channels (final_conv): {out_ch}")
    print(f"Total Parameters: {total_params / 1e3:.2f} K") # Changed to K for better comparison with ptflops
    print(f"Total FLOPs (MACs): {total_flops / 1e6:.2f} MMac") # Changed to MMac for better comparison
    print("-" * 40)


def run_manual_benchmarks():
    input_spatial_sizes = [
        (360, 640),    # H, W
        (720, 1280),   # H, W
        (760, 1360),   # H, W
        (900, 1600),   # H, W
        (1080, 1920),  # H, W
        (1152, 2048),  # H, W
        (1440, 2560),  # H, W
        (3840,2160),
    ]
    in_channels = 3
    actual_final_out_channels = in_channels

    layers = [4, 8, 16, 32]

    print("--- Running UNetEncoderWithSkip Complexity Benchmarks (Manual Trace - Adjusted for MACs) ---")
    print(f"Encoder Layers Channels: {layers}")
    print(f"Fixed Input Channels: {in_channels}")
    print(f"Actual Final Output Channels (from model): {actual_final_out_channels}")
    print("-" * 40)

    model = UNetEncoderWithSkip(in_c=in_channels, layers=layers, k_sz=3)

    for res in input_spatial_sizes:
        print(f"\n--- Calculating for Input Resolution: {res[0]}x{res[1]} ---")
        trace_unet_with_skip_flops_params(
            model,
            input_res=res,
            in_ch=in_channels,
            out_ch=actual_final_out_channels,
            layers=layers,
            k_sz=3
        )


if __name__ == "__main__":
    run_manual_benchmarks()