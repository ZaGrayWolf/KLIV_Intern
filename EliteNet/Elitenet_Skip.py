import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlockNoSkip(nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, pool=True):
        super(ConvBlockNoSkip, self).__init__()
        pad = (k_sz - 1) // 2
        self.out_channels = out_c
        block = []
        if pool:
            self.pool = nn.MaxPool2d(kernel_size=2)
        else:
            self.pool = False

        block.append(nn.Conv2d(in_c, out_c, kernel_size=k_sz, padding=pad))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_c))

        block.append(nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding=pad))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_c))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        if self.pool:
            x = self.pool(x)
        out = self.block(x)
        return out

class UNetEncoderNoSkip(nn.Module):
    def __init__(self, in_c, layers, k_sz=3):
        """
        Encoder-only UNet model without skip connections.
        """
        super(UNetEncoderNoSkip, self).__init__()
        self.first = ConvBlockNoSkip(in_c=in_c, out_c=layers[0], k_sz=k_sz, pool=False)
        self.layers = layers
        self.down_path = nn.ModuleList()
        for i in range(len(layers) - 1):
            block = ConvBlockNoSkip(in_c=layers[i], out_c=layers[i + 1], k_sz=k_sz, pool=True)
            self.down_path.append(block)

    def forward(self, x):
        x = self.first(x)
        feature_maps = [x]
        for down in self.down_path:
            x = down(x)
            feature_maps.append(x)
        return feature_maps



class UNetEncoderWithSkip(nn.Module):
    def __init__(self, in_c, layers, k_sz=3):
        """
        UNet encoder with skip connections.
        """
        super(UNetEncoderWithSkip, self).__init__()
        self.first = ConvBlockNoSkip(in_c=in_c, out_c=layers[0], k_sz=k_sz, pool=False)
        self.layers = layers
        self.down_path = nn.ModuleList()
        self.up_path = nn.ModuleList() # Added up_path for skip connections
        for i in range(len(layers) - 1):
            block = ConvBlockNoSkip(in_c=layers[i], out_c=layers[i + 1], k_sz=k_sz, pool=True)
            self.down_path.append(block)
            self.up_path.append(nn.ConvTranspose2d(layers[i+1], layers[i], kernel_size=2, stride=2)) #added upsampling

        self.final_conv = nn.Conv2d(layers[0], in_c, kernel_size=1) # Added a final conv layer.

    def forward(self, x):
        x1 = self.first(x)
        feature_maps = [x1]
        for i, down in enumerate(self.down_path):
            x = down(x1)
            feature_maps.append(x)
            x1 = x
        
        x = feature_maps[-1]
        for i, up in enumerate(reversed(self.up_path)):
            x = up(x)
            x = x + feature_maps[len(self.layers) - 2 - i] #changed
        
        x = self.final_conv(x)
        return x, feature_maps #returning feature maps for parameter calculation

def calculate_conv2d_flops(in_channels, out_channels, kernel_size, output_size):
    """
    Calculate FLOPs for a single 2D convolution operation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the square kernel (e.g., 3 for 3x3).
        output_size (tuple):  (H, W) of the output feature map.

    Returns:
        int: FLOPs for the convolution operation.
    """
    flops_per_element = (2 * in_channels * kernel_size * kernel_size)
    output_height, output_width = output_size
    total_flops = flops_per_element * out_channels * output_height * output_width
    return total_flops

def calculate_batchnorm_flops(out_channels, output_size):
    """
    Calculate FLOPs for a BatchNorm2d operation.

    Args:
        out_channels (int): Number of output channels (number of features).
        output_size (tuple): (H, W) of the output feature map.

    Returns:
        int: FLOPs for the BatchNorm2d operation.
    """
    output_height, output_width = output_size
    total_flops = 2 * out_channels * output_height * output_width
    return total_flops

def calculate_relu_flops(output_size, output_channels):
    """
    Calculate FLOPs for ReLU operation.

    Args:
        output_size (tuple): (H, W) of the output feature map.
        output_channels: number of channels

    Returns:
        int: FLOPs for the ReLU operation
    """
    h, w = output_size
    return output_channels * h * w

def calculate_maxpool_flops(input_size, kernel_size):
    """
    Calculate FLOPs for MaxPool2d operation.

    Args:
      input_size (tuple): (C, H, W) of the input
      kernel_size (int): Kernel size of the pooling operation

    Returns:
      int: FLOPs for the MaxPool2d operation
    """
    c, h_in, w_in = input_size
    h_out = (h_in - kernel_size) // kernel_size + 1
    w_out = (w_in - kernel_size) // kernel_size + 1
    flops_per_element = kernel_size * kernel_size
    total_flops = c * h_out * w_out * flops_per_element
    return total_flops

def calculate_conv2d_params(in_channels, out_channels, kernel_size):
    """
    Calculate the number of parameters in a Conv2d layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the square kernel.

    Returns:
        int: Number of parameters.
    """
    return out_channels * (in_channels * kernel_size * kernel_size + 1)

def calculate_batchnorm_params(out_channels):
    """
    Calculate the number of parameters in a BatchNorm2d layer.

    Args:
        out_channels (int): Number of output channels.

    Returns:
        int: Number of parameters.
    """
    return 2 * out_channels

def calculate_convtranspose2d_params(in_channels, out_channels, kernel_size):
    """
    Calculate the number of parameters in a ConvTranspose2d layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the square kernel.

    Returns:
        int: Number of parameters.
    """
    return in_channels * (out_channels * kernel_size * kernel_size + 1)

def calculate_model_flops_and_params(model, input_shape):
    """
    Calculate the total FLOPs and parameters for the given model.

    Args:
        model (nn.Module): The PyTorch model.
        input_shape (tuple): The shape of the input tensor (C, H, W).

    Returns:
        tuple: (total_flops, total_params)
            total_flops (int): Total FLOPs.
            total_params (int): Total number of parameters.
    """
    total_flops = 0
    total_params = 0
    c, h, w = input_shape
    x = torch.randn(1, c, h, w)
    feature_map_sizes = [(h, w)]
    # Go through the layers manually
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size[0]
            output_size = feature_map_sizes.pop(0)
            flops = calculate_conv2d_flops(in_channels, out_channels, kernel_size, output_size)
            params = calculate_conv2d_params(in_channels, out_channels, kernel_size)
            total_flops += flops
            total_params += params
            h_out = (output_size[0] + 2 * module.padding[0] - (module.dilation[0] * (module.kernel_size[0] - 1) + 1)) // module.stride[0] + 1
            w_out = (output_size[1] + 2 * module.padding[1] - (module.dilation[1] * (module.kernel_size[1] - 1) + 1)) // module.stride[1] + 1
            feature_map_sizes.append((h_out, w_out))

        elif isinstance(module, nn.BatchNorm2d):
            out_channels = module.num_features
            output_size = feature_map_sizes[-1]
            flops = calculate_batchnorm_flops(out_channels, output_size)
            params = calculate_batchnorm_params(out_channels)
            total_flops += flops
            total_params += params

        elif isinstance(module, nn.ReLU):
            output_size = feature_map_sizes[-1]
            parent_name = name.split('.')[0]
            if parent_name == 'first':
                out_channels = model.first.out_channels
            elif 'down_path' in parent_name:
                index = int(name.split('.')[1])
                out_channels = model.down_path[index].out_channels
            flops = calculate_relu_flops(output_size, out_channels)
            total_flops += flops

        elif isinstance(module, nn.MaxPool2d):
            input_size_for_pool = (c, feature_map_sizes[-1][0], feature_map_sizes[-1][1])
            flops = calculate_maxpool_flops(input_size_for_pool, module.kernel_size)
            total_flops += flops
            h_out = (feature_map_sizes[-1][0] - module.kernel_size) // module.kernel_size + 1
            w_out = (feature_map_sizes[-1][1] - module.kernel_size) // module.kernel_size + 1
            feature_map_sizes[-1] = (h_out, w_out)
            feature_map_sizes[-1] = (h_out, w_out)

        elif isinstance(module, nn.ConvTranspose2d):
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size[0]
            output_size = feature_map_sizes.pop() # important
            params = calculate_convtranspose2d_params(in_channels, out_channels, kernel_size)
            total_params += params
            h_out = (output_size[0] - 1) * module.stride[0] - 2 * module.padding[0] + module.dilation[0] * (module.kernel_size[0] - 1) + module.output_padding[0] + 1
            w_out = (output_size[1] - 1) * module.stride[1] - 2 * module.padding[1] + module.dilation[1] * (module.kernel_size[1] - 1) + module.output_padding[1] + 1
            feature_map_sizes.append((h_out, w_out))
        elif isinstance(module, nn.Conv2d) and name == 'final_conv':
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size[0]
            output_size = feature_map_sizes.pop()
            flops = calculate_conv2d_flops(in_channels, out_channels, kernel_size, output_size)
            params = calculate_conv2d_params(in_channels, out_channels, kernel_size)
            total_flops += flops
            total_params += params
            h_out = (output_size[0] + 2 * module.padding[0] - (module.dilation[0] * (module.kernel_size[0] - 1) + 1)) // module.stride[0] + 1
            w_out = (output_size[1] + 2 * module.padding[1] - (module.dilation[1] * (module.kernel_size[1] - 1) + 1)) // module.stride[1] + 1
            feature_map_sizes.append((h_out, w_out))
    return total_flops, total_params

# Example usage:
in_channels = 3
layers = [4, 8, 16, 32]
input_size = (in_channels, 640,360)

# Instantiate the model *with* skip connections
model = UNetEncoderWithSkip(in_c=in_channels, layers=layers)

# Calculate FLOPs and parameters
flops, params = calculate_model_flops_and_params(model, input_size)

# Print the results
print(f"Model: UNetEncoder (With Skip Connections)")
print(f"Input Shape: {input_size}")
print(f"Total FLOPs: {flops:.2e}")
print(f"Total Parameters: {params:.2e}")
