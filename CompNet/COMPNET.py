import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.shape_input = list(x.shape)
        x = self.encoder(x)
        self.shape_latent = list(x.shape)
        x = self.decoder(x)
        return x

def calculate_conv2d_flops(in_channels, out_channels, kernel_size, output_size):
    flops_per_element = (2 * in_channels * kernel_size * kernel_size)
    output_height, output_width = output_size
    total_flops = flops_per_element * out_channels * output_height * output_width
    return total_flops

def calculate_relu_flops(output_size, output_channels):
    h, w = output_size
    return output_channels * h * w

def calculate_conv2d_params(in_channels, out_channels, kernel_size):
    return out_channels * (in_channels * kernel_size * kernel_size + 1)

def calculate_pixelshuffle_flops(in_channels, upscale_factor, output_size):
    """
    Calculate FLOPs for PixelShuffle operation.

    Args:
        in_channels (int): Number of input channels.
        upscale_factor (int): Upscale factor.
        output_size (tuple): (H, W) of the output feature map.

    Returns:
        int: FLOPs for the PixelShuffle operation.
    """
    _, h, w = output_size
    # Each output pixel is derived from upscale_factor * upscale_factor pixels
    # in the input. The operation involves rearranging data, which doesn't
    # involve arithmetic operations in the traditional sense.
    # However, for the sake of accounting for the data movement, we can
    # approximate the FLOPs as the number of output elements.
    flops =  h * w * in_channels * upscale_factor * upscale_factor
    return flops
def calculate_model_flops_and_params(model, input_shape):
    total_flops = 0
    total_params = 0
    c, h, w = input_shape
    x = torch.randn(1, c, h, w)
    feature_map_sizes = [(h, w)]
    in_channels = c # Keep track of the input channels.
    out_channels = c

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
            w_out = (output_size[1] + 2 * module.padding[1] - (module.dilation[0] * (module.kernel_size[0] - 1) + 1)) // module.stride[0] + 1
            feature_map_sizes.append((h_out, w_out))
        elif isinstance(module, nn.ReLU):
            output_size = feature_map_sizes[-1]
            flops = calculate_relu_flops(output_size, in_channels) # Use in_channels here
            total_flops += flops
        elif isinstance(module, nn.PixelShuffle):
            upscale_factor = module.upscale_factor
            output_size = (in_channels, feature_map_sizes[0][0] * upscale_factor, feature_map_sizes[0][1] * upscale_factor)
            flops = calculate_pixelshuffle_flops(in_channels, upscale_factor, output_size)
            total_flops += flops
            h_out = output_size[1]
            w_out = output_size[2]
            feature_map_sizes[0] = (h_out, w_out)
            in_channels = out_channels // (upscale_factor * upscale_factor) # Update in_channels
        elif isinstance(module, nn.Conv2d): # For the next iteration
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size[0]
            output_size = feature_map_sizes.pop(0)
            flops = calculate_conv2d_flops(in_channels, out_channels, kernel_size, output_size)
            params = calculate_conv2d_params(in_channels, out_channels, kernel_size)
            total_flops += flops
            total_params += params
            h_out = (output_size[0] + 2 * module.padding[0] - (module.dilation[0] * (module.kernel_size[0] - 1) + 1)) // module.stride[0] + 1
            w_out = (output_size[1] + 2 * module.padding[1] - (module.dilation[0] * (module.kernel_size[0] - 1) + 1)) // module.stride[0] + 1
            feature_map_sizes.append((h_out, w_out))

    return total_flops, total_params

if __name__ == '__main__':
    n_updownconv = 3
    in_chn = 3
    out_chn = 3
    input_size = (in_chn, 640, 360)  # Example input size

    model = autoencoder(n_updownconv=n_updownconv, in_chn=in_chn, out_chn=out_chn)
    flops, params = calculate_model_flops_and_params(model, input_size)

    print(f"Model: Autoencoder (n_updownconv={n_updownconv}, in_chn={in_chn}, out_chn={out_chn})")
    print(f"Input Shape: {input_size}")
    print(f"Total FLOPs: {flops:.2e}")
    print(f"Total Parameters: {params:.2e}")
