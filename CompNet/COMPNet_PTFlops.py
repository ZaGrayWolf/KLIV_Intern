import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info

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

if __name__ == '__main__':
    n_updownconv = 1 # Using n_updownconv = 1 as specified in the provided code
    in_chn = 3
    out_chn = 3

    # List of input sizes (H, W) to test
    input_hw_sizes = [
        (1280, 720),
        (760, 1360),   # H, W
        (900, 1600),   # H, W
        (1080, 1920),  # H, W
        (1152, 2048),  # H, W
        (1440, 2560), 
        (3840,2160),# H, W
    ]

    for h, w in input_hw_sizes:
        input_size = (in_chn, h, w)  # Full input size (C, H, W)
        model = autoencoder(n_updownconv=n_updownconv, in_chn=in_chn, out_chn=out_chn)
        
        print(f"\n--- Metrics for Input Shape: {input_size} ---")
        
        # Using ptflops for FLOPs and Parameters
        try:
            # Setting as_strings=False to get numerical values for easier conversion
            # Setting print_per_layer_stat=False to avoid verbose output for multiple input sizes
            macs, params = get_model_complexity_info(
                model, input_size, as_strings=False, print_per_layer_stat=False
            )
            # Convert MACs to FLOPs by multiplying by 2
            flops = macs * 2
            print(f"Total FLOPs: {flops / 1e9:.2f} B")
            print(f"Total Parameters: {params / 1e6:.2f} M")
        except Exception as e:
            print(f"Error calculating FLOPs/Params with ptflops: {e}")