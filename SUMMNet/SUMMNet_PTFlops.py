import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info

# --- SUMNet_all_bn PyTorch Class Definition ---
# This is the network definition provided by the user.
# ptflops will instantiate and analyze this class.

class SUMNet_all_bn(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SUMNet_all_bn, self).__init__()

        # Encoder
        self.conv1     = nn.Conv2d(in_ch, 64, 3, padding=1)
        self.bn1       = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2     = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2       = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.pool1     = nn.MaxPool2d(2, 2, return_indices = True) # spatial / 2
        self.conv3a    = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3a      = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3b    = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3b      = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.pool2     = nn.MaxPool2d(2, 2, return_indices = True) # spatial / 4
        self.conv4a    = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4a      = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4b    = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4b      = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.pool3     = nn.MaxPool2d(2, 2, return_indices = True) # spatial / 8
        self.conv5a    = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5a      = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5b    = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5b      = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.pool4     = nn.MaxPool2d(2, 2, return_indices = True) # spatial / 16

        # Decoder (using donv for deconvolution/decoder convolution)
        # Note: MaxUnpool2d does not change the number of channels
        self.unpool4   = nn.MaxUnpool2d(2, 2) # spatial * 16 -> *8
        self.donv5b    = nn.Conv2d(1024, 512, 3, padding = 1) # Cat: 512 (unpool) + 512 (conv5b skip)
        self.dbn5b     = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.donv5a    = nn.Conv2d(512, 512, 3, padding = 1)
        self.dbn5a     = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.unpool3   = nn.MaxUnpool2d(2, 2) # spatial * 8 -> *4
        self.donv4b    = nn.Conv2d(1024, 512, 3, padding = 1) # Cat: 512 (unpool) + 512 (conv4b skip)
        self.dbn4b     = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.donv4a    = nn.Conv2d(512, 256, 3, padding = 1)
        self.dbn4a     = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.unpool2   = nn.MaxUnpool2d(2, 2) # spatial * 4 -> *2
        self.donv3b    = nn.Conv2d(512, 256, 3, padding = 1) # Cat: 256 (unpool) + 256 (conv3b skip)
        self.dbn3b     = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.donv3a    = nn.Conv2d(256, 128, 3, padding = 1)
        self.dbn3a     = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.unpool1   = nn.MaxUnpool2d(2, 2) # spatial * 2 -> *1 (full resolution)
        self.donv2     = nn.Conv2d(128, 64, 3, padding = 1) # Input is just unpooled 128ch
        self.dbn2      = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        # The input to donv1 is the concatenation of donv2 output (64ch) and conv1 output (64ch)
        self.donv1     = nn.Conv2d(128, 32, 3, padding = 1) # Cat: 64 (donv2) + 64 (conv1 skip)
        self.dbn1      = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.output    = nn.Conv2d(32, out_ch, 1) # 1x1 convolution

    # The forward pass is essential for ptflops to trace the operations
    def forward(self, x):
        # Encoder
        conv1          = F.relu(self.bn1(self.conv1(x)), inplace = True)
        conv2          = F.relu(self.bn2(self.conv2(conv1)), inplace = True)
        pool1, idxs1   = self.pool1(conv2) # need idxs for unpooling

        conv3a         = F.relu(self.bn3a(self.conv3a(pool1)), inplace = True)
        conv3b         = F.relu(self.bn3b(self.conv3b(conv3a)), inplace = True)
        pool2, idxs2   = self.pool2(conv3b) # need idxs

        conv4a         = F.relu(self.bn4a(self.conv4a(pool2)), inplace = True)
        conv4b         = F.relu(self.bn4b(self.conv4b(conv4a)), inplace = True)
        pool3, idxs3   = self.pool3(conv4b) # need idxs

        conv5a         = F.relu(self.bn5a(self.conv5a(pool3)), inplace = True)
        conv5b         = F.relu(self.bn5b(self.conv5b(conv5a)), inplace = True)
        pool4, idxs4   = self.pool4(conv5b) # need idxs

        # Decoder
        # Unpool4 needs the output size of conv5b
        unpool4        = self.unpool4(pool4, idxs4, output_size=conv5b.size())
        donv5b_in      = torch.cat([unpool4, conv5b], 1) # Concat with conv5b skip
        donv5b         = F.relu(self.dbn5b(self.donv5b(donv5b_in)), inplace = True)
        donv5a         = F.relu(self.dbn5a(self.donv5a(donv5b)), inplace = True)

        # Unpool3 needs the output size of conv4b
        unpool3        = self.unpool3(donv5a, idxs3, output_size=conv4b.size())
        donv4b_in      = torch.cat([unpool3, conv4b], 1) # Concat with conv4b skip
        donv4b         = F.relu(self.dbn4b(self.donv4b(donv4b_in)), inplace = True)
        donv4a         = F.relu(self.dbn4a(self.donv4a(donv4b)), inplace = True)

        # Unpool2 needs the output size of conv3b
        unpool2        = self.unpool2(donv4a, idxs2, output_size=conv3b.size())
        donv3b_in      = torch.cat([unpool2, conv3b], 1) # Concat with conv3b skip
        donv3b         = F.relu(self.dbn3b(self.donv3b(donv3b_in)), inplace = True)
        donv3a         = F.relu(self.dbn3a(self.donv3a(donv3b)), inplace = True)

        # Unpool1 needs the output size of conv2
        unpool1        = self.unpool1(donv3a, idxs1, output_size=conv2.size())
        # Input to donv2 is just the unpooled output
        donv2          = F.relu(self.dbn2(self.donv2(unpool1)), inplace = True)
        # Input to donv1 is concat of donv2 output and conv1 skip
        donv1_in       = torch.cat([donv2, conv1], 1)
        donv1          = F.relu(self.dbn1(self.donv1(donv1_in)), inplace = True)

        # Output layer
        output         = self.output(donv1)

        return output


# --- Benchmarking Setup using ptflops ---

def run_ptflops_benchmarks():
    input_spatial_sizes = [
        (360, 640),    # H, W
        (720, 1280),   # H, W
        (760, 1360),   # H, W
        (900, 1600),   # H, W
        (1080, 1920),  # H, W
        (1152, 2048),  # H, W
        (1440, 2560),  # H, W
    ]
    in_channels = 3  # Standard input channels for images
    out_channels = 1 # Example output channels

    print("--- Running SUMNet_all_bn Complexity Benchmarks (using ptflops) ---")
    print(f"Fixed Input Channels: {in_channels}")
    print(f"Fixed Output Channels: {out_channels}")
    print("-" * 40)

    for h, w in input_spatial_sizes:
        # ptflops expects input_size as (C, H, W)
        input_size = (in_channels, h, w)

        # Instantiate the model for the current benchmark run
        model = SUMNet_all_bn(in_ch=in_channels, out_ch=out_channels)
        model.eval() # Set model to evaluation mode (important for BN, Dropout, etc.)

        print(f"\n--- Calculating for Input Resolution: {h}x{w} (Input shape: {input_size}) ---")

        try:
            # Use get_model_complexity_info from ptflops
            # as_strings=True for formatted output like GFLOPs, MParams
            # print_per_layer_stat=True to see breakdown per layer
            flops, params = get_model_complexity_info(
                model,
                input_size,
                as_strings=True,
                print_per_layer_stat=True, # Set to False if you only want the total
                # Add the following lines to potentially get more accurate counts
                # for ops like BN and ReLU if they are counted differently by default
                # ost = {
                #     'multiply_adds': 1, # Count MACs
                #     'norm': 0,          # Don't count BN
                #     'relu': 0,          # Don't count ReLU
                #     'conv_bn': 2,       # Alternative: Count conv+BN together
                #     'conv_relu': 2,     # Alternative: Count conv+ReLU together
                #     'linear': 2,        # Alternative: Count linear ops
                #     'add': 1,
                # }
            )

            print(f"Total FLOPs: {flops}")
            print(f"Total Parameters: {params}")

        except Exception as e:
            print(f"Error calculating complexity for {input_size}: {e}")
            print("This might happen if ptflops cannot trace the model correctly.")

        print("-" * 40) # Separator


    print("--- Benchmarks Complete ---")


# --- Main Execution ---
if __name__ == "__main__":
    run_ptflops_benchmarks()