import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Make sure ptflops is installed: pip install ptflops
from ptflops import get_model_complexity_info

# Make sure timm is installed if you use its layers: pip install timm
from timm.layers import to_2tuple, DropPath, trunc_normal_ # Using timm layers for DropPath

# --- Simulated Model Components (Based on typical UNext structure) ---

class DWConv(nn.Module):
    """ Depth-wise convolution """
    def __init__(self, dim=768, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # DWConv is nn.Conv2d with groups = in_channels
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=dim)

    def forward(self, x, H, W):
        # x is (B, N, C), need to reshape to (B, C, H, W)
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        # Reshape back to (B, N, C)
        x = x.flatten(2).transpose(1, 2)
        return x

class shiftmlp(nn.Module):
    """ Shifted MLP (used in shiftedBlock) """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # The DWConv acts on the spatial representation *after* fc1
        # Kernel size 3x3, stride 1, padding 1 are typical for DWConv in these blocks
        self.dwconv = DWConv(hidden_features, kernel_size=3, stride=1, padding=1)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, H, W):
        # x is (B, N, C)
        x = self.fc1(x)
        x = self.dwconv(x, H, W) # Pass H, W to DWConv
        x = self.act(x)
        x = self.fc2(x)
        return x

class shiftedBlock(nn.Module):
    """ A block combining LayerNorm, MLP, and residual connection """
    def __init__(self, dim, mlp_ratio=4., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        # DropPath is typically applied to the residual branch
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W):
        # x is (B, N, C)
        # Pre-norm structure: x + DropPath(MLP(Norm(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W)) # Pass H, W to MLP/DWConv
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding with overlapping patches """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        # Padding is calculated here based on patch_size/3 as in typical impl
        padding = (patch_size[0] // 3, patch_size[1] // 3)

        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride # Store stride for later use if needed

        # Use a convolution to perform the overlapping patch embedding
        # Note: ptflops usually infers shape from this conv.
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim) # Layer norm after flattening

    def forward(self, x):
        # Input x is (B, C_in, H_in, W_in)
        x = self.proj(x) # Output is (B, embed_dim, H_out, W_out)
        H_out, W_out = x.shape[-2:] # Get actual output spatial dims
        # Flatten spatial dimensions and transpose channels to get (B, N, C)
        x = x.flatten(2).transpose(1, 2) # Output is (B, N, embed_dim), where N = H_out * W_out
        x = self.norm(x)
        return x, H_out, W_out # Return x (B, N, C) and spatial dims (H, W)

# --- Simulated UNext Model Definition ---

class UNext(nn.Module):
    """
    Simulated UNext model structure for FLOPs calculation.
    Based on common UNext variants, but may differ slightly from a specific paper implementation.
    """
    def __init__(self, num_classes=4, input_channels=3, depths=[1, 1], mlp_ratios=[1., 1.], drop_path_rate=0.1, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        # Define channel dimensions for each stage based on common UNext variants
        # These match the dimensions used in the manual calculation script
        self.dims = [16, 32, 128, 160, 256] # Encoder stages 1, 2, 3, Transformer stage 4, Bottleneck stage 5
        self.decoder_dims = [160, 128, 32, 16, 16] # Decoder stages 1, 2, 3, 4, 5
        self.depths = depths # Number of shifted blocks in stage 4 and 5 / DBlock 1 and 2
        self.mlp_ratios = mlp_ratios # MLP ratio for stage 4/DBlock 1 and stage 5/DBlock 2

        # Encoder
        self.encoder1 = nn.Conv2d(input_channels, self.dims[0], 3, padding=1)
        self.ebn1 = nn.BatchNorm2d(self.dims[0])
        self.encoder2 = nn.Conv2d(self.dims[0], self.dims[1], 3, padding=1)
        self.ebn2 = nn.BatchNorm2d(self.dims[1])
        self.encoder3 = nn.Conv2d(self.dims[1], self.dims[2], 3, padding=1)
        self.ebn3 = nn.BatchNorm2d(self.dims[2])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)

        # Transformer Encoder Stages (Stage 4 and 5)
        # Patch Embed 3: Input from encoder3 output (dims[2]), embed_dim dims[3]
        # Using patch_size=3, stride=2 is common to reduce spatial size by 2x
        self.patch_embed3 = OverlapPatchEmbed(in_chans=self.dims[2], embed_dim=self.dims[3], patch_size=3, stride=2)
        # Shifted Blocks for Stage 4
        # Drop path rate schedule - linear decay
        # Total blocks = (depths[0] for block1) + (depths[1] for block2) + (depths[0] for dblock1) + (depths[1] for dblock2)
        total_dpr_blocks = 2 * sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_dpr_blocks)]
        
        cur = 0
        self.block1 = nn.ModuleList([shiftedBlock(dim=self.dims[3], mlp_ratio=mlp_ratios[0], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[0])])
        self.norm3 = norm_layer(self.dims[3]) # Norm after Block 1

        # Patch Embed 4: Input from Stage 4 output (dims[3]), embed_dim dims[4]
        self.patch_embed4 = OverlapPatchEmbed(in_chans=self.dims[3], embed_dim=self.dims[4], patch_size=3, stride=2)
         # Shifted Blocks for Stage 5 (Bottleneck)
        cur += depths[0]
        self.block2 = nn.ModuleList([shiftedBlock(dim=self.dims[4], mlp_ratio=mlp_ratios[1], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[1])])
        self.norm4 = norm_layer(self.dims[4]) # Norm after Block 2

        # Decoder (Upsampling + Concat + Conv + BN + ReLU) + DBlocks
        # Decoder 1: Input from Stage 5 output (dims[4]), Skip from Stage 4 (dims[3])
        self.decoder1 = nn.Conv2d(self.dims[4] + self.dims[3], self.decoder_dims[0], 3, padding=1)
        self.dbn1 = nn.BatchNorm2d(self.decoder_dims[0])
        # DBlock 1 - These blocks operate on the combined feature map after upsampling and concatenation
        # Need to continue the drop path index
        cur += depths[1] # Index after encoder blocks
        self.dblock1 = nn.ModuleList([shiftedBlock(dim=self.decoder_dims[0], mlp_ratio=mlp_ratios[0], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[0])]) # Assuming DBlock1 uses same depths/mlp_ratio as Block1
        self.dnorm3 = norm_layer(self.decoder_dims[0]) # Norm after DBlock 1

        # Decoder 2: Input from Decoder 1 output (decoder_dims[0]), Skip from Stage 3 (dims[2])
        self.decoder2 = nn.Conv2d(self.decoder_dims[0] + self.dims[2], self.decoder_dims[1], 3, padding=1)
        self.dbn2 = nn.BatchNorm2d(self.decoder_dims[1])
         # DBlock 2
        cur += depths[0]
        self.dblock2 = nn.ModuleList([shiftedBlock(dim=self.decoder_dims[1], mlp_ratio=mlp_ratios[1], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[1])]) # Assuming DBlock2 uses same depths/mlp_ratio as Block2
        self.dnorm4 = norm_layer(self.decoder_dims[1]) # Norm after DBlock 2


        # Decoder 3: Input from Decoder 2 output (decoder_dims[1]), Skip from Stage 2 (dims[1])
        self.decoder3 = nn.Conv2d(self.decoder_dims[1] + self.dims[1], self.decoder_dims[2], 3, padding=1)
        self.dbn3 = nn.BatchNorm2d(self.decoder_dims[2])

        # Decoder 4: Input from Decoder 3 output (decoder_dims[2]), Skip from Stage 1 (dims[0])
        self.decoder4 = nn.Conv2d(self.decoder_dims[2] + self.dims[0], self.decoder_dims[3], 3, padding=1)
        self.dbn4 = nn.BatchNorm2d(self.decoder_dims[3])

        # Decoder 5: Input from Decoder 4 output (decoder_dims[3]), No skip (upsample to original resolution)
        self.decoder5 = nn.Conv2d(self.decoder_dims[3], self.decoder_dims[4], 3, padding=1)
        # Based on UNext-S having this, including it for consistency in simulation
        self.dbn5 = nn.BatchNorm2d(self.decoder_dims[4])
        # Note: A ReLU after decoder5 might also be present depending on the exact variant

        # Final convolution
        self.final = nn.Conv2d(self.decoder_dims[4], num_classes, 1)


    def forward(self, x):
         # Encoder
         e1 = self.relu(self.ebn1(self.encoder1(x)))
         e1_p = self.maxpool(e1)
         e2 = self.relu(self.ebn2(self.encoder2(e1_p)))
         e2_p = self.maxpool(e2)
         e3 = self.relu(self.ebn3(self.encoder3(e2_p)))
         e3_p = self.maxpool(e3)

         # Transformer stages
         t4_in_spatial = e3_p # Input to patch embed 3 is spatial
         t4, h4, w4 = self.patch_embed3(t4_in_spatial) # Output is sequence (B, N, C)
         for blk in self.block1:
             t4 = blk(t4, h4, w4) # Blocks operate on sequence (B, N, C), need spatial H,W for DWConv
         t4 = self.norm3(t4) # Norm on sequence
         # Reshape back to spatial after transformer blocks for decoder skip connection and next stage
         t4_out_spatial = t4.transpose(1, 2).reshape(t4.shape[0], t4.shape[-1], h4, w4)

         t5_in_spatial = t4_out_spatial # Input to patch embed 4 is spatial from stage 4
         t5, h5, w5 = self.patch_embed4(t5_in_spatial) # Output is sequence (B, N, C)
         for blk in self.block2:
             t5 = blk(t5, h5, w5) # Blocks operate on sequence (B, N, C)
         t5 = self.norm4(t5) # Norm on sequence
         # Reshape back to spatial for the start of the decoder
         t5_out_spatial = t5.transpose(1, 2).reshape(t5.shape[0], t5.shape[-1], h5, w5)


         # Decoder
         # Decoder 1: Upsample t5_out_spatial to match t4_out_spatial size
         d1_up = F.interpolate(t5_out_spatial, size=t4_out_spatial.shape[-2:], mode='bilinear', align_corners=False)
         # Concatenate with skip connection from t4 (output of Stage 4 transformer)
         d1_cat = torch.cat([d1_up, t4_out_spatial], dim=1)
         # Conv + BN + ReLU
         d1 = self.relu(self.dbn1(self.decoder1(d1_cat)))
         # DBlock 1 - operate on spatial feature map, need H, W
         d1_h, d1_w = d1.shape[-2:]
         d1_flat = d1.flatten(2).transpose(1, 2) # Reshape spatial to sequence
         for blk in self.dblock1:
             d1_flat = blk(d1_flat, d1_h, d1_w) # DBlocks operate on sequence, pass H, W for DWConv
         d1 = self.dnorm3(d1_flat).transpose(1, 2).reshape(d1.shape[0], d1.shape[1], d1_h, d1_w) # Reshape back to spatial


         # Decoder 2: Upsample d1 to match e3 size (output of Encoder Stage 3 before MaxPool)
         d2_up = F.interpolate(d1, size=e3.shape[-2:], mode='bilinear', align_corners=False)
         # Concatenate with skip connection from e3
         d2_cat = torch.cat([d2_up, e3], dim=1)
         # Conv + BN + ReLU
         d2 = self.relu(self.dbn2(self.decoder2(d2_cat)))
          # DBlock 2 - operate on spatial feature map, need H, W
         d2_h, d2_w = d2.shape[-2:]
         d2_flat = d2.flatten(2).transpose(1, 2) # Reshape spatial to sequence
         for blk in self.dblock2:
             d2_flat = blk(d2_flat, d2_h, d2_w) # DBlocks operate on sequence, pass H, W
         d2 = self.dnorm4(d2_flat).transpose(1, 2).reshape(d2.shape[0], d2.shape[1], d2_h, d2_w) # Reshape back to spatial


         # Decoder 3: Upsample d2 to match e2 size
         d3_up = F.interpolate(d2, size=e2.shape[-2:], mode='bilinear', align_corners=False)
         # Concatenate with skip connection from e2
         d3_cat = torch.cat([d3_up, e2], dim=1)
         # Conv + BN + ReLU
         d3 = self.relu(self.dbn3(self.decoder3(d3_cat)))

         # Decoder 4: Upsample d3 to match e1 size
         d4_up = F.interpolate(d3, size=e1.shape[-2:], mode='bilinear', align_corners=False)
         # Concatenate with skip connection from e1
         d4_cat = torch.cat([d4_up, e1], dim=1)
         # Conv + BN + ReLU
         d4 = self.relu(self.dbn4(self.decoder4(d4_cat)))

         # Decoder 5: Upsample d4 to match original input size
         d5_up = F.interpolate(d4, size=x.shape[-2:], mode='bilinear', align_corners=False)
         # Conv + BN + ReLU
         d5 = self.relu(self.dbn5(self.decoder5(d5_up))) # Assuming dbn5 and relu exist here

         # Final convolution
         final = self.final(d5)

         return final


# --- Example Usage with ptflops ---

if __name__ == "__main__":
    # Configuration (Match these with model defaults or your specific setup)
    NUM_CLASSES = 4 # Example: 4 classes
    BATCH_SIZE = 1 # ptflops typically calculates for batch_size=1
    IN_CHANNELS = 3

    # Define the list of input spatial resolutions (H, W)
    input_spatial_sizes = [
        (360, 640),    # H, W
        (720, 1280),   # H, W
        (760, 1360),   # H, W
        (900, 1600),   # H, W
        (1080, 1920),  # H, W
        (1152, 2048),  # H, W
        (1440, 2560), 
        (3840,2160),# H, W
    ]

    # Define depths and mlp_ratios for the UNext variant
    unext_depths = [1, 1] # Number of transformer blocks in Stage 4 and Stage 5/Decoder 1 and Decoder 2
    unext_mlp_ratios = [4., 4.] # MLP expansion ratio for Stage 4/Decoder 1 and Stage 5/Decoder 2

    print("--- Running UNext Model Complexity Benchmarks (ptflops) ---")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"Input Channels: {IN_CHANNELS}")
    print(f"UNext Depths: {unext_depths}")
    print(f"UNext MLP Ratios: {unext_mlp_ratios}")
    print("-" * 60)

    for H, W in input_spatial_sizes:
        # Construct the input_size tuple for ptflops (C, H, W)
        input_size = (IN_CHANNELS, H, W)

        try:
            # Instantiate the model for each input size (ptflops creates a dummy input based on this)
            unext_model = UNext(num_classes=NUM_CLASSES,
                                input_channels=input_size[0],
                                depths=unext_depths,
                                mlp_ratios=unext_mlp_ratios)
            unext_model.eval() # Set to eval mode

            print(f"\nCalculating complexity for Input Resolution: {H}x{W} (Input Shape: (1, {', '.join(map(str, input_size))}))")

            # Use ptflops to get complexity info
            with torch.no_grad(): # Ensure no gradients are computed
                flops, params = get_model_complexity_info(unext_model, input_size,
                                                          as_strings=True,
                                                          print_per_layer_stat=False, # Set to False for cleaner output
                                                          verbose=False)

            print(f"  Parameters: {params}")
            print(f"  FLOPs: {flops}")

        except Exception as e:
             print(f"An error occurred during UNext complexity calculation for {H}x{W}: {e}")
             import traceback
             traceback.print_exc()
        print("-" * 60)

    print("\n--- Benchmarks Complete ---")