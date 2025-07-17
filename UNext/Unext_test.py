import torch
from torch import nn
import torch.nn.functional as F
import math

# Make sure timm is installed if you use its layers outside the provided snippets
# pip install timm
# Updated import path
from timm.layers import to_2tuple, DropPath, trunc_normal_


# Helper to calculate Conv2d output shape
def get_conv_output_shape(input_shape, kernel_size, stride, padding, dilation=1):
    # input_shape is (B, C_in, H_in, W_in)
    B, C_in, H_in, W_in = input_shape

    # Use to_2tuple to handle int or tuple inputs for K, S, P, D
    K_H, K_W = to_2tuple(kernel_size)
    S_H, S_W = to_2tuple(stride)
    P_H, P_W = to_2tuple(padding)
    D_H, D_W = to_2tuple(dilation)

    # Ensure integer inputs for calculation
    H_in, W_in = int(H_in), int(W_in)
    P_H, P_W = int(P_H), int(P_W)
    D_H, D_W = int(D_H), int(D_W)
    K_H, K_W = int(K_H), int(K_W)
    S_H, S_W = int(S_H), int(S_W)

    H_out = math.floor(((H_in + 2 * P_H - D_H * (K_H - 1) - 1) / S_H) + 1)
    W_out = math.floor(((W_in + 2 * P_W - D_W * (K_W - 1) - 1) / S_W) + 1)
    return H_out, W_out

# Helper to calculate Pooling output shape (MaxPooling and AvgPooling use similar formulas)
def get_pool_output_shape(input_shape, kernel_size, stride, padding=0, dilation=1):
    # input_shape is (B, C_in, H_in, W_in)
    B, C_in, H_in, W_in = input_shape
    # Use to_2tuple to handle int or tuple inputs
    K_H, K_W = to_2tuple(kernel_size)
    S_H, S_W = to_2tuple(stride)
    P_H, P_W = to_2tuple(padding)
    # Dilation for pooling usually means dilated pooling (atrous) but for shape calculation,
    # if it's standard max/avg pool, it doesn't affect effective kernel size for output shape.
    # For simplicity, we treat D=1 for output shape calc as in Conv, even if pooling layer has a dilation param
    # that affects the *input* region.
    D_H, D_W = to_2tuple(dilation)

    # Ensure integer inputs for calculation
    H_in, W_in = int(H_in), int(W_in)
    P_H, P_W = int(P_H), int(P_W)
    D_H, D_W = int(D_H), int(D_W) # For pooling, D usually implies kernel stride effectively
    K_H, K_W = int(K_H), int(K_W)
    S_H, S_W = int(S_H), int(S_W)


    H_out = math.floor(((H_in + 2 * P_H - K_H) / S_H) + 1)
    W_out = math.floor(((W_in + 2 * P_W - K_W) / S_W) + 1)
    return H_out, W_out


# --- FLOP Calculation Functions ---

def flops_conv2d(module, input_shape):
    # input_shape is (B, C_in, H_in, W_in)
    B, C_in, H_in, W_in = input_shape
    C_out = module.out_channels
    groups = module.groups

    # Get output spatial dimensions
    # Pass padding from the module, not a fixed value
    H_out, W_out = get_conv_output_shape(input_shape,
                                         module.kernel_size,
                                         module.stride,
                                         module.padding,
                                         module.dilation) # Include dilation

    KH, KW = to_2tuple(module.kernel_size)

    # MACs: Output_elements * C_out * (C_in / groups) * KH * KW
    # Output elements: B * H_out * W_out
    macs = B * H_out * W_out * C_out * (C_in // groups) * KH * KW
    flops = 2 * macs # Multiply by 2 for FLOPs (mult + add)

    if module.bias is not None:
        # Additions for bias: B * C_out * H_out * W_out
        flops += B * C_out * H_out * W_out

    output_shape = (B, C_out, H_out, W_out)
    return flops, output_shape

def flops_dwconv2d(module, input_shape):
    # DWConv is Conv2d with groups = in_channels = out_channels
    # input_shape is (B, C_in, H_in, W_in)
    # This function is specifically for an actual nn.Conv2d instance that is depthwise
    return flops_conv2d(module, input_shape) # flops_conv2d handles groups correctly

def flops_linear(module, input_shape):
    # Assumes input is (B, N, C_in) or (B, C_in)
    # input_shape can be multi-dimensional before the last feature dimension
    # e.g., (B, H*W, C_in) for transformers
    input_elements = 1
    for dim in input_shape[:-1]:
        input_elements *= dim # Includes Batch size and sequence length (N)
    C_in = input_shape[-1]
    C_out = module.out_features

    # MACs: Input_elements * C_in * C_out
    macs = input_elements * C_in * C_out
    flops = 2 * macs # Multiply by 2 for FLOPs (mult + add)

    if module.bias is not None:
        # Additions for bias: Input_elements * C_out
        flops += input_elements * C_out

    output_shape = list(input_shape[:-1]) + [C_out]
    return flops, tuple(output_shape)

def flops_norm(input_shape):
    # Approx 4 FLOPs per element for BN/LN (subtract mean, divide std, multiply gamma, add beta)
    # Input shape can be (B, C, H, W) for BN or (B, N, C) for LN
    num_elements = 1
    for dim in input_shape:
        num_elements *= dim
    flops = 4 * num_elements
    return flops, input_shape # Shape doesn't change

def flops_activation(input_shape):
    # Approx 1 FLOP per element for ReLU/GELU/etc.
    num_elements = 1
    for dim in input_shape:
        num_elements *= dim
    flops = num_elements
    return flops, input_shape # Shape doesn't change

def flops_add(input_shape):
    # 1 FLOP per element for addition (e.g., skip connections, residuals)
    num_elements = 1
    for dim in input_shape:
        num_elements *= dim
    flops = num_elements
    return flops, input_shape # Shape doesn't change

def flops_upsample_bilinear(input_shape, output_size):
    # Approximating bilinear interpolation FLOPs
    # Each output pixel requires ~4 multiplications and ~3 additions per channel
    # Let's simplify: roughly 7 FLOPs per element per channel
    B, C_in, H_in, W_in = input_shape
    H_out, W_out = to_2tuple(output_size)

    num_elements_out = B * C_in * H_out * W_out
    flops = 7 * num_elements_out
    output_shape = (B, C_in, H_out, W_out)
    return flops, output_shape


# --- Simulated Model Components (Replacing Placeholders) ---

class DWConv(nn.Module):
    """ Depth-wise convolution """
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

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
        self.dwconv = DWConv(hidden_features) # DWConv kernel size is usually fixed
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
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W):
        # x is (B, N, C)
        # Pre-norm
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
        # This is why the __init__ does not accept a 'padding' argument
        padding = (patch_size[0] // 3, patch_size[1] // 3)

        self.img_size = img_size
        self.patch_size = patch_size
        # Note: The actual output spatial size depends on input size, stride, padding
        # This calculation in init is an estimate, actual is determined by conv output
        # self.H, self.W = img_size[0] // stride[0], img_size[1] // stride[1]
        # self.num_patches = self.H * self.W

        # Use a convolution to perform the overlapping patch embedding
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

# --- Simulated UNext Model Definitions ---

class UNext(nn.Module):
    def __init__(self, num_classes=4, input_channels=3, depths=[1, 1], mlp_ratios=[1., 1.], drop_path_rate=0.1, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        # Define channel dimensions for each stage based on common UNext variants
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
        # Corrected: Removed padding=1 as OverlapPatchEmbed calculates it internally
        self.patch_embed3 = OverlapPatchEmbed(in_chans=self.dims[2], embed_dim=self.dims[3], patch_size=3, stride=2)
        # Shifted Blocks for Stage 4
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths) + sum(depths))] # Re-linspace over all blocks
        cur = 0
        self.block1 = nn.ModuleList([shiftedBlock(dim=self.dims[3], mlp_ratio=mlp_ratios[0], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[0])])
        self.norm3 = norm_layer(self.dims[3]) # Norm after Block 1

        # Patch Embed 4: Input from Stage 4 output (dims[3]), embed_dim dims[4]
        # Corrected: Removed padding=1
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
        # DBlock 1
        # Note: dpr indices should continue from encoder blocks
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
        self.dbn5 = nn.BatchNorm2d(self.decoder_dims[4]) # UNext_S had this, let's include for UNext sim
        # Note: Original UNext might have skipped dbn5 and relu after decoder5

        # Final convolution
        self.final = nn.Conv2d(self.decoder_dims[4], num_classes, 1)

    # Add forward pass to allow model introspection for shapes (though not used by manual flops counter)
    def forward(self, x):
         # This forward is minimal, just to allow layer access and basic shape tracking
         # The manual FLOPs function doesn't execute this, but relies on its structure
         # print("Warning: Running simplified forward for UNext simulation.")
         # Encoder
         e1 = self.relu(self.ebn1(self.encoder1(x)))
         e1_p = self.maxpool(e1)
         e2 = self.relu(self.ebn2(self.encoder2(e1_p)))
         e2_p = self.maxpool(e2)
         e3 = self.relu(self.ebn3(self.encoder3(e2_p)))
         e3_p = self.maxpool(e3)

         # Transformer stages
         t4, h4, w4 = self.patch_embed3(e3_p)
         for blk in self.block1:
             t4 = blk(t4, h4, w4)
         t4 = self.norm3(t4)
         t4_spatial = t4.transpose(1, 2).reshape(t4.shape[0], t4.shape[-1], h4, w4) # Reshape back

         t5, h5, w5 = self.patch_embed4(t4_spatial)
         for blk in self.block2:
             t5 = blk(t5, h5, w5)
         t5 = self.norm4(t5)
         t5_spatial = t5.transpose(1, 2).reshape(t5.shape[0], t5.shape[-1], h5, w5) # Reshape back

         # Decoder
         d1_up = F.interpolate(t5_spatial, size=t4_spatial.shape[-2:], mode='bilinear', align_corners=False)
         d1_cat = torch.cat([d1_up, t4_spatial], dim=1)
         d1 = self.relu(self.dbn1(self.decoder1(d1_cat)))
         # DBlock 1
         d1_flat = d1.flatten(2).transpose(1, 2)
         for blk in self.dblock1:
             d1_flat = blk(d1_flat, d1.shape[-2], d1.shape[-1])
         d1 = self.dnorm3(d1_flat).transpose(1, 2).reshape(d1.shape[0], d1.shape[1], d1.shape[2], d1.shape[3])

         d2_up = F.interpolate(d1, size=e3.shape[-2:], mode='bilinear', align_corners=False)
         d2_cat = torch.cat([d2_up, e3], dim=1)
         d2 = self.relu(self.dbn2(self.decoder2(d2_cat)))
          # DBlock 2
         d2_flat = d2.flatten(2).transpose(1, 2)
         for blk in self.dblock2:
             d2_flat = blk(d2_flat, d2.shape[-2], d2.shape[-1])
         d2 = self.dnorm4(d2_flat).transpose(1, 2).reshape(d2.shape[0], d2.shape[1], d2.shape[2], d2.shape[3])

         d3_up = F.interpolate(d2, size=e2.shape[-2:], mode='bilinear', align_corners=False)
         d3_cat = torch.cat([d3_up, e2], dim=1)
         d3 = self.relu(self.dbn3(self.decoder3(d3_cat)))

         d4_up = F.interpolate(d3, size=e1.shape[-2:], mode='bilinear', align_corners=False)
         d4_cat = torch.cat([d4_up, e1], dim=1)
         d4 = self.relu(self.dbn4(self.decoder4(d4_cat)))

         d5_up = F.interpolate(d4, size=x.shape[-2:], mode='bilinear', align_corners=False)
         d5 = self.relu(self.dbn5(self.decoder5(d5_up)))

         final = self.final(d5)

         return final


class UNext_S(nn.Module):
     def __init__(self, num_classes=4, input_channels=3, depths=[1, 1], mlp_ratios=[1., 1.], drop_path_rate=0.1, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        # Define channel dimensions for each stage for the 'S' variant
        self.dims = [8, 16, 32, 64, 128] # Encoder stages 1, 2, 3, Transformer stage 4, Bottleneck stage 5
        self.decoder_dims = [64, 32, 16, 8, 8] # Decoder stages 1, 2, 3, 4, 5
        self.depths = depths
        self.mlp_ratios = mlp_ratios

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
        # Corrected: Removed padding=1
        self.patch_embed3 = OverlapPatchEmbed(in_chans=self.dims[2], embed_dim=self.dims[3], patch_size=3, stride=2)
        # Shifted Blocks for Stage 4
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths) + sum(depths))] # Re-linspace over all blocks
        cur = 0
        self.block1 = nn.ModuleList([shiftedBlock(dim=self.dims[3], mlp_ratio=mlp_ratios[0], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[0])])
        self.norm3 = norm_layer(self.dims[3]) # Norm after Block 1

        # Patch Embed 4: Input from Stage 4 output (dims[3]), embed_dim dims[4]
        # Corrected: Removed padding=1
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
        # DBlock 1
        cur += depths[1]
        self.dblock1 = nn.ModuleList([shiftedBlock(dim=self.decoder_dims[0], mlp_ratio=mlp_ratios[0], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[0])])
        self.dnorm3 = norm_layer(self.decoder_dims[0]) # Norm after DBlock 1

        # Decoder 2: Input from Decoder 1 output (decoder_dims[0]), Skip from Stage 3 (dims[2])
        self.decoder2 = nn.Conv2d(self.decoder_dims[0] + self.dims[2], self.decoder_dims[1], 3, padding=1)
        self.dbn2 = nn.BatchNorm2d(self.decoder_dims[1])
         # DBlock 2
        cur += depths[0]
        self.dblock2 = nn.ModuleList([shiftedBlock(dim=self.decoder_dims[1], mlp_ratio=mlp_ratios[1], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[1])])
        self.dnorm4 = norm_layer(self.decoder_dims[1]) # Norm after DBlock 2


        # Decoder 3: Input from Decoder 2 output (decoder_dims[1]), Skip from Stage 2 (dims[1])
        self.decoder3 = nn.Conv2d(self.decoder_dims[1] + self.dims[1], self.decoder_dims[2], 3, padding=1)
        self.dbn3 = nn.BatchNorm2d(self.decoder_dims[2])

        # Decoder 4: Input from Decoder 3 output (decoder_dims[2]), Skip from Stage 1 (dims[0])
        self.decoder4 = nn.Conv2d(self.decoder_dims[2] + self.dims[0], self.decoder_dims[3], 3, padding=1)
        self.dbn4 = nn.BatchNorm2d(self.decoder_dims[3])

        # Decoder 5: Input from Decoder 4 output (decoder_dims[3]), No skip (upsample to original resolution)
        self.decoder5 = nn.Conv2d(self.decoder_dims[3], self.decoder_dims[4], 3, padding=1)
        self.dbn5 = nn.BatchNorm2d(self.decoder_dims[4])

        # Final convolution
        self.final = nn.Conv2d(self.decoder_dims[4], num_classes, 1)

    # Add forward pass
     def forward(self, x):
         # print("Warning: Running simplified forward for UNext_S simulation.")
         # Encoder
         e1 = self.relu(self.ebn1(self.encoder1(x)))
         e1_p = self.maxpool(e1)
         e2 = self.relu(self.ebn2(self.encoder2(e1_p)))
         e2_p = self.maxpool(e2)
         e3 = self.relu(self.ebn3(self.encoder3(e2_p)))
         e3_p = self.maxpool(e3)

         # Transformer stages
         t4, h4, w4 = self.patch_embed3(e3_p)
         for blk in self.block1:
             t4 = blk(t4, h4, w4)
         t4 = self.norm3(t4)
         t4_spatial = t4.transpose(1, 2).reshape(t4.shape[0], t4.shape[-1], h4, w4) # Reshape back

         t5, h5, w5 = self.patch_embed4(t4_spatial)
         for blk in self.block2:
             t5 = blk(t5, h5, w5)
         t5 = self.norm4(t5)
         t5_spatial = t5.transpose(1, 2).reshape(t5.shape[0], t5.shape[-1], h5, w5) # Reshape back


         # Decoder
         d1_up = F.interpolate(t5_spatial, size=t4_spatial.shape[-2:], mode='bilinear', align_corners=False)
         d1_cat = torch.cat([d1_up, t4_spatial], dim=1)
         d1 = self.relu(self.dbn1(self.decoder1(d1_cat)))
          # DBlock 1
         d1_flat = d1.flatten(2).transpose(1, 2)
         for blk in self.dblock1:
             d1_flat = blk(d1_flat, d1.shape[-2], d1.shape[-1])
         d1 = self.dnorm3(d1_flat).transpose(1, 2).reshape(d1.shape[0], d1.shape[1], d1.shape[2], d1.shape[3])


         d2_up = F.interpolate(d1, size=e3.shape[-2:], mode='bilinear', align_corners=False)
         d2_cat = torch.cat([d2_up, e3], dim=1)
         d2 = self.relu(self.dbn2(self.decoder2(d2_cat)))
          # DBlock 2
         d2_flat = d2.flatten(2).transpose(1, 2)
         for blk in self.dblock2:
             d2_flat = blk(d2_flat, d2.shape[-2], d2.shape[-1])
         d2 = self.dnorm4(d2_flat).transpose(1, 2).reshape(d2.shape[0], d2.shape[1], d2.shape[2], d2.shape[3])

         d3_up = F.interpolate(d2, size=e2.shape[-2:], mode='bilinear', align_corners=False)
         d3_cat = torch.cat([d3_up, e2], dim=1)
         d3 = self.relu(self.dbn3(self.decoder3(d3_cat)))

         d4_up = F.interpolate(d3, size=e1.shape[-2:], mode='bilinear', align_corners=False)
         d4_cat = torch.cat([d4_up, e1], dim=1)
         d4 = self.relu(self.dbn4(self.decoder4(d4_cat)))

         d5_up = F.interpolate(d4, size=x.shape[-2:], mode='bilinear', align_corners=False)
         d5 = self.relu(self.dbn5(self.decoder5(d5_up)))

         final = self.final(d5)

         return final


# --- Manual FLOP Calculation Function (Revised to return FLOPs and Activation MB) ---

def calculate_manual_flops(model, input_shape):
    total_flops = 0
    total_params = sum(p.numel() for p in model.parameters())
    # print(f"Model Parameters: {total_params / 1e6:.2f} M") # For debugging

    total_activations = 0 # Track total number of elements in activations
    shape_dict = {} # To store shapes for skip connections etc.

    B, C_in, H_in, W_in = input_shape
    current_shape = input_shape
    # print(f"Starting FLOPs and Activation Pass Size calculation for {type(model).__name__} with input shape {input_shape}")
    total_activations += B * C_in * H_in * W_in # Input tensor itself

    # --- Encoder ---
    # Stage 1 (Conv -> BN -> ReLU -> MaxPool)
    # print("Encoder Stage 1...")
    f, current_shape = flops_conv2d(model.encoder1, current_shape)
    total_flops += f
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3] # Output of conv

    f, current_shape = flops_norm(current_shape) # ebn1
    total_flops += f
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3] # Output of BN

    f, current_shape = flops_activation(current_shape) # relu
    total_flops += f
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3] # Output of ReLU

    # Store shape *before* maxpool for skip connection
    shape_dict['t1'] = current_shape
    H_pool, W_pool = get_pool_output_shape(current_shape, 2, 2) # max_pool2d
    current_shape = (B, current_shape[1], H_pool, W_pool)
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3] # Output of MaxPool
    # print(f"  End Stage 1. Current shape: {current_shape}")

    # Stage 2 (Conv -> BN -> ReLU -> MaxPool)
    # print("Encoder Stage 2...")
    f, current_shape = flops_conv2d(model.encoder2, current_shape)
    total_flops += f
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3]

    f, current_shape = flops_norm(current_shape) # ebn2
    total_flops += f
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3]

    f, current_shape = flops_activation(current_shape) # relu
    total_flops += f
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3]

    # Store shape *before* maxpool for skip connection
    shape_dict['t2'] = current_shape
    H_pool, W_pool = get_pool_output_shape(current_shape, 2, 2) # max_pool2d
    current_shape = (B, current_shape[1], H_pool, W_pool)
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3]
    # print(f"  End Stage 2. Current shape: {current_shape}")

    # Stage 3 (Conv -> BN -> ReLU -> MaxPool)
    # print("Encoder Stage 3...")
    f, current_shape = flops_conv2d(model.encoder3, current_shape)
    total_flops += f
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3]

    f, current_shape = flops_norm(current_shape) # ebn3
    total_flops += f
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3]

    f, current_shape = flops_activation(current_shape) # relu
    total_flops += f
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3]

    # Store shape *before* maxpool for skip connection
    shape_dict['t3'] = current_shape
    H_pool, W_pool = get_pool_output_shape(current_shape, 2, 2) # max_pool2d
    current_shape = (B, current_shape[1], H_pool, W_pool)
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3]
    # print(f"  End Stage 3. Current shape: {current_shape}")

    # --- Transformer/MLP Stages (Spatial to Sequence Transition) ---
    # Stage 4 (PatchEmbed3 + Block1)
    # print("Transformer Stage 4 (PatchEmbed3 + Block1)...")
    patch_embed_in_shape = current_shape
    # PatchEmbed 3 Projection (Conv)
    f_pe_proj, current_shape_conv = flops_conv2d(model.patch_embed3.proj, patch_embed_in_shape)
    total_flops += f_pe_proj
    total_activations += current_shape_conv[0] * current_shape_conv[1] * current_shape_conv[2] * current_shape_conv[3]

    B_pe3, C_pe3_out, H_pe3, W_pe3 = current_shape_conv # Output spatial shape of PatchEmbed Conv
    N_pe3 = H_pe3 * W_pe3
    # Reshape for LayerNorm and Blocks (B, N, C)
    current_shape_flat = (B_pe3, N_pe3, C_pe3_out)

    # PatchEmbed 3 LayerNorm
    f_pe_norm, current_shape_flat = flops_norm(current_shape_flat)
    total_flops += f_pe_norm
    total_activations += current_shape_flat[0] * current_shape_flat[1] * current_shape_flat[2]

    shape_dict['t4_spatial_for_skip'] = current_shape_conv # Store spatial shape *after* patch embed conv (for skip)

    # Block 1 (shiftedBlock)
    # print(f"  Block 1 ({len(model.block1)} blocks)...")
    block1_input_shape = current_shape_flat # (B, N, C)
    current_shape_block = block1_input_shape
    for i, block in enumerate(model.block1):
        # print(f"    Block 1.{i}...")
        shape_before_block = current_shape_block # (B, N, C)
        dim = shape_before_block[-1] # C
        mlp_ratio = model.mlp_ratios[0]
        hidden_dim = int(dim * mlp_ratio)

        # LayerNorm (pre-norm)
        f_norm, shape_after_norm = flops_norm(current_shape_block)
        total_flops += f_norm
        total_activations += shape_after_norm[0] * shape_after_norm[1] * shape_after_norm[2]

        # MLP
        # fc1: Input (B, N, C) -> Output (B, N, hidden_dim)
        f_fc1, shape_after_fc1 = flops_linear(block.mlp.fc1, shape_after_norm)
        total_flops += f_fc1
        total_activations += shape_after_fc1[0] * shape_after_fc1[1] * shape_after_fc1[2]

        # DWConv: Input (B, hidden_dim, H, W) -> Output (B, hidden_dim, H, W)
        shape_dwconv_in = (B_pe3, hidden_dim, H_pe3, W_pe3)
        f_dwc, shape_after_dwc = flops_dwconv2d(block.mlp.dwconv.dwconv, shape_dwconv_in)
        total_flops += f_dwc
        total_activations += shape_after_dwc[0] * shape_after_dwc[1] * shape_after_dwc[2] * shape_after_dwc[3]

        # Reshape back after DWConv to (B, N, hidden_dim)
        shape_after_dwc_flat = (B_pe3, N_pe3, hidden_dim)

        # act: Input (B, N, hidden_dim) -> Output (B, N, hidden_dim)
        f_act, shape_after_act = flops_activation(shape_after_dwc_flat)
        total_flops += f_act
        total_activations += shape_after_act[0] * shape_after_act[1] * shape_after_act[2]

        # fc2: Input (B, N, hidden_dim) -> Output (B, N, dim)
        f_fc2, shape_after_fc2 = flops_linear(block.mlp.fc2, shape_after_act)
        total_flops += f_fc2
        total_activations += shape_after_fc2[0] * shape_after_fc2[1] * shape_after_fc2[2]

        # Residual Add: Add shape_before_block (B, N, dim) and shape_after_fc2 (B, N, dim)
        f_add, current_shape_block = flops_add(shape_before_block) # Shape remains (B, N, dim)
        total_flops += f_add
        total_activations += current_shape_block[0] * current_shape_block[1] * current_shape_block[2]

    # Norm 3 after block1
    # print("  Norm 3 after Block 1...")
    f, current_shape_flat = flops_norm(current_shape_block) # model.norm3 operates on (B, N, C)
    total_flops += f
    total_activations += current_shape_flat[0] * current_shape_flat[1] * current_shape_flat[2]

    # Reshape back to spatial for potential next stages (Bottleneck)
    current_shape = (B_pe3, C_pe3_out, H_pe3, W_pe3) # (B, C_stage4, H_stage4, W_stage4)
    # print(f"  End Stage 4. Current spatial shape (post-transformer): {current_shape}")


    # Stage 5 (PatchEmbed4 + Block2 - Bottleneck)
    # print("Transformer Stage 5 (PatchEmbed4 + Block2 - Bottleneck)...")
    patch_embed_in_shape = current_shape
    # PatchEmbed 4 Projection (Conv)
    f_pe_proj, current_shape_conv = flops_conv2d(model.patch_embed4.proj, patch_embed_in_shape)
    total_flops += f_pe_proj
    total_activations += current_shape_conv[0] * current_shape_conv[1] * current_shape_conv[2] * current_shape_conv[3]

    B_pe4, C_pe4_out, H_pe4, W_pe4 = current_shape_conv # Output spatial shape of PatchEmbed Conv
    N_pe4 = H_pe4 * W_pe4
    # Reshape for LayerNorm and Blocks (B, N, C)
    current_shape_flat = (B_pe4, N_pe4, C_pe4_out)

    # PatchEmbed 4 LayerNorm
    f_pe_norm, current_shape_flat = flops_norm(current_shape_flat)
    total_flops += f_pe_norm
    total_activations += current_shape_flat[0] * current_shape_flat[1] * current_shape_flat[2]

    # Block 2 (shiftedBlock) - Bottleneck
    # print(f"  Block 2 ({len(model.block2)} blocks)...")
    block2_input_shape = current_shape_flat # (B, N, C)
    current_shape_block = block2_input_shape
    for i, block in enumerate(model.block2):
        # print(f"    Block 2.{i}...")
        shape_before_block = current_shape_block # (B, N, C)
        dim = shape_before_block[-1] # C
        mlp_ratio = model.mlp_ratios[1]
        hidden_dim = int(dim * mlp_ratio)

        # LayerNorm (pre-norm)
        f_norm, shape_after_norm = flops_norm(current_shape_block)
        total_flops += f_norm
        total_activations += shape_after_norm[0] * shape_after_norm[1] * shape_after_norm[2]

        f_fc1, shape_after_fc1 = flops_linear(block.mlp.fc1, shape_after_norm)
        total_flops += f_fc1
        total_activations += shape_after_fc1[0] * shape_after_fc1[1] * shape_after_fc1[2]

        shape_dwconv_in = (B_pe4, hidden_dim, H_pe4, W_pe4)
        f_dwc, shape_after_dwc = flops_dwconv2d(block.mlp.dwconv.dwconv, shape_dwconv_in)
        total_flops += f_dwc
        total_activations += shape_after_dwc[0] * shape_after_dwc[1] * shape_after_dwc[2] * shape_after_dwc[3]

        shape_after_dwc_flat = (B_pe4, N_pe4, hidden_dim)
        f_act, shape_after_act = flops_activation(shape_after_dwc_flat)
        total_flops += f_act
        total_activations += shape_after_act[0] * shape_after_act[1] * shape_after_act[2]

        f_fc2, shape_after_fc2 = flops_linear(block.mlp.fc2, shape_after_act)
        total_flops += f_fc2
        total_activations += shape_after_fc2[0] * shape_after_fc2[1] * shape_after_fc2[2]

        # Residual Add
        f_add, current_shape_block = flops_add(shape_before_block)
        total_flops += f_add
        total_activations += current_shape_block[0] * current_shape_block[1] * current_shape_block[2]

    # Norm 4 after block2
    # print("  Norm 4 after Block 2...")
    f, current_shape_flat = flops_norm(current_shape_block) # model.norm4 operates on (B, N, C)
    total_flops += f
    total_activations += current_shape_flat[0] * current_shape_flat[1] * current_shape_flat[2]

    # Reshape back to spatial for decoder
    current_shape = (B_pe4, C_pe4_out, H_pe4, W_pe4) # (B, C_stage5, H_stage5, W_stage5)
    shape_dict['t5_spatial_for_decoder'] = current_shape # Store this for the first decoder upsample

    # print(f"  End Stage 5 (Bottleneck). Current spatial shape (post-transformer): {current_shape}")

    # --- Decoder ---
    # Decoder 1 (Upsample -> Concat -> Conv -> BN -> ReLU -> DBlock1)
    # print("Decoder Stage 1...")
    # Upsample
    target_size_d1 = shape_dict['t4_spatial_for_skip'][-2:]
    f_up, upsampled_shape = flops_upsample_bilinear(current_shape, target_size_d1)
    total_flops += f_up
    total_activations += upsampled_shape[0] * upsampled_shape[1] * upsampled_shape[2] * upsampled_shape[3]

    # Concat
    skip_t4_shape = shape_dict['t4_spatial_for_skip']
    current_shape = (upsampled_shape[0], upsampled_shape[1] + skip_t4_shape[1], upsampled_shape[2], upsampled_shape[3])
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3] # Concatenated tensor

    # Conv
    f, current_shape = flops_conv2d(model.decoder1, current_shape)
    total_flops += f
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3]

    # BN
    f, current_shape = flops_norm(current_shape) # dbn1
    total_flops += f
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3]

    # ReLU
    f, current_shape = flops_activation(current_shape) # relu
    total_flops += f
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3]

    # DBlock 1
    B_d1, C_d1, H_d1, W_d1 = current_shape
    N_d1 = H_d1 * W_d1
    current_shape_flat = (B_d1, N_d1, C_d1)
    dblock1_input_shape = current_shape_flat
    for i, block in enumerate(model.dblock1):
        # print(f"    DBlock 1.{i}...")
        shape_before_dblock = current_shape_flat # (B, N, C)
        dim = shape_before_dblock[-1]
        mlp_ratio = model.mlp_ratios[0]
        hidden_dim = int(dim * mlp_ratio)

        f_norm, shape_after_norm = flops_norm(current_shape_flat)
        total_flops += f_norm
        total_activations += shape_after_norm[0] * shape_after_norm[1] * shape_after_norm[2]

        f_fc1, shape_after_fc1 = flops_linear(block.mlp.fc1, shape_after_norm)
        total_flops += f_fc1
        total_activations += shape_after_fc1[0] * shape_after_fc1[1] * shape_after_fc1[2]

        shape_dwconv_in = (B_d1, hidden_dim, H_d1, W_d1)
        f_dwc, shape_after_dwc = flops_dwconv2d(block.mlp.dwconv.dwconv, shape_dwconv_in)
        total_flops += f_dwc
        total_activations += shape_after_dwc[0] * shape_after_dwc[1] * shape_after_dwc[2] * shape_after_dwc[3]

        shape_after_dwc_flat = (B_d1, N_d1, hidden_dim)
        f_act, shape_after_act = flops_activation(shape_after_dwc_flat)
        total_flops += f_act
        total_activations += shape_after_act[0] * shape_after_act[1] * shape_after_act[2]

        f_fc2, shape_after_fc2 = flops_linear(block.mlp.fc2, shape_after_act)
        total_flops += f_fc2
        total_activations += shape_after_fc2[0] * shape_after_fc2[1] * shape_after_fc2[2]

        f_add, current_shape_flat = flops_add(shape_before_dblock)
        total_flops += f_add
        total_activations += current_shape_flat[0] * current_shape_flat[1] * current_shape_flat[2]

    f, current_shape_flat = flops_norm(current_shape_flat) # dnorm3
    total_flops += f
    total_activations += current_shape_flat[0] * current_shape_flat[1] * current_shape_flat[2]

    # Reshape back to spatial
    current_shape = (B_d1, C_d1, H_d1, W_d1)
    # print(f"  End Decoder Stage 1. Current spatial shape: {current_shape}")


    # Decoder 2 (Upsample -> Concat -> Conv -> BN -> ReLU -> DBlock2)
    # print("Decoder Stage 2...")
    # Upsample
    target_size_d2 = shape_dict['t3'][-2:] # Skip from encoder3 output (before maxpool)
    f_up, upsampled_shape = flops_upsample_bilinear(current_shape, target_size_d2)
    total_flops += f_up
    total_activations += upsampled_shape[0] * upsampled_shape[1] * upsampled_shape[2] * upsampled_shape[3]

    # Concat
    skip_t3_shape = shape_dict['t3']
    current_shape = (upsampled_shape[0], upsampled_shape[1] + skip_t3_shape[1], upsampled_shape[2], upsampled_shape[3])
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3]

    # Conv
    f, current_shape = flops_conv2d(model.decoder2, current_shape)
    total_flops += f
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3]

    # BN
    f, current_shape = flops_norm(current_shape) # dbn2
    total_flops += f
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3]

    # ReLU
    f, current_shape = flops_activation(current_shape) # relu
    total_flops += f
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3]

    # DBlock 2
    B_d2, C_d2, H_d2, W_d2 = current_shape
    N_d2 = H_d2 * W_d2
    current_shape_flat = (B_d2, N_d2, C_d2)
    dblock2_input_shape = current_shape_flat
    for i, block in enumerate(model.dblock2):
        # print(f"    DBlock 2.{i}...")
        shape_before_dblock = current_shape_flat # (B, N, C)
        dim = shape_before_dblock[-1]
        mlp_ratio = model.mlp_ratios[1]
        hidden_dim = int(dim * mlp_ratio)

        f_norm, shape_after_norm = flops_norm(current_shape_flat)
        total_flops += f_norm
        total_activations += shape_after_norm[0] * shape_after_norm[1] * shape_after_norm[2]

        f_fc1, shape_after_fc1 = flops_linear(block.mlp.fc1, shape_after_norm)
        total_flops += f_fc1
        total_activations += shape_after_fc1[0] * shape_after_fc1[1] * shape_after_fc1[2]

        shape_dwconv_in = (B_d2, hidden_dim, H_d2, W_d2)
        f_dwc, shape_after_dwc = flops_dwconv2d(block.mlp.dwconv.dwconv, shape_dwconv_in)
        total_flops += f_dwc
        total_activations += shape_after_dwc[0] * shape_after_dwc[1] * shape_after_dwc[2] * shape_after_dwc[3]

        shape_after_dwc_flat = (B_d2, N_d2, hidden_dim)
        f_act, shape_after_act = flops_activation(shape_after_dwc_flat)
        total_flops += f_act
        total_activations += shape_after_act[0] * shape_after_act[1] * shape_after_act[2]

        f_fc2, shape_after_fc2 = flops_linear(block.mlp.fc2, shape_after_act)
        total_flops += f_fc2
        total_activations += shape_after_fc2[0] * shape_after_fc2[1] * shape_after_fc2[2]

        f_add, current_shape_flat = flops_add(shape_before_dblock)
        total_flops += f_add
        total_activations += current_shape_flat[0] * current_shape_flat[1] * current_shape_flat[2]

    f, current_shape_flat = flops_norm(current_shape_flat) # dnorm4
    total_flops += f
    total_activations += current_shape_flat[0] * current_shape_flat[1] * current_shape_flat[2]

    # Reshape back to spatial
    current_shape = (B_d2, C_d2, H_d2, W_d2)
    # print(f"  End Decoder Stage 2. Current spatial shape: {current_shape}")


    # Decoder 3 (Upsample -> Concat -> Conv -> BN -> ReLU)
    # print("Decoder Stage 3...")
    # Upsample
    target_size_d3 = shape_dict['t2'][-2:] # Skip from encoder2 output (before maxpool)
    f_up, upsampled_shape = flops_upsample_bilinear(current_shape, target_size_d3)
    total_flops += f_up
    total_activations += upsampled_shape[0] * upsampled_shape[1] * upsampled_shape[2] * upsampled_shape[3]

    # Concat
    skip_t2_shape = shape_dict['t2']
    current_shape = (upsampled_shape[0], upsampled_shape[1] + skip_t2_shape[1], upsampled_shape[2], upsampled_shape[3])
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3]

    # Conv
    f, current_shape = flops_conv2d(model.decoder3, current_shape)
    total_flops += f
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3]

    # BN
    f, current_shape = flops_norm(current_shape) # dbn3
    total_flops += f
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3]

    # ReLU
    f, current_shape = flops_activation(current_shape) # relu
    total_flops += f
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3]
    # print(f"  End Decoder Stage 3. Current spatial shape: {current_shape}")


    # Decoder 4 (Upsample -> Concat -> Conv -> BN -> ReLU)
    # print("Decoder Stage 4...")
    # Upsample
    target_size_d4 = shape_dict['t1'][-2:] # Skip from encoder1 output (before maxpool)
    f_up, upsampled_shape = flops_upsample_bilinear(current_shape, target_size_d4)
    total_flops += f_up
    total_activations += upsampled_shape[0] * upsampled_shape[1] * upsampled_shape[2] * upsampled_shape[3]

    # Concat
    skip_t1_shape = shape_dict['t1']
    current_shape = (upsampled_shape[0], upsampled_shape[1] + skip_t1_shape[1], upsampled_shape[2], upsampled_shape[3])
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3]

    # Conv
    f, current_shape = flops_conv2d(model.decoder4, current_shape)
    total_flops += f
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3]

    # BN
    f, current_shape = flops_norm(current_shape) # dbn4
    total_flops += f
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3]

    # ReLU
    f, current_shape = flops_activation(current_shape) # relu
    total_flops += f
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3]
    # print(f"  End Decoder Stage 4. Current spatial shape: {current_shape}")

    # Decoder 5 (Upsample -> Conv -> BN -> ReLU)
    # print("Decoder Stage 5...")
    # Upsample
    target_size_d5 = input_shape[-2:] # Original input resolution
    f_up, upsampled_shape = flops_upsample_bilinear(current_shape, target_size_d5)
    total_flops += f_up
    total_activations += upsampled_shape[0] * upsampled_shape[1] * upsampled_shape[2] * upsampled_shape[3]

    # Conv
    f, current_shape = flops_conv2d(model.decoder5, upsampled_shape)
    total_flops += f
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3]

    # BN
    f, current_shape = flops_norm(current_shape) # dbn5
    total_flops += f
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3]

    # ReLU
    f, current_shape = flops_activation(current_shape) # relu
    total_flops += f
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3]
    # print(f"  End Decoder Stage 5. Current spatial shape: {current_shape}")

    # Final convolution
    # print("Final Convolution...")
    f, current_shape = flops_conv2d(model.final, current_shape)
    total_flops += f
    total_activations += current_shape[0] * current_shape[1] * current_shape[2] * current_shape[3] # Final output

    # Convert total_activations (number of elements) to MB
    # Assuming float32 (4 bytes per element)
    activation_mb = (total_activations * 4) / (1024 * 1024)

    return total_flops, activation_mb

# --- Main execution for activation pass size and FLOPs calculation ---

if __name__ == "__main__":
    input_sizes = [
        (360, 640),    # H, W
        (720, 1280),   # H, W
        (760, 1360),   # H, W
        (900, 1600),   # H, W
        (1080, 1920),  # H, W
        (1152, 2048),  # H, W
        (1440, 2560),
        (3840,2160),# H, W
    ]

    unext_depths = [1, 1]
    unext_mlp_ratios = [4., 4.]

    # Instantiate UNext model with custom depths and mlp_ratios
    model = UNext(num_classes=4, input_channels=3, depths=unext_depths, mlp_ratios=unext_mlp_ratios)
    batch_size = 1

    print("--- Calculating FLOPs and Total Activation Pass Size (MB) for UNext Model ---")
    print(f"Model: UNext (Batch Size: {batch_size}, Depths: {unext_depths}, MLP Ratios: {unext_mlp_ratios})")
    print("-" * 70)

    for H, W in input_sizes:
        input_shape = (batch_size, 3, H, W)  # (B, C, H, W)
        flops, activations_mb = calculate_manual_flops(model, input_shape)
        print(f"Input size ({H}, {W}):")
        print(f"  FLOPs: {flops / 1e9:.2f} GFLOPs")
        print(f"  Activation Pass Size: {activations_mb:.2f} MB")
        print("-" * 30)

    print("-" * 70)
    print("\nNote: 'GFLOPs' refers to Giga Floating Point Operations.")
    print("'MB' refers to Megabytes of memory for activation tensors (assuming float32).")
    print("This metric is useful for estimating memory footprint during inference.")