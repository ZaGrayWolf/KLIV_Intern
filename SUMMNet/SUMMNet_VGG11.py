import torch
import torch.nn as nn
import torch.nn.functional as F
from   torchvision import models
import math
from ptflops import get_model_complexity_info

# --- Your SUMNet Model Definition ---

class SUMNet(nn.Module):
    def __init__(self):
        super(SUMNet, self).__init__()

        # Load VGG11 features (encoder)
        # Note: VGG11 features layers are:
        # 0: Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # 1: ReLU
        # 2: MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # 3: Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # 4: ReLU
        # 5: MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # 6: Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # 7: ReLU
        # 8: Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # 9: ReLU
        # 10: MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # 11: Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # 12: ReLU
        # 13: Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # 14: ReLU
        # 15: MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # 16: Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # 17: ReLU
        # 18: Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # 19: ReLU
        # 20: MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.encoder   = models.vgg11(weights = models.VGG11_Weights.IMAGENET1K_V1).features # Use weights instead of pretrained
        self.preconv   = nn.Conv2d(3, 3, 1) # 1x1 convolution before the main encoder

        # Assigning VGG layers to specific names for clarity
        self.conv1     = self.encoder[0]
        self.pool1     = nn.MaxPool2d(2, 2, return_indices = True) # Store indices for unpooling
        self.conv2     = self.encoder[3]
        self.pool2     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv3a    = self.encoder[6]
        self.conv3b    = self.encoder[8]
        self.pool3     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv4a    = self.encoder[11]
        self.conv4b    = self.encoder[13]
        self.pool4     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv5a    = self.encoder[16]
        self.conv5b    = self.encoder[18]
        self.pool5     = nn.MaxPool2d(2, 2, return_indices = True)

        # Decoder (Upsampling and Convolutional layers)
        self.unpool5   = nn.MaxUnpool2d(2, 2)
        # Concatenation happens in forward pass, input channels reflect this
        self.donv5b    = nn.Conv2d(self.encoder[18].out_channels * 2, 512, 3, padding = 1) # conv5b out_channels * 2 (unpool + skip)
        self.donv5a    = nn.Conv2d(512, 512, 3, padding = 1)
        self.unpool4   = nn.MaxUnpool2d(2, 2)
        self.donv4b    = nn.Conv2d(self.encoder[13].out_channels + 512, 512, 3, padding = 1) # conv4b out_channels + donv5a out_channels
        self.donv4a    = nn.Conv2d(512, 256, 3, padding = 1)
        self.unpool3   = nn.MaxUnpool2d(2, 2)
        self.donv3b    = nn.Conv2d(self.encoder[8].out_channels + 256, 256, 3, padding = 1) # conv3b out_channels + donv4a out_channels
        self.donv3a    = nn.Conv2d(256,128, 3, padding = 1)
        self.unpool2   = nn.MaxUnpool2d(2, 2)
        self.donv2     = nn.Conv2d(self.encoder[3].out_channels + 128, 64, 3, padding = 1) # conv2 out_channels + donv3a out_channels
        self.unpool1   = nn.MaxUnpool2d(2, 2)
        self.donv1     = nn.Conv2d(self.encoder[0].out_channels + 64, 32, 3, padding = 1) # conv1 out_channels + donv2 out_channels
        self.output    = nn.Conv2d(32, 5, 1) # Final 1x1 convolution for class scores

    def forward(self, x):
        # Encoder Pass (VGG11 features)
        preconv        = F.relu(self.preconv(x), inplace = True)
        conv1          = F.relu(self.conv1(preconv), inplace = True) # Output channels: 64
        pool1, idxs1   = self.pool1(conv1) # Spatial size / 2
        conv2          = F.relu(self.conv2(pool1), inplace = True) # Output channels: 128
        pool2, idxs2   = self.pool2(conv2) # Spatial size / 2
        conv3a         = F.relu(self.conv3a(pool2), inplace = True) # Output channels: 256
        conv3b         = F.relu(self.conv3b(conv3a), inplace = True) # Output channels: 256
        pool3, idxs3   = self.pool3(conv3b) # Spatial size / 2
        conv4a         = F.relu(self.conv4a(pool3), inplace = True) # Output channels: 512
        conv4b         = F.relu(self.conv4b(conv4a), inplace = True) # Output channels: 512
        pool4, idxs4   = self.pool4(conv4b) # Spatial size / 2
        conv5a         = F.relu(self.conv5a(pool4), inplace = True) # Output channels: 512
        conv5b         = F.relu(self.conv5b(conv5a), inplace = True) # Output channels: 512
        pool5, idxs5   = self.pool5(conv5b) # Spatial size / 2 (e.g., 256 -> 8x8)

        # Decoder Pass (Upsampling and Convolutional layers)
        # Unpool 5: Upsample pool5 using indices from pool5, then concatenate with conv5b
        # pool5 shape: (B, 512, H/32, W/32)
        # conv5b shape: (B, 512, H/16, W/16) -- Note: VGG pooling reduces by 2x each time.
        # Input (256x256) -> Pool1 (128) -> Pool2 (64) -> Pool3 (32) -> Pool4 (16) -> Pool5 (8)
        # So conv5b is H/16, W/16. pool5 is H/32, W/32.
        # MaxUnpool needs the size of the *output* it should produce, which is the size of the skip connection.
        unpool5_out_size = conv5b.size()[-2:]
        unpool5        = torch.cat([self.unpool5(pool5, idxs5, output_size=unpool5_out_size), conv5b], 1) # Concat channels: 512 + 512 = 1024
        donv5b         = F.relu(self.donv5b(unpool5), inplace = True) # Input channels: 1024, Output channels: 512
        donv5a         = F.relu(self.donv5a(donv5b), inplace = True) # Input channels: 512, Output channels: 512

        # Unpool 4: Upsample donv5a to match conv4b size, then concatenate
        # donv5a shape: (B, 512, H/16, W/16)
        # conv4b shape: (B, 512, H/8, W/8)
        unpool4_out_size = conv4b.size()[-2:]
        unpool4        = torch.cat([self.unpool4(donv5a, idxs4, output_size=unpool4_out_size), conv4b], 1) # Concat channels: 512 + 512 = 1024
        donv4b         = F.relu(self.donv4b(unpool4), inplace = True) # Input channels: 1024, Output channels: 512
        donv4a         = F.relu(self.donv4a(donv4b), inplace = True) # Input channels: 512, Output channels: 256

        # Unpool 3: Upsample donv4a to match conv3b size, then concatenate
        # donv4a shape: (B, 256, H/8, W/8)
        # conv3b shape: (B, 256, H/4, W/4)
        unpool3_out_size = conv3b.size()[-2:]
        unpool3        = torch.cat([self.unpool3(donv4a, idxs3, output_size=unpool3_out_size), conv3b], 1) # Concat channels: 256 + 256 = 512
        donv3b         = F.relu(self.donv3b(unpool3), inplace = True) # Input channels: 512, Output channels: 256
        donv3a         = F.relu(self.donv3a(donv3b)) # Input channels: 256, Output channels: 128

        # Unpool 2: Upsample donv3a to match conv2 size, then concatenate
        # donv3a shape: (B, 128, H/4, W/4)
        # conv2 shape: (B, 128, H/2, W/2)
        unpool2_out_size = conv2.size()[-2:]
        unpool2        = torch.cat([self.unpool2(donv3a, idxs2, output_size=unpool2_out_size), conv2], 1) # Concat channels: 128 + 128 = 256
        donv2          = F.relu(self.donv2(unpool2), inplace = True) # Input channels: 256, Output channels: 64

        # Unpool 1: Upsample donv2 to match conv1 size, then concatenate
        # donv2 shape: (B, 64, H/2, W/2)
        # conv1 shape: (B, 64, H, W)
        unpool1_out_size = conv1.size()[-2:]
        unpool1        = torch.cat([self.unpool1(donv2, idxs1, output_size=unpool1_out_size), conv1], 1) # Concat channels: 64 + 64 = 128
        donv1          = F.relu(self.donv1(unpool1), inplace = True) # Input channels: 128, Output channels: 32

        # Final Output Layer
        output         = self.output(donv1) # Input channels: 32, Output channels: 5
        return torch.sigmoid(output)

# --- Helper Functions for Manual FLOPs Calculation ---

# Using helpers from previous task, adapted for clarity and specific needs
def get_conv_output_shape(input_shape, kernel_size, stride, padding, dilation=1):
    # input_shape is (B, C_in, H_in, W_in)
    B, C_in, H_in, W_in = input_shape
    K_H, K_W = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    S_H, S_W = (stride, stride) if isinstance(stride, int) else stride
    P_H, P_W = (padding, padding) if isinstance(padding, int) else padding
    D_H, D_W = (dilation, dilation) if isinstance(dilation, int) else dilation

    H_out = math.floor(((H_in + 2 * P_H - D_H * (K_H - 1) - 1) / S_H) + 1)
    W_out = math.floor(((W_in + 2 * P_W - D_W * (K_W - 1) - 1) / S_W) + 1)
    return H_out, W_out

def get_pool_output_shape(input_shape, kernel_size, stride, padding=0):
     # input_shape is (B, C_in, H_in, W_in)
    B, C_in, H_in, W_in = input_shape
    K_H, K_W = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    S_H, S_W = (stride, stride) if isinstance(stride, int) else stride
    P_H, P_W = (padding, padding) if isinstance(padding, int) else padding

    H_out = math.floor(((H_in + 2 * P_H - K_H) / S_H) + 1)
    W_out = math.floor(((W_in + 2 * P_W - K_W) / S_W) + 1)
    return H_out, W_out


def flops_conv2d(module, input_shape):
    # input_shape is (B, C_in, H_in, W_in)
    B, C_in, H_in, W_in = input_shape
    C_out = module.out_channels
    groups = module.groups

    H_out, W_out = get_conv_output_shape(input_shape,
                                         module.kernel_size,
                                         module.stride,
                                         module.padding,
                                         module.dilation)

    KH, KW = module.kernel_size
    # MACs: Output_elements * C_out * (C_in / groups) * KH * KW
    macs = B * H_out * W_out * C_out * (C_in // groups) * KH * KW
    flops = 2 * macs # Multiply by 2 for FLOPs (mult + add)

    if module.bias is not None:
        flops += B * C_out * H_out * W_out # Additions for bias

    output_shape = (B, C_out, H_out, W_out)
    return flops, output_shape

def flops_relu(input_shape):
    # 1 FLOP per element for ReLU (comparison + assignment)
    num_elements = 1
    for dim in input_shape:
        num_elements *= dim
    flops = num_elements
    return flops, input_shape # Shape doesn't change

def flops_sigmoid(input_shape):
     # Approximation: Exp + Div + Add ~ 3 FLOPs per element
    num_elements = 1
    for dim in input_shape:
        num_elements *= dim
    flops = 3 * num_elements
    return flops, input_shape # Shape doesn't change


def flops_pool2d(input_shape, kernel_size, stride, padding=0):
    # Pooling is typically considered 0 FLOPs (memory operations)
    H_out, W_out = get_pool_output_shape(input_shape, kernel_size, stride, padding)
    output_shape = (input_shape[0], input_shape[1], H_out, W_out)
    return 0, output_shape

# Note: MaxUnpool2d needs the output_size to determine the target shape
def flops_unpool2d(input_shape, output_size):
    # MaxUnpooling is typically considered 0 FLOPs (memory operations using indices)
    B, C, H_in, W_in = input_shape
    H_out, W_out = output_size
    output_shape = (B, C, H_out, W_out)
    return 0, output_shape

def flops_concat(shape1, shape2):
    # Concatenation is typically considered 0 FLOPs (memory operation)
    # Assuming concatenation along dim 1 (channels)
    B, C1, H, W = shape1
    _, C2, _, _ = shape2 # Assume same H, W for concatenation
    output_shape = (B, C1 + C2, H, W)
    return 0, output_shape


def calculate_manual_flops(model, input_shape):
    total_flops = 0
    current_shape = input_shape
    # Dictionary to store shapes of encoder outputs before pooling for skip connections
    encoder_shapes = {}

    B, C_in, H_in, W_in = input_shape

    # --- Encoder Pass ---

    # preconv
    f, current_shape = flops_conv2d(model.preconv, current_shape)
    total_flops += f
    f, current_shape = flops_relu(current_shape)
    total_flops += f
    # print(f"After preconv+relu: {current_shape}")

    # conv1 (VGG features[0])
    f, current_shape = flops_conv2d(model.conv1, current_shape)
    total_flops += f
    f, current_shape = flops_relu(current_shape)
    total_flops += f
    encoder_shapes['conv1'] = current_shape # Store shape before pooling
    f, current_shape = flops_pool2d(current_shape, model.pool1.kernel_size, model.pool1.stride)
    total_flops += f
    # print(f"After conv1+relu+pool1: {current_shape}")

    # conv2 (VGG features[3])
    f, current_shape = flops_conv2d(model.conv2, current_shape)
    total_flops += f
    f, current_shape = flops_relu(current_shape)
    total_flops += f
    encoder_shapes['conv2'] = current_shape # Store shape before pooling
    f, current_shape = flops_pool2d(current_shape, model.pool2.kernel_size, model.pool2.stride)
    total_flops += f
    # print(f"After conv2+relu+pool2: {current_shape}")

    # conv3a (VGG features[6])
    f, current_shape = flops_conv2d(model.conv3a, current_shape)
    total_flops += f
    f, current_shape = flops_relu(current_shape)
    total_flops += f
    # print(f"After conv3a+relu: {current_shape}")

    # conv3b (VGG features[8])
    f, current_shape = flops_conv2d(model.conv3b, current_shape)
    total_flops += f
    f, current_shape = flops_relu(current_shape)
    total_flops += f
    encoder_shapes['conv3b'] = current_shape # Store shape before pooling
    f, current_shape = flops_pool2d(current_shape, model.pool3.kernel_size, model.pool3.stride)
    total_flops += f
    # print(f"After conv3b+relu+pool3: {current_shape}")

    # conv4a (VGG features[11])
    f, current_shape = flops_conv2d(model.conv4a, current_shape)
    total_flops += f
    f, current_shape = flops_relu(current_shape)
    total_flops += f
    # print(f"After conv4a+relu: {current_shape}")

    # conv4b (VGG features[13])
    f, current_shape = flops_conv2d(model.conv4b, current_shape)
    total_flops += f
    f, current_shape = flops_relu(current_shape)
    total_flops += f
    encoder_shapes['conv4b'] = current_shape # Store shape before pooling
    f, current_shape = flops_pool2d(current_shape, model.pool4.kernel_size, model.pool4.stride)
    total_flops += f
    # print(f"After conv4b+relu+pool4: {current_shape}")

    # conv5a (VGG features[16])
    f, current_shape = flops_conv2d(model.conv5a, current_shape)
    total_flops += f
    f, current_shape = flops_relu(current_shape)
    total_flops += f
    # print(f"After conv5a+relu: {current_shape}")

    # conv5b (VGG features[18])
    f, current_shape = flops_conv2d(model.conv5b, current_shape)
    total_flops += f
    f, current_shape = flops_relu(current_shape)
    total_flops += f
    encoder_shapes['conv5b'] = current_shape # Store shape before pooling
    f, current_shape = flops_pool2d(current_shape, model.pool5.kernel_size, model.pool5.stride)
    total_flops += f
    # print(f"After conv5b+relu+pool5: {current_shape}")


    # --- Decoder Pass ---
    # Note: current_shape is now the output of pool5

    # unpool5 + concat with conv5b
    # Input to unpool5 is current_shape (output of pool5)
    # Skip connection shape is encoder_shapes['conv5b'] (shape before pool5)
    # Corrected: Access shape dimensions using tuple indexing [-2:]
    unpool5_out_size = encoder_shapes['conv5b'][-2:]
    f_unpool, shape_unpool5 = flops_unpool2d(current_shape, unpool5_out_size)
    total_flops += f_unpool
    f_cat, current_shape = flops_concat(shape_unpool5, encoder_shapes['conv5b'])
    total_flops += f_cat # current_shape is now the concatenated shape
    # print(f"After unpool5+concat: {current_shape}")

    # donv5b
    f, current_shape = flops_conv2d(model.donv5b, current_shape)
    total_flops += f
    f, current_shape = flops_relu(current_shape)
    total_flops += f
    # print(f"After donv5b+relu: {current_shape}")

    # donv5a
    f, current_shape = flops_conv2d(model.donv5a, current_shape)
    total_flops += f
    f, current_shape = flops_relu(current_shape)
    total_flops += f
    # print(f"After donv5a+relu: {current_shape}")

    # unpool4 + concat with conv4b
    # Input to unpool4 is current_shape (output of donv5a)
    # Skip connection shape is encoder_shapes['conv4b'] (shape before pool4)
    # Corrected: Access shape dimensions using tuple indexing [-2:]
    unpool4_out_size = encoder_shapes['conv4b'][-2:]
    f_unpool, shape_unpool4 = flops_unpool2d(current_shape, unpool4_out_size)
    total_flops += f_unpool
    f_cat, current_shape = flops_concat(shape_unpool4, encoder_shapes['conv4b'])
    total_flops += f_cat
    # print(f"After unpool4+concat: {current_shape}")

    # donv4b
    f, current_shape = flops_conv2d(model.donv4b, current_shape)
    total_flops += f
    f, current_shape = flops_relu(current_shape)
    total_flops += f
    # print(f"After donv4b+relu: {current_shape}")

    # donv4a
    f, current_shape = flops_conv2d(model.donv4a, current_shape)
    total_flops += f
    f, current_shape = flops_relu(current_shape) # Note: Original code uses F.relu(..., inplace=True) here but F.relu(donv3b) later
    total_flops += f
    # print(f"After donv4a+relu: {current_shape}")

    # unpool3 + concat with conv3b
    # Input to unpool3 is current_shape (output of donv4a)
    # Skip connection shape is encoder_shapes['conv3b'] (shape before pool3)
    # Corrected: Access shape dimensions using tuple indexing [-2:]
    unpool3_out_size = encoder_shapes['conv3b'][-2:]
    f_unpool, shape_unpool3 = flops_unpool2d(current_shape, unpool3_out_size)
    total_flops += f_unpool
    f_cat, current_shape = flops_concat(shape_unpool3, encoder_shapes['conv3b'])
    total_flops += f_cat
    # print(f"After unpool3+concat: {current_shape}")

    # donv3b
    f, current_shape = flops_conv2d(model.donv3b, current_shape)
    total_flops += f
    f, current_shape = flops_relu(current_shape)
    total_flops += f
    # print(f"After donv3b+relu: {current_shape}")

    # donv3a
    f, current_shape = flops_conv2d(model.donv3a, current_shape)
    total_flops += f
    f, current_shape = flops_relu(current_shape) # Note: Original code uses F.relu(donv3b) without inplace
    total_flops += f
    # print(f"After donv3a+relu: {current_shape}")

    # unpool2 + concat with conv2
    # Input to unpool2 is current_shape (output of donv3a)
    # Skip connection shape is encoder_shapes['conv2'] (shape before pool2)
    # Corrected: Access shape dimensions using tuple indexing [-2:]
    unpool2_out_size = encoder_shapes['conv2'][-2:]
    f_unpool, shape_unpool2 = flops_unpool2d(current_shape, unpool2_out_size)
    total_flops += f_unpool
    f_cat, current_shape = flops_concat(shape_unpool2, encoder_shapes['conv2'])
    total_flops += f_cat
    # print(f"After unpool2+concat: {current_shape}")

    # donv2
    f, current_shape = flops_conv2d(model.donv2, current_shape)
    total_flops += f
    f, current_shape = flops_relu(current_shape)
    total_flops += f
    # print(f"After donv2+relu: {current_shape}")

    # unpool1 + concat with conv1
    # Input to unpool1 is current_shape (output of donv2)
    # Skip connection shape is encoder_shapes['conv1'] (shape before pool1)
    # Corrected: Access shape dimensions using tuple indexing [-2:]
    unpool1_out_size = encoder_shapes['conv1'][-2:]
    f_unpool, shape_unpool1 = flops_unpool2d(current_shape, unpool1_out_size)
    total_flops += f_unpool
    f_cat, current_shape = flops_concat(shape_unpool1, encoder_shapes['conv1'])
    total_flops += f_cat
    # print(f"After unpool1+concat: {current_shape}")

    # donv1
    f, current_shape = flops_conv2d(model.donv1, current_shape)
    total_flops += f
    f, current_shape = flops_relu(current_shape)
    total_flops += f
    # print(f"After donv1+relu: {current_shape}")

    # output
    f, current_shape = flops_conv2d(model.output, current_shape)
    total_flops += f
    # print(f"After output conv: {current_shape}")

    # sigmoid (final activation)
    f, current_shape = flops_sigmoid(current_shape)
    total_flops += f
    # print(f"After sigmoid: {current_shape}")


    return total_flops

# --- Parameter Counting Function ---
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Main Execution ---

if __name__ == "__main__":
    # Input configuration
    BATCH_SIZE = 1
    INPUT_CHANNELS = 3
    INPUT_H = 640
    INPUT_W = 360
    input_shape = (BATCH_SIZE, INPUT_CHANNELS, INPUT_H, INPUT_W)
    input_size_ptflops = (INPUT_CHANNELS, INPUT_H, INPUT_W) # ptflops expects (C, H, W)

    # Instantiate the model
    model = SUMNet()
    model.eval() # Set to evaluation mode

    print(f"Analyzing SUMNet model with input shape: {input_shape}")
    print("-" * 40)

    # --- Manual Calculation ---
    print("--- Manual Calculation ---")
    # Need a dummy input tensor to get the actual shapes from the VGG layers
    # This is a bit of a cheat for manual calculation, as it uses the model's forward to get intermediate shapes
    # A true manual trace would calculate these shapes based on input size, kernel, stride, padding
    # However, using a dummy forward ensures the shapes match the actual model execution.
    dummy_input = torch.randn(input_shape)
    # Pass the dummy input through the model to populate necessary intermediate shapes
    # We don't need the output, just the shapes captured in calculate_manual_flops
    with torch.no_grad():
         # Call the forward pass to get shapes for manual calculation
         # The calculate_manual_flops function will now run the trace using these shapes
         # No need to actually call the forward here, the calculate_manual_flops does the tracing.
         # The shapes needed for the manual calculation are obtained *within* that function's trace
         pass # We removed the need to pass dummy input explicitly

    manual_params = count_parameters(model)
    manual_flops = calculate_manual_flops(model, input_shape)


    print(f"Manual Parameters: {manual_params:,}")
    print(f"Manual FLOPs: {manual_flops:,}")
    print(f"Manual Parameters: {manual_params / 1e6:.2f} M")
    print(f"Manual FLOPs: {manual_flops / 1e9:.2f} GFLOPs")
    print("-" * 40)

    # --- PTFlops Calculation ---
    print("--- PTFlops Calculation ---")
    try:
        with torch.no_grad():
            flops_pt, params_pt = get_model_complexity_info(model, input_size_ptflops,
                                                            as_strings=False, # Get numerical values
                                                            print_per_layer_stat=False,
                                                            verbose=False)

        # ptflops reports MACs for Conv, so multiply by 2 for FLOPs
        # It also includes additions from bias in Conv, so this is generally consistent
        # The definition of FLOPs can vary (e.g., 1 MAC = 1 FLOP or 2 FLOPs)
        # We'll report both the raw ptflops output (often MACs) and the *2 FLOPs* version
        print(f"PTFlops (MACs): {flops_pt:,}")
        print(f"PTFlops (Approx. 2*MACs): {flops_pt * 2:,}") # Assuming 1 MAC = 2 FLOPs
        print(f"PTFlops Parameters: {params_pt:,}")

        print(f"PTFlops (MACs): {flops_pt / 1e9:.2f} G")
        print(f"PTFlops (Approx. 2*MACs): {(flops_pt * 2) / 1e9:.2f} GFLOPs")
        print(f"PTFlops Parameters: {params_pt / 1e6:.2f} M")

    except Exception as e:
        print(f"An error occurred during PTFlops calculation: {e}")
        import traceback
        traceback.print_exc()
    print("-" * 40)

    print("Note: Minor differences between manual and ptflops results may occur")
    print("due to varying definitions of FLOPs for certain operations (e.g., ReLU, Pooling)")
    print("and how libraries handle specific layer implementations.")
    print("PTFlops typically reports MACs for convolutions, which are often multiplied by 2 for FLOPs.")