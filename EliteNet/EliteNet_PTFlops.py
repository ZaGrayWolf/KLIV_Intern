#!/usr/bin/env python3
"""
SegNet Profiling Script - Theoretical and PTFLOPS Analysis
- Provides reliable theoretical FLOPs calculations
- Attempts PTFLOPS measurement with proper error handling
- Outputs only Parameters and FLOPs as requested
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info

class SegNet(nn.Module):
    def __init__(self, input_channels=3, n_labels=21, kernel_size=3):
        super(SegNet, self).__init__()
        
        # Encoder blocks
        self.conv1_1 = nn.Conv2d(input_channels, 64, kernel_size, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        
        # Decoder blocks (symmetric to encoder)
        self.conv5_3_D = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn5_3_D = nn.BatchNorm2d(512)
        self.conv5_2_D = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn5_2_D = nn.BatchNorm2d(512)
        self.conv5_1_D = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn5_1_D = nn.BatchNorm2d(512)
        
        self.conv4_3_D = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn4_3_D = nn.BatchNorm2d(512)
        self.conv4_2_D = nn.Conv2d(512, 512, kernel_size, padding=1)
        self.bn4_2_D = nn.BatchNorm2d(512)
        self.conv4_1_D = nn.Conv2d(512, 256, kernel_size, padding=1)
        self.bn4_1_D = nn.BatchNorm2d(256)
        
        self.conv3_3_D = nn.Conv2d(256, 256, kernel_size, padding=1)
        self.bn3_3_D = nn.BatchNorm2d(256)
        self.conv3_2_D = nn.Conv2d(256, 256, kernel_size, padding=1)
        self.bn3_2_D = nn.BatchNorm2d(256)
        self.conv3_1_D = nn.Conv2d(256, 128, kernel_size, padding=1)
        self.bn3_1_D = nn.BatchNorm2d(128)
        
        self.conv2_2_D = nn.Conv2d(128, 128, kernel_size, padding=1)
        self.bn2_2_D = nn.BatchNorm2d(128)
        self.conv2_1_D = nn.Conv2d(128, 64, kernel_size, padding=1)
        self.bn2_1_D = nn.BatchNorm2d(64)
        
        self.conv1_2_D = nn.Conv2d(64, 64, kernel_size, padding=1)
        self.bn1_2_D = nn.BatchNorm2d(64)
        
        # Final classification layer
        self.classifier = nn.Conv2d(64, n_labels, 1)
        
        # Pooling/Unpooling layers
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
    
    def forward(self, x):
        # Encoder Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x1_size = x.size()
        x, indices1 = self.pool(x)
        
        # Encoder Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x2_size = x.size()
        x, indices2 = self.pool(x)
        
        # Encoder Block 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))
        x3_size = x.size()
        x, indices3 = self.pool(x)
        
        # Encoder Block 4
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = F.relu(self.bn4_3(self.conv4_3(x)))
        x4_size = x.size()
        x, indices4 = self.pool(x)
        
        # Encoder Block 5
        x = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(x)))
        x = F.relu(self.bn5_3(self.conv5_3(x)))
        x5_size = x.size()
        x, indices5 = self.pool(x)
        
        # Decoder Block 5
        x = self.unpool(x, indices5, output_size=x5_size)
        x = F.relu(self.bn5_3_D(self.conv5_3_D(x)))
        x = F.relu(self.bn5_2_D(self.conv5_2_D(x)))
        x = F.relu(self.bn5_1_D(self.conv5_1_D(x)))
        
        # Decoder Block 4
        x = self.unpool(x, indices4, output_size=x4_size)
        x = F.relu(self.bn4_3_D(self.conv4_3_D(x)))
        x = F.relu(self.bn4_2_D(self.conv4_2_D(x)))
        x = F.relu(self.bn4_1_D(self.conv4_1_D(x)))
        
        # Decoder Block 3
        x = self.unpool(x, indices3, output_size=x3_size)
        x = F.relu(self.bn3_3_D(self.conv3_3_D(x)))
        x = F.relu(self.bn3_2_D(self.conv3_2_D(x)))
        x = F.relu(self.bn3_1_D(self.conv3_1_D(x)))
        
        # Decoder Block 2
        x = self.unpool(x, indices2, output_size=x2_size)
        x = F.relu(self.bn2_2_D(self.conv2_2_D(x)))
        x = F.relu(self.bn2_1_D(self.conv2_1_D(x)))
        
        # Decoder Block 1
        x = self.unpool(x, indices1, output_size=x1_size)
        x = F.relu(self.bn1_2_D(self.conv1_2_D(x)))
        
        # Final classification
        x = self.classifier(x)
        
        return x

def calculate_conv_flops(input_shape, output_channels, kernel_size, stride=1, padding=0):
    """Calculate FLOPs for a convolutional layer"""
    in_channels, in_height, in_width = input_shape
    out_height = (in_height + 2 * padding - kernel_size) // stride + 1
    out_width = (in_width + 2 * padding - kernel_size) // stride + 1
    
    # FLOPs = (Kernel_size^2 * Input_channels) * Output_height * Output_width * Output_channels * 2
    flops = (kernel_size * kernel_size * in_channels) * out_height * out_width * output_channels * 2
    
    return flops, (output_channels, out_height, out_width)

def calculate_segnet_theoretical_flops(input_shape, n_labels=21):
    """Calculate theoretical FLOPs for SegNet"""
    total_flops = 0
    current_shape = input_shape
    
    # Encoder configuration: (output_channels, kernel_size, stride, padding)
    encoder_configs = [
        [(64, 3, 1, 1), (64, 3, 1, 1)],          # Block 1: 2 layers
        [(128, 3, 1, 1), (128, 3, 1, 1)],        # Block 2: 2 layers
        [(256, 3, 1, 1), (256, 3, 1, 1), (256, 3, 1, 1)],  # Block 3: 3 layers
        [(512, 3, 1, 1), (512, 3, 1, 1), (512, 3, 1, 1)],  # Block 4: 3 layers
        [(512, 3, 1, 1), (512, 3, 1, 1), (512, 3, 1, 1)]   # Block 5: 3 layers
    ]
    
    # Calculate encoder FLOPs
    for block in encoder_configs:
        for out_ch, kernel, stride, padding in block:
            flops, current_shape = calculate_conv_flops(current_shape, out_ch, kernel, stride, padding)
            total_flops += flops
        # After each block, spatial dimensions are halved by MaxPool2d
        current_shape = (current_shape[0], current_shape[1] // 2, current_shape[2] // 2)
    
    # Decoder configuration (symmetric to encoder)
    decoder_configs = [
        [(512, 3, 1, 1), (512, 3, 1, 1), (512, 3, 1, 1)],  # Block 5 decoder
        [(512, 3, 1, 1), (512, 3, 1, 1), (256, 3, 1, 1)],  # Block 4 decoder
        [(256, 3, 1, 1), (256, 3, 1, 1), (128, 3, 1, 1)],  # Block 3 decoder
        [(128, 3, 1, 1), (64, 3, 1, 1)],                   # Block 2 decoder
        [(64, 3, 1, 1)]                                     # Block 1 decoder
    ]
    
    # Calculate decoder FLOPs
    for block in decoder_configs:
        # Before each decoder block, spatial dimensions are doubled by MaxUnpool2d
        current_shape = (current_shape[0], current_shape[1] * 2, current_shape[2] * 2)
        for out_ch, kernel, stride, padding in block:
            flops, current_shape = calculate_conv_flops(current_shape, out_ch, kernel, stride, padding)
            total_flops += flops
    
    # Final classification layer (1x1 conv)
    final_flops, _ = calculate_conv_flops(current_shape, n_labels, 1, 1, 0)
    total_flops += final_flops
    
    return total_flops

def get_model_parameters(model):
    """Calculate total number of parameters in the model"""
    return sum(p.numel() for p in model.parameters())

def run_ptflops_measurement(input_sizes):
    """Attempt PTFLOPS measurement with error handling"""
    print("PTFLOPS Measurements:")
    print("=" * 60)
    print(f"{'Input Size':>20} | {'PTFLOPS (G)':>12} | {'Parameters (M)':>15}")
    print("-" * 60)
    
    for inp in input_sizes:
        try:
            model = SegNet(input_channels=3, n_labels=21)
            model.eval()
            
            # Convert input size to (C, H, W) format for ptflops
            input_shape = (inp[0], inp[2], inp[1])  # (C, H, W)
            
            # Use ptflops to calculate complexity
            flops, params = get_model_complexity_info(
                model, 
                input_shape, 
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False
            )
            
            print(f"{inp[0]}x{inp[1]}x{inp[2]:<8} | {flops/1e9:11.2f} | {params/1e6:14.2f}")
            
        except Exception as e:
            print(f"{inp[0]}x{inp[1]}x{inp[2]:<8} | {'Error':>11} | {'N/A':>14}")

def main():
    """Main execution function"""
    input_sizes = [
        (3, 640, 360),
        #(3, 1280, 720),
        #(3, 1360, 760),
        #(3, 1600, 900),
        #(3, 1920, 1080),
       # (3, 2048, 1152),
       # (3, 2560, 1440)
       # (3, 3840, 2160)
    ]
    
    # Get model parameters (constant across all input sizes)
    model = SegNet(input_channels=3, n_labels=21)
    total_params = get_model_parameters(model)
    
    # Calculate theoretical FLOPs
    print("THEORETICAL FLOPS CALCULATION FOR SEGNET")
    print("=" * 60)
    print(f"{'Input Size':>20} | {'Theoretical FLOPs (G)':>20} | {'Parameters (M)':>15}")
    print("-" * 60)
    
    for inp in input_sizes:
        theoretical_flops = calculate_segnet_theoretical_flops(inp)
        print(f"{inp[0]}x{inp[1]}x{inp[2]:<8} | {theoretical_flops/1e9:19.2f} | {total_params/1e6:14.2f}")
    
    print("\n")
    
    # Attempt PTFLOPS measurement
    run_ptflops_measurement(input_sizes)

if __name__ == "__main__":
    main()
