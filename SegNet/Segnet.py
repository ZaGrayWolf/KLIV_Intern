#!/usr/bin/env python3
"""
SegNet Profiling Script - Complete Analysis with Activation Memory
- Calculates forward pass activation memory usage
- Measures latency for all input sizes
- Provides theoretical and measured FLOPs
- Handles all MaxUnpool2d shape issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import warnings
warnings.filterwarnings("ignore")

# Try to import profiling libraries with graceful fallbacks
try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    PTFLOPS_AVAILABLE = False

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

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
        # Store sizes for unpooling
        sizes = []
        indices = []
        
        # Encoder Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        sizes.append(x.size())
        x, idx = self.pool(x)
        indices.append(idx)
        
        # Encoder Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        sizes.append(x.size())
        x, idx = self.pool(x)
        indices.append(idx)
        
        # Encoder Block 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))
        sizes.append(x.size())
        x, idx = self.pool(x)
        indices.append(idx)
        
        # Encoder Block 4
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = F.relu(self.bn4_3(self.conv4_3(x)))
        sizes.append(x.size())
        x, idx = self.pool(x)
        indices.append(idx)
        
        # Encoder Block 5
        x = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(x)))
        x = F.relu(self.bn5_3(self.conv5_3(x)))
        sizes.append(x.size())
        x, idx = self.pool(x)
        indices.append(idx)
        
        # Decoder - reverse order
        # Decoder Block 5
        x = self.unpool(x, indices[4], output_size=sizes[4])
        x = F.relu(self.bn5_3_D(self.conv5_3_D(x)))
        x = F.relu(self.bn5_2_D(self.conv5_2_D(x)))
        x = F.relu(self.bn5_1_D(self.conv5_1_D(x)))
        
        # Decoder Block 4
        x = self.unpool(x, indices[3], output_size=sizes[3])
        x = F.relu(self.bn4_3_D(self.conv4_3_D(x)))
        x = F.relu(self.bn4_2_D(self.conv4_2_D(x)))
        x = F.relu(self.bn4_1_D(self.conv4_1_D(x)))
        
        # Decoder Block 3
        x = self.unpool(x, indices[2], output_size=sizes[2])
        x = F.relu(self.bn3_3_D(self.conv3_3_D(x)))
        x = F.relu(self.bn3_2_D(self.conv3_2_D(x)))
        x = F.relu(self.bn3_1_D(self.conv3_1_D(x)))
        
        # Decoder Block 2
        x = self.unpool(x, indices[1], output_size=sizes[1])
        x = F.relu(self.bn2_2_D(self.conv2_2_D(x)))
        x = F.relu(self.bn2_1_D(self.conv2_1_D(x)))
        
        # Decoder Block 1
        x = self.unpool(x, indices[0], output_size=sizes[0])
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

def calculate_activation_memory(input_shape):
    """Calculate forward pass activation memory for SegNet"""
    total_activation_mb = 0
    max_activation_mb = 0
    current_shape = input_shape
    
    # Encoder activation sizes
    encoder_channels = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    
    for channels in encoder_channels:
        # Calculate activation size for this layer (float32 = 4 bytes)
        activation_elements = channels * current_shape[1] * current_shape[2]
        activation_mb = (activation_elements * 4) / (1024 ** 2)
        
        total_activation_mb += activation_mb
        max_activation_mb = max(max_activation_mb, activation_mb)
        
        # Update shape for next layer
        current_shape = (channels, current_shape[1], current_shape[2])
        
        # After every 2-3 layers, spatial dimensions are halved
        if channels in [64, 128, 256, 512]:
            current_shape = (current_shape[0], current_shape[1] // 2, current_shape[2] // 2)
    
    # Decoder activation sizes (symmetric to encoder)
    decoder_channels = [512, 512, 512, 512, 512, 256, 256, 256, 128, 128, 64, 64]
    
    for channels in decoder_channels:
        # Before decoder layers, spatial dimensions are doubled
        if channels in [512, 256, 128, 64]:
            current_shape = (current_shape[0], current_shape[1] * 2, current_shape[2] * 2)
        
        activation_elements = channels * current_shape[1] * current_shape[2]
        activation_mb = (activation_elements * 4) / (1024 ** 2)
        
        total_activation_mb += activation_mb
        max_activation_mb = max(max_activation_mb, activation_mb)
        
        current_shape = (channels, current_shape[1], current_shape[2])
    
    # Final classification layer
    final_elements = 21 * input_shape[1] * input_shape[2]  # n_labels = 21
    final_mb = (final_elements * 4) / (1024 ** 2)
    total_activation_mb += final_mb
    max_activation_mb = max(max_activation_mb, final_mb)
    
    return total_activation_mb, max_activation_mb

def get_model_parameters(model):
    """Calculate total number of parameters in the model"""
    return sum(p.numel() for p in model.parameters())

def test_model_forward(input_sizes):
    """Test that the model can perform forward pass without errors"""
    print("TESTING MODEL FORWARD PASS")
    print("=" * 50)
    print(f"{'Input Size':>20} | {'Forward Pass':>15} | {'Output Shape':>20}")
    print("-" * 60)
    
    device = torch.device('cpu')  # Force CPU to avoid device errors
    
    for inp in input_sizes:
        try:
            model = SegNet(input_channels=3, n_labels=21).to(device)
            model.eval()
            dummy_input = torch.randn(1, *inp).to(device)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"{inp[0]}x{inp[1]}x{inp[2]:<8} | {'SUCCESS':>15} | {str(tuple(output.shape)):>20}")
            
        except Exception as e:
            print(f"{inp[0]}x{inp[1]}x{inp[2]:<8} | {'FAILED':>15} | {str(e)[:20]:>20}")

def run_thop_measurement(input_sizes):
    """Attempt THOP measurement with error handling"""
    if not THOP_AVAILABLE:
        print("THOP not available. Install with: pip install thop")
        return
        
    print("THOP (PyTorch-OpCounter) Measurements:")
    print("=" * 70)
    print(f"{'Input Size':>20} | {'FLOPs (G)':>12} | {'MACs (G)':>12} | {'Parameters (M)':>15}")
    print("-" * 70)
    
    device = torch.device('cpu')  # Force CPU to avoid device errors
    
    for inp in input_sizes:
        try:
            model = SegNet(input_channels=3, n_labels=21).to(device)
            model.eval()
            dummy_input = torch.randn(1, *inp).to(device)
            
            # Use THOP to profile the model
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            
            print(f"{inp[0]}x{inp[1]}x{inp[2]:<8} | {flops/1e9:11.2f} | {flops/(2*1e9):11.2f} | {params/1e6:14.2f}")
            
        except Exception as e:
            print(f"{inp[0]}x{inp[1]}x{inp[2]:<8} | {'Error':>11} | {'Error':>11} | {'N/A':>14}")

def measure_latency_all_inputs(input_sizes, num_runs=5):
    """Measure inference latency for ALL input sizes"""
    print("LATENCY MEASUREMENTS (CPU) - ALL INPUT SIZES")
    print("=" * 60)
    print(f"{'Input Size':>20} | {'Avg Latency (ms)':>18}")
    print("-" * 42)
    
    device = torch.device('cpu')
    
    for inp in input_sizes:
        try:
            model = SegNet(input_channels=3, n_labels=21).to(device)
            model.eval()
            dummy_input = torch.randn(1, *inp).to(device)
            
            # Warmup (reduced for larger inputs)
            warmup_runs = 2 if inp[1] * inp[2] > 1000000 else 3
            with torch.no_grad():
                for _ in range(warmup_runs):
                    _ = model(dummy_input)
            
            # Measure latency (reduced runs for very large inputs)
            actual_runs = 3 if inp[1] * inp[2] > 2000000 else num_runs
            timings = []
            
            for _ in range(actual_runs):
                start_time = time.perf_counter()
                with torch.no_grad():
                    _ = model(dummy_input)
                end_time = time.perf_counter()
                timings.append((end_time - start_time) * 1000)  # Convert to ms
            
            avg_latency = sum(timings) / len(timings)
            print(f"{inp[0]}x{inp[1]}x{inp[2]:<8} | {avg_latency:17.2f}")
            
        except Exception as e:
            print(f"{inp[0]}x{inp[1]}x{inp[2]:<8} | Error: {str(e)[:15]}")

def main():
    """Main execution function"""
    print("SegNet Profiling Script - Complete Analysis with Activation Memory")
    print("=" * 80)
    print("Note: Running on CPU to avoid device compatibility issues")
    print("=" * 80)
    
    input_sizes = [
        (3, 640, 360),
        (3, 1280, 720),
        (3, 1360, 760),
        (3, 1600, 900),
        (3, 1920, 1080),
        (3, 2048, 1152),
        (3, 2560, 1440),
        (3, 3840, 2160)
    ]
    
    # Test model forward pass first
    test_model_forward(input_sizes)
    print("\n")
    
    # Get model parameters (constant across all input sizes)
    model = SegNet(input_channels=3, n_labels=21)
    total_params = get_model_parameters(model)
    
    # Calculate theoretical FLOPs and activation memory
    print("THEORETICAL ANALYSIS FOR SEGNET")
    print("=" * 80)
    print(f"{'Input Size':>15} | {'FLOPs (G)':>10} | {'Params (M)':>10} | {'Total Act (MB)':>14} | {'Peak Act (MB)':>13}")
    print("-" * 80)
    
    for inp in input_sizes:
        theoretical_flops = calculate_segnet_theoretical_flops(inp)
        total_act_mb, peak_act_mb = calculate_activation_memory(inp)
        print(f"{inp[0]}x{inp[1]}x{inp[2]:<5} | {theoretical_flops/1e9:10.2f} | {total_params/1e6:10.2f} | {total_act_mb:14.2f} | {peak_act_mb:13.2f}")
    
    print("\n")
    
    # Attempt THOP measurement
    run_thop_measurement(input_sizes)
    
    print("\n")
    
    # Measure latency for ALL input sizes
    measure_latency_all_inputs(input_sizes)
    
    print("\nAnalysis complete!")
    print("Note: All measurements performed on CPU for maximum compatibility")
    print("Activation memory calculated for forward pass only (float32 precision)")

if __name__ == "__main__":
    main()
