#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 22:57:38 2018

@author: sumanthnandamuri
"""
import torch
import torch.nn as nn
import time
import numpy as np
import json
# Import get_model_complexity_info from ptflops
from ptflops import get_model_complexity_info

# Model Definition (UNet Encoder without Skip Connections)
class UNetEncoderNoSkip(nn.Module):
    def __init__(self):
        super(UNetEncoderNoSkip, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool1(x1)
        x3 = self.enc2(x2)
        x4 = self.pool2(x3)
        x5 = self.enc3(x4)
        x6 = self.pool3(x5)
        x7 = self.enc4(x6)
        return x7

# Manual FLOPs and Params calculator for EACH Conv layer (as provided)
# Note: This function calculates metrics for a *single* Conv layer
def calculate_flops_and_params_single_conv(input_shape, output_channels, kernel_size=(3, 3), stride=(1, 1), padding=1):
    # input_shape expected format: (Height, Width, In_channels) based on usage below
    in_channels = input_shape[2]
    input_h, input_w = input_shape[0], input_shape[1]
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride

    # Calculate output spatial dimensions
    out_height = (input_h - kernel_h + 2 * padding) // stride_h + 1
    out_width = (input_w - kernel_w + 2 * padding) // stride_w + 1


    # Parameters (including bias)
    params = (kernel_h * kernel_w * in_channels + 1) * output_channels

    # FLOPs (using the provided formula structure)
    # Note: This counts multiplications, divisions, and additions/subtractions separately
    # and sums them. A common convention is 2 * MACs for FLOPs (1 mult + 1 add).
    # This formula may differ from ptflops.
    mults = (kernel_h * kernel_w * in_channels) * out_height * out_width * output_channels
    divs = out_height * out_width * output_channels # Division part seems unusual for basic conv?
    add_subs = (kernel_h * kernel_w * in_channels - 1) * out_height * out_width * output_channels

    total_flops = mults + divs + add_subs

    return params, total_flops, (out_height, out_width, output_channels)


# Main function to benchmark multiple input sizes
def run_all_benchmarks():
    input_spatial_sizes = [
        (360, 640),    # H, W
        (720, 1280),   # H, W
        (760, 1360),   # H, W
        (900, 1600),   # H, W
        (1080, 1920),  # H, W
        (1152, 2048),  # H, W
        (1440, 2560),  # H, W
    ]

    # Define the layer sequence and their output channels, kernel size, stride, padding
    # This mirrors the forward pass logic
    layer_configs = [
        # enc1
        {'type': 'conv', 'out_channels': 64, 'kernel_size': (3, 3), 'stride': (1, 1), 'padding': 1},
        {'type': 'relu'}, # Relu after conv1a
        {'type': 'conv', 'out_channels': 64, 'kernel_size': (3, 3), 'stride': (1, 1), 'padding': 1},
        {'type': 'relu'}, # Relu after conv1b
        {'type': 'pool', 'kernel_size': (2, 2), 'stride': (2, 2)}, # MaxPool1

        # enc2
        {'type': 'conv', 'out_channels': 128, 'kernel_size': (3, 3), 'stride': (1, 1), 'padding': 1},
        {'type': 'relu'}, # Relu after conv2a
        {'type': 'conv', 'out_channels': 128, 'kernel_size': (3, 3), 'stride': (1, 1), 'padding': 1},
        {'type': 'relu'}, # Relu after conv2b
        {'type': 'pool', 'kernel_size': (2, 2), 'stride': (2, 2)}, # MaxPool2

        # enc3
        {'type': 'conv', 'out_channels': 256, 'kernel_size': (3, 3), 'stride': (1, 1), 'padding': 1},
        {'type': 'relu'}, # Relu after conv3a
        {'type': 'conv', 'out_channels': 256, 'kernel_size': (3, 3), 'stride': (1, 1), 'padding': 1},
        {'type': 'relu'}, # Relu after conv3b
        {'type': 'pool', 'kernel_size': (2, 2), 'stride': (2, 2)}, # MaxPool3

        # enc4
        {'type': 'conv', 'out_channels': 512, 'kernel_size': (3, 3), 'stride': (1, 1), 'padding': 1},
        {'type': 'relu'}, # Relu after conv4a
        {'type': 'conv', 'out_channels': 512, 'kernel_size': (3, 3), 'stride': (1, 1), 'padding': 1},
        {'type': 'relu'}, # Relu after conv4b (final layer in this model)
    ]


    results = []

    for (h, w) in input_spatial_sizes:
        print(f"\n--- Analyzing Input Size: 3 x {h} x {w} ---")

        # --- Manual Calculation (using calculate_flops_and_params_single_conv) ---
        # Note: This manual calculation *only* counts FLOPs/params for the Conv layers
        # and assumes ReLU/Pool have negligible FLOPs based on the structure of the provided function
        # and how it was previously used. It uses the specific FLOPs definition from that function.
        print("  Performing Manual Calculation (Conv layers only)...")
        curr_shape_manual = (h, w, 3) # Input shape for manual function (H, W, C)
        total_params_manual = 0
        total_flops_manual = 0 # FLOPs using the provided formula structure

        # Need to map the layer_configs to the manual function calls
        manual_input_channels = 3 # Start with 3 input channels
        manual_current_spatial_shape = (h, w)

        for i, config in enumerate(layer_configs):
            if config['type'] == 'conv':
                params, flops, output_spatial_shape = calculate_flops_and_params_single_conv(
                    manual_current_spatial_shape + (manual_input_channels,), # (H, W, C_in)
                    config['out_channels'],
                    config['kernel_size'],
                    config['stride'],
                    config.get('padding', 0) # Get padding, default to 0 if not specified
                )
                total_params_manual += params
                total_flops_manual += flops
                manual_input_channels = config['out_channels']
                manual_current_spatial_shape = output_spatial_shape[:2] # Update H, W

            elif config['type'] == 'pool':
                 # MaxPool reduces spatial size, manual function needs updated shape
                 pool_k = config['kernel_size'][0] # Assuming square kernel
                 pool_s = config['stride'][0] # Assuming square stride
                 manual_current_spatial_shape = (
                     (manual_current_spatial_shape[0] - pool_k) // pool_s + 1,
                     (manual_current_spatial_shape[1] - pool_k) // pool_s + 1
                 )
                 # Manual function doesn't calculate FLOPs for pooling, assuming 0
            elif config['type'] == 'relu':
                 # Manual function doesn't calculate FLOPs for ReLU, assuming 0 or negligible
                 pass


        print(f"  Manual Params (Conv layers only): {total_params_manual:,}")
        print(f"  Manual FLOPs (Conv layers only, using provided formula): {total_flops_manual:,}")
        print(f"  Manual Params (Conv layers only): {total_params_manual / 1e6:.2f} M")
        print(f"  Manual FLOPs (Conv layers only, using provided formula): {total_flops_manual / 1e9:.2f} G")


        # --- PTFlops Calculation ---
        print("  Performing PTFlops Calculation (Full Model)...")
        model = UNetEncoderNoSkip()
        model.eval()
        input_size_ptflops = (3, h, w) # PTFlops expects (C, H, W)

        try:
            # Use ptflops to get complexity info
            with torch.no_grad():
                flops_pt, params_pt = get_model_complexity_info(model, input_size_ptflops,
                                                                as_strings=False, # Get numerical values
                                                                print_per_layer_stat=False,
                                                                verbose=False)

            # ptflops typically reports MACs for Conv. Common to multiply by 2 for FLOPs.
            # PTFlops parameters include all trainable parameters in the model.
            print(f"  PTFlops Parameters: {params_pt:,}")
            print(f"  PTFlops (MACs): {flops_pt:,}")
            print(f"  PTFlops (Approx. 2*MACs): {flops_pt * 2:,}") # Assuming 1 MAC = 2 FLOPs

            print(f"  PTFlops Parameters: {params_pt / 1e6:.2f} M")
            print(f"  PTFlops (MACs): {flops_pt / 1e9:.2f} G")
            print(f"  PTFlops (Approx. 2*MACs): {(flops_pt * 2) / 1e9:.2f} GFLOPs")


        except Exception as e:
            print(f"  An error occurred during PTFlops calculation for {h}x{w}: {e}")
            import traceback
            traceback.print_exc()

        # --- Latency Benchmark ---
        print("  Performing Latency Benchmark...")
        # Define input for benchmarking latency
        input_tensor = torch.randn(1, 3, h, w)
        # Model is already instantiated above for ptflops, reuse it
        # model = UNetEncoderNoSkip() # Re-instantiate if needed, but reusing is faster
        # model.eval() # Make sure it's still in eval mode

        with torch.no_grad():
            for _ in range(10):  # Warm-up
                _ = model(input_tensor)

            latencies = []
            for _ in range(50):
                start = time.time()
                _ = model(input_tensor)
                latencies.append((time.time() - start) * 1000)

        mean_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        print(f"  Latency: Mean = {mean_latency:.2f} ms, P95 = {p95_latency:.2f} ms, P99 = {p99_latency:.2f} ms")
        print("-" * 40)


        # Store results (Optional: saving to JSON as in original code)
        # We'll store both sets of metrics for completeness if saving
        results.append({
            "input_shape": f"3 x {h} x {w}",
            "manual_params_conv_only": total_params_manual,
            "manual_flops_conv_only_provided_formula": total_flops_manual,
            "ptflops_params": params_pt,
            "ptflops_macs": flops_pt,
            "ptflops_flops_approx_2xmacs": flops_pt * 2, # Store the 2*MACs value
            "latency_ms_mean": round(mean_latency, 2),
            "latency_ms_p95": round(p95_latency, 2),
            "latency_ms_p99": round(p99_latency, 2),
        })


    # Save results to file (optional)
    # with open("unet_encoder_benchmark_results.json", "w") as f:
    #     json.dump(results, f, indent=4)

    # Print final summary (optional, detailed prints are done per input size)
    # You can uncomment this if you prefer a summary at the end
    # print("\n--- Benchmark Summary ---")
    # for r in results:
    #      print(f"\n✅ {r['input_shape']}")
    #      print(f"  Manual Params (Conv): {r['manual_params_conv_only'] / 1e6:.2f}M")
    #      print(f"  Manual FLOPs (Conv, provided formula): {r['manual_flops_conv_only_provided_formula'] / 1e9:.2f} G")
    #      print(f"  PTFlops Params: {r['ptflops_params'] / 1e6:.2f} M")
    #      print(f"  PTFlops FLOPs (Approx. 2xMACs): {r['ptflops_flops_approx_2xmacs'] / 1e9:.2f} GFLOPs")
    #      print(f"  Latency: Mean = {r['latency_ms_mean']} ms")
    # print("-------------------------")


# Run all benchmarks
if __name__ == "__main__":
    run_all_benchmarks()