import torch
import torch.nn as nn
import time
import numpy as np
import json

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

# FLOPs and Params calculator for each Conv layer
def calculate_flops_and_params(input_shape, output_channels, kernel_size=(3, 3), stride=(1, 1)):
    in_channels = input_shape[2]
    out_height = (input_shape[0] - kernel_size[0] + 2) // stride[0] + 1
    out_width = (input_shape[1] - kernel_size[1] + 2) // stride[1] + 1

    params = (kernel_size[0] * kernel_size[1] * in_channels + 1) * output_channels
    mults = (kernel_size[0] * kernel_size[1] * in_channels) * out_height * out_width * output_channels
    divs = out_height * out_width * output_channels
    add_subs = (kernel_size[0] * kernel_size[1] * in_channels - 1) * out_height * out_width * output_channels
    flops = mults + divs + add_subs

    return params, flops, (out_height, out_width, output_channels)

# Main function to benchmark multiple input sizes
def run_all_benchmarks():
    input_sizes = [
        (360, 640),    # 3x640x360
        (720, 1280),   # 3x1280x720
        (760, 1360),   # 3x1360x760
        (900, 1600),   # 3x1600x900
        (1080, 1920),  # 3x1920x1080
        (1152, 2048),  # 3x2048x1152
        (1440, 2560),  # 3x2560x1440
    ]

    encoder_channels = [64, 64, 128, 128, 256, 256, 512, 512]
    kernel_size = (3, 3)
    stride = (1, 1)
    results = []

    for (h, w) in input_sizes:
        print(f"\n📏 Input size: 3 x {h} x {w}")
        curr_shape = (h, w, 3)
        total_params = 0
        total_flops = 0

        # Define input for benchmarking latency
        input_tensor = torch.randn(1, 3, h, w)
        model = UNetEncoderNoSkip()
        model.eval()

        # Benchmark latency
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

        # Count params and flops layer by layer
        for i, out_channels in enumerate(encoder_channels):
            params, flops, output_shape = calculate_flops_and_params(curr_shape, out_channels, kernel_size, stride)
            total_params += params
            total_flops += flops
            curr_shape = output_shape

            # After enc1, enc2, enc3 → MaxPool
            if i in [1, 3, 5]:
                curr_shape = (curr_shape[0] // 2, curr_shape[1] // 2, curr_shape[2])

        results.append({
            "input_shape": f"3 x {h} x {w}",
            "total_params": total_params,
            "total_flops": total_flops,
            "latency_ms_mean": round(mean_latency, 2),
            "latency_ms_p95": round(p95_latency, 2),
            "latency_ms_p99": round(p99_latency, 2),
        })

    # Save to file
    with open("unet_encoder_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Print
    for r in results:
        print(f"\n✅ {r['input_shape']}")
        print(f"  Params: {r['total_params'] / 1e6:.2f}M")
        print(f"  FLOPs: {r['total_flops'] / 1e9:.2f}B")
        print(f"  Latency: Mean = {r['latency_ms_mean']} ms, P95 = {r['latency_ms_p95']} ms, P99 = {r['latency_ms_p99']} ms")

# Run all benchmarks
if __name__ == "__main__":
    run_all_benchmarks()
