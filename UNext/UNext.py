import torch
from torch import nn
import torch.nn.functional as F
import math
import time

# Ensure timm is installed: pip install timm
from timm.layers import to_2tuple, DropPath, trunc_normal_

# Helper to calculate Conv2d output shape
def get_conv_output_shape(input_shape, kernel_size, stride, padding, dilation=1):
    B, C_in, H_in, W_in = input_shape
    K_H, K_W = to_2tuple(kernel_size)
    S_H, S_W = to_2tuple(stride)
    P_H, P_W = to_2tuple(padding)
    D_H, D_W = to_2tuple(dilation)

    H_in, W_in = int(H_in), int(W_in)
    P_H, P_W = int(P_H), int(P_W)
    D_H, D_W = int(D_H), int(D_W)
    K_H, K_W = int(K_H), int(K_W)
    S_H, S_W = int(S_H), int(S_W)

    H_out = math.floor(((H_in + 2 * P_H - D_H * (K_H - 1) - 1) / S_H) + 1)
    W_out = math.floor(((W_in + 2 * P_W - D_W * (K_W - 1) - 1) / S_W) + 1)
    return H_out, W_out

# Helper to calculate Pooling output shape
def get_pool_output_shape(input_shape, kernel_size, stride, padding=0, dilation=1):
    B, C_in, H_in, W_in = input_shape
    K_H, K_W = to_2tuple(kernel_size)
    S_H, S_W = to_2tuple(stride)
    P_H, P_W = to_2tuple(padding)
    D_H, D_W = to_2tuple(dilation)

    H_in, W_in = int(H_in), int(W_in)
    P_H, P_W = int(P_H), int(P_W)
    K_H, K_W = int(K_H), int(K_W)
    S_H, S_W = int(S_H), int(S_W)

    H_out = math.floor(((H_in + 2 * P_H - K_H) / S_H) + 1)
    W_out = math.floor(((W_in + 2 * P_W - K_W) / S_W) + 1)
    return H_out, W_out

# --- Model Components ---
class DWConv(nn.Module):
    """ Depth-wise convolution """
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class shiftmlp(nn.Module):
    """ Shifted MLP """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        return x

class shiftedBlock(nn.Module):
    """ Block combining LayerNorm, MLP, and residual connection """
    def __init__(self, dim, mlp_ratio=4., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding with overlapping patches """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        # Padding calculation consistent with typical implementations
        padding = (patch_size[0] // 2, patch_size[1] // 2) if patch_size[0] > 1 else (0,0)


        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        H_out, W_out = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H_out, W_out

# --- UNext Model Definitions ---
class UNext(nn.Module):
    def __init__(self, num_classes=4, input_channels=3, depths=[1, 1], mlp_ratios=[4., 4.], drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dims = [16, 32, 128, 160, 256]
        self.decoder_dims = [160, 128, 32, 16, 16]
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

        # Transformer Encoder Stages
        self.patch_embed3 = OverlapPatchEmbed(in_chans=self.dims[2], embed_dim=self.dims[3], patch_size=3, stride=2)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths) * 2)] # Total blocks in encoder + decoder transformers
        cur = 0
        self.block1 = nn.ModuleList([shiftedBlock(dim=self.dims[3], mlp_ratio=mlp_ratios[0], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[0])])
        self.norm3 = norm_layer(self.dims[3])
        cur += depths[0]

        self.patch_embed4 = OverlapPatchEmbed(in_chans=self.dims[3], embed_dim=self.dims[4], patch_size=3, stride=2)
        self.block2 = nn.ModuleList([shiftedBlock(dim=self.dims[4], mlp_ratio=mlp_ratios[1], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[1])])
        self.norm4 = norm_layer(self.dims[4])
        cur += depths[1]

        # Decoder
        self.decoder1 = nn.Conv2d(self.dims[4] + self.dims[3], self.decoder_dims[0], 3, padding=1)
        self.dbn1 = nn.BatchNorm2d(self.decoder_dims[0])
        self.dblock1 = nn.ModuleList([shiftedBlock(dim=self.decoder_dims[0], mlp_ratio=mlp_ratios[0], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[0])])
        self.dnorm3 = norm_layer(self.decoder_dims[0])
        cur += depths[0]
        
        self.decoder2 = nn.Conv2d(self.decoder_dims[0] + self.dims[2], self.decoder_dims[1], 3, padding=1)
        self.dbn2 = nn.BatchNorm2d(self.decoder_dims[1])
        self.dblock2 = nn.ModuleList([shiftedBlock(dim=self.decoder_dims[1], mlp_ratio=mlp_ratios[1], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[1])])
        self.dnorm4 = norm_layer(self.decoder_dims[1])
        
        self.decoder3 = nn.Conv2d(self.decoder_dims[1] + self.dims[1], self.decoder_dims[2], 3, padding=1)
        self.dbn3 = nn.BatchNorm2d(self.decoder_dims[2])
        self.decoder4 = nn.Conv2d(self.decoder_dims[2] + self.dims[0], self.decoder_dims[3], 3, padding=1)
        self.dbn4 = nn.BatchNorm2d(self.decoder_dims[3])
        self.decoder5 = nn.Conv2d(self.decoder_dims[3], self.decoder_dims[4], 3, padding=1)
        self.dbn5 = nn.BatchNorm2d(self.decoder_dims[4])
        self.final = nn.Conv2d(self.decoder_dims[4], num_classes, 1)

    def _decode_block(self, x, skip, conv, bn, blocks, norm):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(bn(conv(x)))
        # Reshape for shiftedBlock
        B_dec, C_dec, H_dec, W_dec = x.shape
        x_flat = x.flatten(2).transpose(1, 2) # (B, N, C)
        for blk in blocks:
            x_flat = blk(x_flat, H_dec, W_dec)
        # Reshape back
        x = norm(x_flat).transpose(1, 2).reshape(B_dec, C_dec, H_dec, W_dec)
        return x

    def _upsample(self, x, skip, conv, bn, target_size_override=None):
        if target_size_override:
            size = target_size_override
        else:
            size = skip.shape[2:] if skip is not None else None
        
        if size is None: # For final upsample if no skip or if target_size_override is not given for it
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        else:
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.relu(bn(conv(x)))

    def forward(self, x):
        input_size = x.shape[2:] # Store original input size for final interpolation

        # Encoder
        e1 = self.relu(self.ebn1(self.encoder1(x)))     # (B, dims[0], H, W)
        e2_in = self.maxpool(e1)                       # (B, dims[0], H/2, W/2)
        e2 = self.relu(self.ebn2(self.encoder2(e2_in)))# (B, dims[1], H/2, W/2)
        e3_in = self.maxpool(e2)                       # (B, dims[1], H/4, W/4)
        e3 = self.relu(self.ebn3(self.encoder3(e3_in)))# (B, dims[2], H/4, W/4)
        
        # Transformer Stage 1 (t4)
        t4_in = self.maxpool(e3) # (B, dims[2], H/8, W/8)
        t4, h4, w4 = self.patch_embed3(t4_in) # t4: (B, N4, dims[3]), N4=h4*w4
        for blk in self.block1:
            t4 = blk(t4, h4, w4)
        t4 = self.norm3(t4)
        # Reshape t4 from (B, N4, dims[3]) to (B, dims[3], h4, w4) for conv/interpolate
        t4_spatial = t4.transpose(1, 2).reshape(t4.shape[0], self.dims[3], h4, w4)
        
        # Transformer Stage 2 (t5 - bottleneck)
        t5, h5, w5 = self.patch_embed4(t4_spatial) # t5: (B, N5, dims[4]), N5=h5*w5
        for blk in self.block2:
            t5 = blk(t5, h5, w5)
        t5 = self.norm4(t5)
        # Reshape t5 from (B, N5, dims[4]) to (B, dims[4], h5, w5)
        t5_spatial = t5.transpose(1, 2).reshape(t5.shape[0], self.dims[4], h5, w5)
        
        # Decoder
        d1 = self._decode_block(t5_spatial, t4_spatial, self.decoder1, self.dbn1, self.dblock1, self.dnorm3)
        d2 = self._decode_block(d1, e3, self.decoder2, self.dbn2, self.dblock2, self.dnorm4)
        d3 = self._upsample(d2, e2, self.decoder3, self.dbn3)
        d4 = self._upsample(d3, e1, self.decoder4, self.dbn4)
        # Last upsample needs to go to original input size, not just scale_factor=2 if e1 was already small
        d5 = self._upsample(d4, None, self.decoder5, self.dbn5, target_size_override=input_size) # No skip, direct to input_size for first decoder5
        
        # Final interpolation to match input size, if d5 is not already there (e.g. if input_size was odd)
        # This ensures output is exactly same HxW as input x
        # However, the decoder5 already uses target_size_override=input_size for its F.interpolate.
        # So, an additional F.interpolate(d5, input_size) after final conv might be redundant if dbn5(decoder5(d5_up)) already matches input_size.
        # For UNet-like structures, the final conv often maps to classes, and then a final upsample ensures size match.
        # Let's assume the self._upsample(..., target_size_override=input_size) for d5 handles this.
        # If dbn5(decoder5(upsampled_d4)) does not change spatial dimensions then it should be fine.
        # The original search result had F.interpolate(d5, x.shape[2:]...) *after* self.final. Let's stick to that.

        out = self.final(d5)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        return out

class UNext_S(UNext):
    def __init__(self, num_classes=4, input_channels=3, depths=[1, 1], mlp_ratios=[4., 4.], drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__(num_classes, input_channels, depths, mlp_ratios, drop_path_rate, norm_layer)
        # Override dimensions for the "Small" version
        self.dims = [8, 16, 32, 64, 128]
        self.decoder_dims = [64, 32, 16, 8, 8]

        # Re-initialize layers that depend on these dims
        # Encoder
        self.encoder1 = nn.Conv2d(input_channels, self.dims[0], 3, padding=1)
        self.ebn1 = nn.BatchNorm2d(self.dims[0])
        self.encoder2 = nn.Conv2d(self.dims[0], self.dims[1], 3, padding=1)
        self.ebn2 = nn.BatchNorm2d(self.dims[1])
        self.encoder3 = nn.Conv2d(self.dims[1], self.dims[2], 3, padding=1)
        self.ebn3 = nn.BatchNorm2d(self.dims[2])

        # Transformer Encoder Stages
        self.patch_embed3 = OverlapPatchEmbed(in_chans=self.dims[2], embed_dim=self.dims[3], patch_size=3, stride=2)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths) * 2)]
        cur = 0
        self.block1 = nn.ModuleList([shiftedBlock(dim=self.dims[3], mlp_ratio=mlp_ratios[0], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[0])])
        self.norm3 = norm_layer(self.dims[3])
        cur += depths[0]

        self.patch_embed4 = OverlapPatchEmbed(in_chans=self.dims[3], embed_dim=self.dims[4], patch_size=3, stride=2)
        self.block2 = nn.ModuleList([shiftedBlock(dim=self.dims[4], mlp_ratio=mlp_ratios[1], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[1])])
        self.norm4 = norm_layer(self.dims[4])
        cur += depths[1]
        
        # Decoder
        self.decoder1 = nn.Conv2d(self.dims[4] + self.dims[3], self.decoder_dims[0], 3, padding=1)
        self.dbn1 = nn.BatchNorm2d(self.decoder_dims[0])
        self.dblock1 = nn.ModuleList([shiftedBlock(dim=self.decoder_dims[0], mlp_ratio=mlp_ratios[0], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[0])])
        self.dnorm3 = norm_layer(self.decoder_dims[0])
        cur += depths[0]
        
        self.decoder2 = nn.Conv2d(self.decoder_dims[0] + self.dims[2], self.decoder_dims[1], 3, padding=1)
        self.dbn2 = nn.BatchNorm2d(self.decoder_dims[1])
        self.dblock2 = nn.ModuleList([shiftedBlock(dim=self.decoder_dims[1], mlp_ratio=mlp_ratios[1], drop_path=dpr[cur + i], norm_layer=norm_layer)
                                     for i in range(depths[1])])
        self.dnorm4 = norm_layer(self.decoder_dims[1])
        
        self.decoder3 = nn.Conv2d(self.decoder_dims[1] + self.dims[1], self.decoder_dims[2], 3, padding=1)
        self.dbn3 = nn.BatchNorm2d(self.decoder_dims[2])
        self.decoder4 = nn.Conv2d(self.decoder_dims[2] + self.dims[0], self.decoder_dims[3], 3, padding=1)
        self.dbn4 = nn.BatchNorm2d(self.decoder_dims[3])
        self.decoder5 = nn.Conv2d(self.decoder_dims[3], self.decoder_dims[4], 3, padding=1)
        self.dbn5 = nn.BatchNorm2d(self.decoder_dims[4])
        self.final = nn.Conv2d(self.decoder_dims[4], num_classes, 1)


# --- Latency and Memory Profiling Function ---
def measure_model_latency_and_memory(model, input_shape, warmup_runs=5, measurement_runs=100):
    """
    Measure actual model latency and forward pass memory usage with proper CUDA synchronization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval() # Set model to evaluation mode
    
    input_tensor = torch.randn(input_shape, device=device, dtype=torch.float32)
    
    print(f"  Device: {device}")
    print(f"  Warmup runs: {warmup_runs}")
    
    # Warmup runs
    if device.type == 'cuda':
        torch.cuda.empty_cache() # Clear cache before measurements
        torch.cuda.reset_peak_memory_stats(device) # Reset memory stats for the specific device
    
    with torch.no_grad(): # Disable gradient calculations for warmup and inference
        for _ in range(warmup_runs):
            _ = model(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize() # Wait for all CUDA operations to complete
    
    # Reset memory tracking specifically for forward pass measurement
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    
    # Measure forward activation memory (peak memory during inference)
    forward_activation_memory_bytes = 0
    if device.type == 'cuda':
        baseline_memory_bytes = torch.cuda.memory_allocated(device)
        with torch.no_grad():
            _ = model(input_tensor)
            torch.cuda.synchronize()
        forward_peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        # The peak memory includes model weights, input tensor, output tensor, and activations.
        # For "activation size", we often mean the extra memory *beyond* model and persistent tensors.
        # However, max_memory_allocated gives the total peak.
        # A more precise way to get *just* activations is complex and often involves hooks.
        # Here, we'll report peak allocated during forward pass.
        forward_activation_memory_bytes = forward_peak_memory_bytes # This is the peak usage
    else: # CPU memory measurement is less direct with PyTorch tools for peak activation
        print("  CPU memory profiling for activations is not as precise as CUDA; reporting general usage.")
        # For CPU, process-level memory would be needed, which is outside torch.cuda.

    print(f"  Measurement runs for latency: {measurement_runs}")
    
    # Latency measurement using CUDA events for GPU, time.perf_counter for CPU
    latencies_ms = []
    
    with torch.no_grad():
        for i in range(measurement_runs):
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else: # CPU timing
                start_time = time.perf_counter()
            
            _ = model(input_tensor) # Forward pass
            
            if device.type == 'cuda':
                end_event.record()
                torch.cuda.synchronize() # IMPORTANT: Wait for GPU operations to finish before getting time
                elapsed_time_ms = start_event.elapsed_time(end_event)
            else: # CPU timing
                end_time = time.perf_counter()
                elapsed_time_ms = (end_time - start_time) * 1000 # Convert to milliseconds
            
            latencies_ms.append(elapsed_time_ms)
    
    # Calculate latency statistics
    avg_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0
    min_latency_ms = min(latencies_ms) if latencies_ms else 0
    max_latency_ms = max(latencies_ms) if latencies_ms else 0
    # Basic standard deviation
    if len(latencies_ms) > 1:
        mean = avg_latency_ms
        std_dev_ms = (sum([(x - mean) ** 2 for x in latencies_ms]) / (len(latencies_ms) -1)) ** 0.5 # Sample std dev
    else:
        std_dev_ms = 0
    median_latency_ms = sorted(latencies_ms)[len(latencies_ms)//2] if latencies_ms else 0

    latency_stats = {
        'mean_ms': avg_latency_ms,
        'min_ms': min_latency_ms,
        'max_ms': max_latency_ms,
        'std_ms': std_dev_ms,
        'median_ms': median_latency_ms
    }
    
    memory_stats = {
        'forward_pass_peak_memory_bytes': forward_activation_memory_bytes,
        'forward_pass_peak_memory_mb': forward_activation_memory_bytes / (1024**2)
    }
    
    return {
        'latency': latency_stats,
        'memory': memory_stats,
        'input_shape': input_shape,
        'device': str(device)
    }

# --- Main Execution Block ---
if __name__ == "__main__":
    input_sizes_to_test = [
        (360, 640),    # H, W (changed order to H, W to match typical usage in input_configs)
        (720, 1280),
        (760, 1360),
        (900, 1600),
        (1080, 1920),
        (1152, 2048),
        (1440, 2560),
        (2160, 3840),  
    ]

    # Define model configurations
    # For simplicity, using fixed depths and mlp_ratios from your original structure
    # You might want to vary these if testing different model configs
    fixed_depths = [1, 1] # Example: [depth_stage4, depth_stage5_bottleneck]
    fixed_mlp_ratios = [4., 4.] # Example: [mlp_ratio_stage4, mlp_ratio_stage5_bottleneck]
    batch_size = 1 # Common for inference latency measurement

    models_to_benchmark = [
        ("UNext", UNext(num_classes=4, input_channels=3, depths=fixed_depths, mlp_ratios=fixed_mlp_ratios)),
       # ("UNext_S", UNext_S(num_classes=4, input_channels=3, depths=fixed_depths, mlp_ratios=fixed_mlp_ratios))
    ]

    print("=" * 80)
    print("UNEXT MODEL PERFORMANCE BENCHMARKING")
    print(f"Batch Size: {batch_size}")
    print(f"Model Depths: {fixed_depths}")
    print(f"MLP Ratios: {fixed_mlp_ratios}")
    print("=" * 80)

    for model_name, model_instance in models_to_benchmark:
        print(f"\n--- Benchmarking Model: {model_name} ---")
        print("-" * 50)
        
        # Simple parameter count
        total_params = sum(p.numel() for p in model_instance.parameters())
        trainable_params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
        print(f"  Total Parameters: {total_params/1e6:.2f} M")
        print(f"  Trainable Parameters: {trainable_params/1e6:.2f} M")

        for H, W in input_sizes_to_test:
            current_input_shape = (batch_size, 3, H, W)  # (B, C, H, W)
            
            print(f"\n  Testing Input Resolution: {H}x{W}")
            print("  " + "-" * 30)
            
            try:
                # Measure actual latency and memory
                # Using 5 warmup runs and 100 measurement runs as requested
                performance_results = measure_model_latency_and_memory(
                    model_instance, 
                    current_input_shape,
                    warmup_runs=5,
                    measurement_runs=50
                )
                
                lat_stats = performance_results['latency']
                mem_stats = performance_results['memory']
                
                print(f"  Measured Performance on {performance_results['device']}:")
                print(f"    Average Latency: {lat_stats['mean_ms']:.2f} ± {lat_stats['std_ms']:.2f} ms")
                print(f"    Min/Max Latency: {lat_stats['min_ms']:.2f} / {lat_stats['max_ms']:.2f} ms")
                print(f"    Median Latency:  {lat_stats['median_ms']:.2f} ms")
                
                if performance_results['device'].startswith('cuda'):
                    print(f"    Forward Pass Peak GPU Memory: {mem_stats['forward_pass_peak_memory_mb']:.2f} MB")
                else:
                    print(f"    Forward Pass Peak Memory (CPU): Not precisely measured by this script.")

                throughput_fps = 1000.0 / lat_stats['mean_ms'] if lat_stats['mean_ms'] > 0 else 0
                print(f"    Estimated Throughput: {throughput_fps:.2f} FPS")
                
            except RuntimeError as e: # Catch potential OOM errors or other runtime issues
                print(f"  Error during measurement for {H}x{W}: {str(e)}")
                if "out of memory" in str(e).lower():
                    print("  This was likely an Out-of-Memory (OOM) error. Skipping to next resolution.")
                # break # Optionally break from inner loop if one resolution fails catastrophically
            except Exception as e:
                print(f"  An unexpected error occurred for {H}x{W}: {str(e)}")


    print("\n" + "=" * 80)
    print("BENCHMARKING COMPLETE")
    print("=" * 80)
    print("\nNotes:")
    print("- Latency measurements use CUDA events for GPU for precision.")
    print("- Forward Pass Peak Memory (GPU) is `torch.cuda.max_memory_allocated()` during inference.")
    print("  This includes model weights, input/output tensors, and activations.")
    print("- Warmup runs (5) are performed before measurements (100 runs for latency average).")
