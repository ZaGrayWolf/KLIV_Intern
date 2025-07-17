import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
from typing import Tuple, List, Dict

# --- Theoretical Calculation Functions ---

def calculate_conv_flops_params(input_shape: Tuple[int, int, int], output_channels: int, kernel_size: int | Tuple[int, int], stride: int | Tuple[int, int], padding: int | Tuple[int, int]):
    """Calculates FLOPs and Parameters for a standard Conv2d layer with bias."""
    if len(input_shape) != 3:
        raise ValueError(f"Input shape must be (H, W, C), but got {input_shape}")
    in_channels = input_shape[2]
    # Ensure kernel_size, stride, padding are tuples
    k_h, k_w = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    s_h, s_w = (stride, stride) if isinstance(stride, int) else stride
    p_h, p_w = (padding, padding) if isinstance(padding, int) else padding

    # Calculate output spatial dimensions
    out_height = (input_shape[0] - k_h + 2 * p_h) // s_h + 1
    out_width = (input_shape[1] - k_w + 2 * p_w) // s_w + 1

    if out_height <= 0 or out_width <= 0:
        # This case should ideally not happen for well-designed networks, but good to handle.
        # It usually means the kernel is too large or stride is too high for the input.
        print(f"Warning: Convolution resulted in non-positive output dimensions: ({out_height}, {out_width}) for input {input_shape}, kernel {kernel_size}, stride {stride}, padding {padding}. Returning 0 for this layer.")
        return 0, 0, (max(0, out_height), max(0, out_width), output_channels)


    # Parameters: (kernel_height * kernel_width * in_channels + bias) * output_channels
    # +1 for bias per output channel
    params = (k_h * k_w * in_channels + 1) * output_channels

    # FLOPs: Sum of multiplications and additions over all output elements
    # Each output element is calculated by: k_h * k_w * in_channels multiplications
    # and (k_h * k_w * in_channels - 1) additions. Plus 1 addition for bias.
    # Total operations per output element: k_h * k_w * in_channels (mults) + (k_h * k_w * in_channels - 1 + 1) (adds)
    # = 2 * k_h * k_w * in_channels operations (MACs converted to 2 FLOPs each for mult+add)
    # Or, specifically:
    num_macs_per_output_channel = k_h * k_w * in_channels
    mults = num_macs_per_output_channel * out_height * out_width * output_channels
    # Each output element requires (num_macs_per_output_channel - 1) additions for the sum, plus 1 for the bias
    # So, num_macs_per_output_channel additions per output element
    adds = num_macs_per_output_channel * out_height * out_width * output_channels

    total_flops = mults + adds

    return params, total_flops, (out_height, out_width, output_channels)

def calculate_bn_flops_params(input_shape: Tuple[int, int, int]):
    """Calculates FLOPs and Parameters for a BatchNorm2d layer.
       BN performs (x - mean) / sqrt(var + eps) * gamma + beta
       For each element: 1 subtraction, 1 addition (eps), 1 sqrt, 1 division, 1 multiplication (gamma), 1 addition (beta)
       Roughly 6 operations per element.
       Commonly simplified to 2 FLOPs per element (mult and add for gamma and beta scaling/shifting).
       Here, I'll use 2 FLOPs per element for consistency with many online calculators (affine transformation).
    """
    if len(input_shape) != 3:
        raise ValueError(f"Input shape must be (H, W, C), but got {input_shape}")

    num_features = input_shape[2]
    # Parameters: gamma and beta for each feature map
    params = 2 * num_features
    total_elements = input_shape[0] * input_shape[1] * input_shape[2]
    # Flops: 2 operations (1 mult, 1 add) per element for affine transformation (gamma*x + beta)
    flops = 2 * total_elements

    return params, flops, input_shape

def calculate_relu_flops(input_shape: Tuple[int, int, int]):
    """Calculates FLOPs for a ReLU activation.
       ReLU (max(0, x)) is typically considered 1 FLOP per element (a comparison).
    """
    if len(input_shape) != 3:
        raise ValueError(f"Input shape must be (H, W, C), but got {input_shape}")

    total_elements = input_shape[0] * input_shape[1] * input_shape[2]
    flops = total_elements # 1 comparison per element

    return 0, flops, input_shape # No parameters for ReLU

def calculate_pooling_flops_params(input_shape: Tuple[int, int, int], kernel_size: int | Tuple[int, int], stride: int | Tuple[int, int]):
    """Calculates FLOPs and Parameters for MaxPool2d.
       Pooling operations are generally considered to have 0 FLOPs (or very few comparisons/data movement).
    """
    if len(input_shape) != 3:
        raise ValueError(f"Input shape must be (H, W, C), but got {input_shape}")

    k_h, k_w = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    s_h, s_w = (stride, stride) if isinstance(stride, int) else stride

    # Output spatial dimensions for MaxPool2d (floor division for integer output)
    out_height = (input_shape[0] - k_h) // s_h + 1 # Common formula for max pool with padding=0
    out_width = (input_shape[1] - k_w) // s_w + 1

    if out_height <= 0 or out_width <= 0:
         print(f"Warning: Pooling resulted in non-positive output dimensions: ({out_height}, {out_width}) for input {input_shape}, kernel {kernel_size}, stride {stride}. Returning 0 for this layer.")
         return 0, 0, (max(0, out_height), max(0, out_width), input_shape[2])

    return 0, 0, (out_height, out_width, input_shape[2])

def calculate_unpooling_flops_params(input_shape: Tuple[int, int, int], kernel_size: int | Tuple[int, int], stride: int | Tuple[int, int], output_size: Tuple[int, int, int]):
    """Calculates FLOPs and Parameters for MaxUnpool2d.
       Unpooling involves data movement/placement based on indices, typically 0 FLOPs.
       Requires `output_size` from the corresponding encoder feature map.
    """
    if len(input_shape) != 3:
        raise ValueError(f"Input shape must be (H, W, C), but got {input_shape}")
    if len(output_size) != 3:
        raise ValueError(f"Output size must be (H, W, C), but got {output_size}")

    # MaxUnpool2d recovers the original size. The output shape should be provided.
    # The channels remain the same.
    return 0, 0, (output_size[0], output_size[1], input_shape[2])

def calculate_concat_flops_params(input_shapes: List[Tuple[int, int, int]]):
    """Calculates FLOPs and Parameters for torch.cat.
       Concatenation is a memory operation, typically 0 FLOPs.
       Assumes concatenation along the channel dimension (dim=1 in PyTorch tensor, so C in (H,W,C)).
       All input shapes must have the same H and W.
    """
    if not input_shapes:
        return 0, 0, None

    first_shape = input_shapes[0]
    # Check if all spatial dimensions match for concatenation
    for s in input_shapes:
        if s[0] != first_shape[0] or s[1] != first_shape[1]:
            raise ValueError(f"Spatial dimensions mismatch in concat inputs: {input_shapes}. All H and W must be identical.")

    total_channels = sum(s[2] for s in input_shapes)
    output_shape = (first_shape[0], first_shape[1], total_channels)

    return 0, 0, output_shape

def get_activation_size_mb(shape: Tuple[int, ...], bytes_per_element: int = 4):
    """Calculates the memory size of an activation map in MB."""
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    return (num_elements * bytes_per_element) / (1024 * 1024)

# --- SUMNet_all_bn PyTorch Class Definition ---

class SUMNet_all_bn(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super(SUMNet_all_bn, self).__init__()

        # Encoder
        self.conv1     = nn.Conv2d(in_ch, 64, 3, padding=1)
        self.bn1       = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2     = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2       = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.pool1     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv3a    = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3a      = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3b    = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3b      = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.pool2     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv4a    = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4a      = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4b    = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4b      = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.pool3     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv5a    = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5a      = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5b    = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5b      = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.pool4     = nn.MaxPool2d(2, 2, return_indices = True)

        # Decoder
        self.unpool4   = nn.MaxUnpool2d(2, 2)
        # donv5b_in will be (pool4_channels + conv5b_channels) = 512 + 512 = 1024
        self.donv5b    = nn.Conv2d(1024, 512, 3, padding = 1)
        self.dbn5b     = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.donv5a    = nn.Conv2d(512, 512, 3, padding = 1)
        self.dbn5a     = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.unpool3   = nn.MaxUnpool2d(2, 2)
        # donv4b_in will be (donv5a_channels + conv4b_channels) = 512 + 512 = 1024
        self.donv4b    = nn.Conv2d(1024, 512, 3, padding = 1)
        self.dbn4b     = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.donv4a    = nn.Conv2d(512, 256, 3, padding = 1)
        self.dbn4a     = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.unpool2   = nn.MaxUnpool2d(2, 2)
        # donv3b_in will be (donv4a_channels + conv3b_channels) = 256 + 256 = 512
        self.donv3b    = nn.Conv2d(512, 256, 3, padding = 1)
        self.dbn3b     = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.donv3a    = nn.Conv2d(256, 128, 3, padding = 1)
        self.dbn3a     = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.unpool1   = nn.MaxUnpool2d(2, 2)
        # donv2_in is just unpool1_channels = 128
        self.donv2     = nn.Conv2d(128, 64, 3, padding = 1)
        self.dbn2      = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        # donv1_in will be (donv2_channels + conv1_channels) = 64 + 64 = 128
        self.donv1     = nn.Conv2d(128, 32, 3, padding = 1)
        self.dbn1      = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.output    = nn.Conv2d(32, out_ch, 1)

        # Activation tracking for memory analysis
        self.activation_shapes = []
        self.activation_memories = []

    def forward(self, x):
        # Clear previous activation tracking
        self.activation_shapes = []
        self.activation_memories = []
        
        # Track input
        self._track_activation(x, "input")
        
        # Encoder
        conv1_out = self.conv1(x)
        conv1_bn = self.bn1(conv1_out)
        conv1 = F.relu(conv1_bn, inplace=True)
        self._track_activation(conv1, "conv1") # Shape of conv1 after BN+ReLU for skip connection
        
        conv2_out = self.conv2(conv1)
        conv2_bn = self.bn2(conv2_out)
        conv2 = F.relu(conv2_bn, inplace=True)
        self._track_activation(conv2, "conv2") # Shape of conv2 after BN+ReLU for skip connection
        
        pool1, idxs1 = self.pool1(conv2)
        self._track_activation(pool1, "pool1")

        conv3a_out = self.conv3a(pool1)
        conv3a_bn = self.bn3a(conv3a_out)
        conv3a = F.relu(conv3a_bn, inplace=True)
        self._track_activation(conv3a, "conv3a")
        
        conv3b_out = self.conv3b(conv3a)
        conv3b_bn = self.bn3b(conv3b_out)
        conv3b = F.relu(conv3b_bn, inplace=True)
        self._track_activation(conv3b, "conv3b") # Shape of conv3b after BN+ReLU for skip connection
        
        pool2, idxs2 = self.pool2(conv3b)
        self._track_activation(pool2, "pool2")

        conv4a_out = self.conv4a(pool2)
        conv4a_bn = self.bn4a(conv4a_out)
        conv4a = F.relu(conv4a_bn, inplace=True)
        self._track_activation(conv4a, "conv4a")
        
        conv4b_out = self.conv4b(conv4a)
        conv4b_bn = self.bn4b(conv4b_out)
        conv4b = F.relu(conv4b_bn, inplace=True)
        self._track_activation(conv4b, "conv4b") # Shape of conv4b after BN+ReLU for skip connection
        
        pool3, idxs3 = self.pool3(conv4b)
        self._track_activation(pool3, "pool3")

        conv5a_out = self.conv5a(pool3)
        conv5a_bn = self.bn5a(conv5a_out)
        conv5a = F.relu(conv5a_bn, inplace=True)
        self._track_activation(conv5a, "conv5a")
        
        conv5b_out = self.conv5b(conv5a)
        conv5b_bn = self.bn5b(conv5b_out)
        conv5b = F.relu(conv5b_bn, inplace=True)
        self._track_activation(conv5b, "conv5b") # Shape of conv5b after BN+ReLU for skip connection
        
        pool4, idxs4 = self.pool4(conv5b)
        self._track_activation(pool4, "pool4")

        # Decoder
        unpool4 = self.unpool4(pool4, idxs4, output_size=conv5b.size())
        self._track_activation(unpool4, "unpool4")
        
        donv5b_in = torch.cat([unpool4, conv5b], 1)
        self._track_activation(donv5b_in, "donv5b_in") # For concatenation, memory will be sum of inputs + output
        
        donv5b_out = self.donv5b(donv5b_in)
        donv5b_bn = self.dbn5b(donv5b_out)
        donv5b = F.relu(donv5b_bn, inplace=True)
        self._track_activation(donv5b, "donv5b")
        
        donv5a_out = self.donv5a(donv5b)
        donv5a_bn = self.dbn5a(donv5a_out)
        donv5a = F.relu(donv5a_bn, inplace=True)
        self._track_activation(donv5a, "donv5a")

        unpool3 = self.unpool3(donv5a, idxs3, output_size=conv4b.size())
        self._track_activation(unpool3, "unpool3")
        
        donv4b_in = torch.cat([unpool3, conv4b], 1)
        self._track_activation(donv4b_in, "donv4b_in")
        
        donv4b_out = self.donv4b(donv4b_in)
        donv4b_bn = self.dbn4b(donv4b_out)
        donv4b = F.relu(donv4b_bn, inplace=True)
        self._track_activation(donv4b, "donv4b")
        
        donv4a_out = self.donv4a(donv4b)
        donv4a_bn = self.dbn4a(donv4a_out)
        donv4a = F.relu(donv4a_bn, inplace=True)
        self._track_activation(donv4a, "donv4a")

        unpool2 = self.unpool2(donv4a, idxs2, output_size=conv3b.size())
        self._track_activation(unpool2, "unpool2")
        
        donv3b_in = torch.cat([unpool2, conv3b], 1)
        self._track_activation(donv3b_in, "donv3b_in")
        
        donv3b_out = self.donv3b(donv3b_in)
        donv3b_bn = self.dbn3b(donv3b_out)
        donv3b = F.relu(donv3b_bn, inplace=True)
        self._track_activation(donv3b, "donv3b")
        
        donv3a_out = self.donv3a(donv3b)
        donv3a_bn = self.dbn3a(donv3a_out)
        donv3a = F.relu(donv3a_bn, inplace=True)
        self._track_activation(donv3a, "donv3a")

        unpool1 = self.unpool1(donv3a, idxs1, output_size=conv2.size())
        self._track_activation(unpool1, "unpool1")
        
        # Note: donv2 in forward receives unpool1, not a concatenation based on the original forward pass
        donv2_out = self.donv2(unpool1)
        donv2_bn = self.dbn2(donv2_out)
        donv2 = F.relu(donv2_bn, inplace=True)
        self._track_activation(donv2, "donv2")
        
        donv1_in = torch.cat([donv2, conv1], 1)
        self._track_activation(donv1_in, "donv1_in")
        
        donv1_out = self.donv1(donv1_in)
        donv1_bn = self.dbn1(donv1_out)
        donv1 = F.relu(donv1_bn, inplace=True)
        self._track_activation(donv1, "donv1")

        output = self.output(donv1)
        self._track_activation(output, "output")

        return output

    def _track_activation(self, tensor: torch.Tensor, name: str):
        """Track activation shapes and memory usage"""
        # Exclude pooling indices for memory tracking if they are part of the same tuple
        # For simplicity, assuming 'tensor' here is the actual data tensor
        shape = tuple(tensor.shape)
        # Use .numel() for total number of elements, .element_size() for bytes per element
        memory_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
        self.activation_shapes.append((name, shape))
        self.activation_memories.append((name, memory_mb))

    def get_total_activation_memory(self):
        """Get total activation memory in MB"""
        return sum(mem for _, mem in self.activation_memories)

# --- Performance Measurement Functions ---

def measure_latency_cuda(model: nn.Module, input_tensor: torch.Tensor, warmup_runs: int = 10, num_runs: int = 100):
    """Measure model inference latency using CUDA events with proper synchronization."""
    model.eval()
    
    # Warmup runs
    print(f"Performing {warmup_runs} warmup runs...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # Synchronize before timing
    torch.cuda.synchronize()
    
    # Timing runs
    print(f"Measuring latency over {num_runs} runs...")
    timings = []
    
    with torch.no_grad():
        for i in range(num_runs):
            # Create CUDA events
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Record start event
            start_event.record()
            
            # Forward pass
            _ = model(input_tensor)
            
            # Record end event
            end_event.record()
            
            # Synchronize and get elapsed time
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)  # Returns time in milliseconds
            timings.append(elapsed_time)
    
    return np.array(timings)

def measure_latency_cpu(model: nn.Module, input_tensor: torch.Tensor, warmup_runs: int = 7, num_runs: int = 25):
    """Measure model inference latency on CPU."""
    model.eval()
    
    # Warmup runs
    print(f"Performing {warmup_runs} warmup runs...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # Timing runs
    print(f"Measuring latency over {num_runs} runs...")
    timings = []
    
    with torch.no_grad():
        for i in range(num_runs):
            start_time = time.perf_counter()
            _ = model(input_tensor)
            end_time = time.perf_counter()
            elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
            timings.append(elapsed_time)
    
    return np.array(timings)

def total_sumnet_flops_params(input_res: Tuple[int, int], in_ch: int, out_ch: int):
    """Calculates total theoretical FLOPs and Parameters for the SUMNet_all_bn architecture."""
    total_params = 0
    total_flops = 0
    total_activation_size_mb = 0 # Peak activation memory (sum of all activations stored at any given time) would be complex to track,
                                 # this currently calculates the sum of all activation map sizes (cumulative, not peak).
                                 # For peak, one would need to simulate the graph and determine concurrent tensors.

    # Initial input shape: (Height, Width, Channels) - Batch dimension (N) is not included in HWC.
    # The calculations for H, W, C are based on one sample.
    curr_shape = (input_res[0], input_res[1], in_ch)
    total_activation_size_mb += get_activation_size_mb(curr_shape)

    # Dictionary to store shapes of encoder features after BN+ReLU for skip connections
    skip_shapes = {} # { 'conv1': (H,W,C), 'conv2': (H,W,C), ... }

    # --- Encoder ---

    # conv1 -> bn1 -> ReLU
    p, f, curr_shape = calculate_conv_flops_params(curr_shape, 64, 3, 1, 1)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape) # After Conv
    
    p, f, curr_shape = calculate_bn_flops_params(curr_shape)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape) # After BN
    
    p, f, curr_shape = calculate_relu_flops(curr_shape)
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape) # After ReLU
    skip_shapes['conv1'] = curr_shape # Save for skip connection

    # conv2 -> bn2 -> ReLU
    p, f, curr_shape = calculate_conv_flops_params(curr_shape, 128, 3, 1, 1)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_bn_flops_params(curr_shape)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_relu_flops(curr_shape)
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    skip_shapes['conv2'] = curr_shape

    # pool1
    p, f, curr_shape = calculate_pooling_flops_params(curr_shape, 2, 2)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape) # After Pool

    # conv3a -> bn3a -> ReLU
    p, f, curr_shape = calculate_conv_flops_params(curr_shape, 256, 3, 1, 1)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_bn_flops_params(curr_shape)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_relu_flops(curr_shape)
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)

    # conv3b -> bn3b -> ReLU
    p, f, curr_shape = calculate_conv_flops_params(curr_shape, 256, 3, 1, 1)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_bn_flops_params(curr_shape)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_relu_flops(curr_shape)
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    skip_shapes['conv3b'] = curr_shape

    # pool2
    p, f, curr_shape = calculate_pooling_flops_params(curr_shape, 2, 2)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)

    # conv4a -> bn4a -> ReLU
    p, f, curr_shape = calculate_conv_flops_params(curr_shape, 512, 3, 1, 1)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_bn_flops_params(curr_shape)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_relu_flops(curr_shape)
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)

    # conv4b -> bn4b -> ReLU
    p, f, curr_shape = calculate_conv_flops_params(curr_shape, 512, 3, 1, 1)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_bn_flops_params(curr_shape)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_relu_flops(curr_shape)
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    skip_shapes['conv4b'] = curr_shape

    # pool3
    p, f, curr_shape = calculate_pooling_flops_params(curr_shape, 2, 2)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)

    # conv5a -> bn5a -> ReLU
    p, f, curr_shape = calculate_conv_flops_params(curr_shape, 512, 3, 1, 1)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_bn_flops_params(curr_shape)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_relu_flops(curr_shape)
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)

    # conv5b -> bn5b -> ReLU
    p, f, curr_shape = calculate_conv_flops_params(curr_shape, 512, 3, 1, 1)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_bn_flops_params(curr_shape)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_relu_flops(curr_shape)
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    skip_shapes['conv5b'] = curr_shape

    # pool4
    p, f, curr_shape = calculate_pooling_flops_params(curr_shape, 2, 2)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)

    # --- Decoder ---

    # unpool4
    # unpool4 receives pool4_output_shape and output_size=conv5b.size()
    p, f, curr_shape = calculate_unpooling_flops_params(curr_shape, 2, 2, skip_shapes['conv5b'])
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)

    # concat (unpool4, conv5b)
    p, f, curr_shape = calculate_concat_flops_params([curr_shape, skip_shapes['conv5b']])
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape) # After Concat

    # donv5b -> dbn5b -> ReLU
    p, f, curr_shape = calculate_conv_flops_params(curr_shape, 512, 3, 1, 1)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_bn_flops_params(curr_shape)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_relu_flops(curr_shape)
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)

    # donv5a -> dbn5a -> ReLU
    p, f, curr_shape = calculate_conv_flops_params(curr_shape, 512, 3, 1, 1)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_bn_flops_params(curr_shape)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_relu_flops(curr_shape)
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)

    # unpool3
    p, f, curr_shape = calculate_unpooling_flops_params(curr_shape, 2, 2, skip_shapes['conv4b'])
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)

    # concat (unpool3, conv4b)
    p, f, curr_shape = calculate_concat_flops_params([curr_shape, skip_shapes['conv4b']])
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)

    # donv4b -> dbn4b -> ReLU
    p, f, curr_shape = calculate_conv_flops_params(curr_shape, 512, 3, 1, 1)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_bn_flops_params(curr_shape)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_relu_flops(curr_shape)
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)

    # donv4a -> dbn4a -> ReLU
    p, f, curr_shape = calculate_conv_flops_params(curr_shape, 256, 3, 1, 1)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_bn_flops_params(curr_shape)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_relu_flops(curr_shape)
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)

    # unpool2
    p, f, curr_shape = calculate_unpooling_flops_params(curr_shape, 2, 2, skip_shapes['conv3b'])
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)

    # concat (unpool2, conv3b)
    p, f, curr_shape = calculate_concat_flops_params([curr_shape, skip_shapes['conv3b']])
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)

    # donv3b -> dbn3b -> ReLU
    p, f, curr_shape = calculate_conv_flops_params(curr_shape, 256, 3, 1, 1)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_bn_flops_params(curr_shape)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_relu_flops(curr_shape)
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)

    # donv3a -> dbn3a -> ReLU
    p, f, curr_shape = calculate_conv_flops_params(curr_shape, 128, 3, 1, 1)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_bn_flops_params(curr_shape)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_relu_flops(curr_shape)
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)

    # unpool1
    p, f, curr_shape = calculate_unpooling_flops_params(curr_shape, 2, 2, skip_shapes['conv2']) # Should unpool to conv2.size()
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)

    # donv2 -> dbn2 -> ReLU (Note: This is not a concatenation in the forward pass)
    p, f, curr_shape = calculate_conv_flops_params(curr_shape, 64, 3, 1, 1)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_bn_flops_params(curr_shape)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_relu_flops(curr_shape)
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    # The output of donv2 is what is concatenated with conv1.
    donv2_output_shape = curr_shape 

    # concat (donv2, conv1)
    p, f, curr_shape = calculate_concat_flops_params([donv2_output_shape, skip_shapes['conv1']])
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)

    # donv1 -> dbn1 -> ReLU
    p, f, curr_shape = calculate_conv_flops_params(curr_shape, 32, 3, 1, 1)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_bn_flops_params(curr_shape)
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)
    
    p, f, curr_shape = calculate_relu_flops(curr_shape)
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)

    # output (final 1x1 conv)
    p, f, curr_shape = calculate_conv_flops_params(curr_shape, out_ch, 1, 1, 0) # 1x1 conv, no padding
    total_params += p
    total_flops += f
    total_activation_size_mb += get_activation_size_mb(curr_shape)

    return total_params, total_flops, total_activation_size_mb

def run_comprehensive_benchmark():
    """Run comprehensive benchmark including latency and memory analysis."""
    
    input_spatial_sizes = [
        (360, 640),    # H, W
        (720, 1280),   # H, W
        (760, 1360),   # H, W
        (900, 1600),   # H, W
        (1080, 1920),  # H, W
        (1152, 2048),  # H, W
        (1440, 2560),  # H, W
    ]
    
    in_channels = 3
    out_channels = 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("-" * 80)
    
    results = []
    
    for i, (height, width) in enumerate(input_spatial_sizes):
        print(f"\nBenchmark {i+1}: Input Resolution {height}×{width}")
        print("-" * 60)
        
        # Create model and move to device
        model = SUMNet_all_bn(in_channels, out_channels).to(device)
        model.eval()
        
        # Create input tensor
        input_tensor = torch.randn(1, in_channels, height, width).to(device)
        
        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Input tensor size: {input_tensor.numel() * input_tensor.element_size() / (1024*1024):.2f} MB")
        
        # Theoretical calculations
        theoretical_params, theoretical_flops, theoretical_activation_mb = total_sumnet_flops_params(
            (height, width), in_channels, out_channels
        )
        
        # Measure latency
        if device.type == 'cuda':
            latencies = measure_latency_cuda(model, input_tensor)
        else:
            latencies = measure_latency_cpu(model, input_tensor)
        
        # Get actual activation memory by running forward pass
        with torch.no_grad():
            _ = model(input_tensor)
            actual_activation_mb = model.get_total_activation_memory()
        
        # Calculate statistics
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        
        # Count actual parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Store results
        result = {
            'resolution': f"{height}×{width}",
            'height': height,
            'width': width,
            'theoretical_params': theoretical_params,
            'actual_params': total_params,
            'theoretical_flops': theoretical_flops,
            'theoretical_activation_mb': theoretical_activation_mb,
            'actual_activation_mb': actual_activation_mb,
            'mean_latency_ms': mean_latency,
            'std_latency_ms': std_latency,
            'min_latency_ms': min_latency,
            'max_latency_ms': max_latency,
            'input_size_mb': input_tensor.numel() * input_tensor.element_size() / (1024*1024)
        }
        results.append(result)
        
        # Print results
        print(f"\nResults:")
        print(f"  Parameters: {total_params/1e6:.2f}M (theoretical: {theoretical_params/1e6:.2f}M)")
        print(f"  FLOPs: {theoretical_flops/1e9:.2f}G")
        print(f"  Theoretical Activation Memory (Cumulative): {theoretical_activation_mb:.2f} MB")
        print(f"  Actual Activation Memory (Cumulative): {actual_activation_mb:.2f} MB")
        print(f"  Mean Latency: {mean_latency:.2f} ± {std_latency:.2f} ms")
        print(f"  Min/Max Latency: {min_latency:.2f}/{max_latency:.2f} ms")
        print(f"  Input Memory: {result['input_size_mb']:.2f} MB")
        
        # Memory efficiency - note that 'actual_activation_mb' is also cumulative, not peak.
        # So this 'efficiency' is more about consistency between theoretical and actual sum of sizes.
        efficiency = actual_activation_mb / theoretical_activation_mb if theoretical_activation_mb > 0 else 0
        print(f"  Memory Efficiency (Cumulative): {efficiency:.2f}")
        
    return results

def print_summary_table(results: List[Dict]):
    """Print a summary table of all results."""
    print("\n" + "="*120)
    print("COMPREHENSIVE PERFORMANCE SUMMARY")
    print("="*120)
    
    header = (f"{'Resolution':<12} {'Params(M)':<10} {'FLOPs(G)':<10} "
              f"{'Act.Mem(MB)':<12} {'Latency(ms)':<15} {'Input Mem(MB)':<12} {'Throughput':<12}")
    print(header)
    print("-" * 120)
    
    for result in results:
        throughput = 1000 / result['mean_latency_ms']  # images per second
        row = (f"{result['resolution']:<12} "
               f"{result['actual_params']/1e6:<10.2f} "
               f"{result['theoretical_flops']/1e9:<10.2f} "
               f"{result['actual_activation_mb']:<12.2f} "
               f"{result['mean_latency_ms']:<15.2f} "
               f"{result['input_size_mb']:<12.2f} "
               f"{throughput:<12.2f}")
        print(row)

# --- Main Execution ---
if __name__ == "__main__":
    print("SUMNet_all_bn Comprehensive Performance Analysis")
    print("="*80)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available, using CPU")
    
    # Run comprehensive benchmark
    results = run_comprehensive_benchmark()
    
    # Print summary
    print_summary_table(results)
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)