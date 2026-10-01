"""
efficiency.py - Parameter Count, FLOPs, Latency Benchmarking, and Memory Footprint Report
"""

import time
import torch
import numpy as np


def count_parameters(model):
    """Calculates total, trainable, and frozen parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return {
        "Total_Parameters": total_params,
        "Trainable_Parameters": trainable_params,
        "Frozen_Parameters": frozen_params,
        "Total_Parameters_M": total_params / 1e6,
    }


def estimate_flops(model, input_size=(1, 3, 224, 224), device="cpu"):
    """
    Estimates FLOPs / MACs for a forward pass.
    Uses analytical hook-based tracking for standard Conv/Linear layers.
    """
    model.eval()
    flops = 0
    hooks = []

    def conv_hook(module, input, output):
        nonlocal flops
        # output shape: (B, C_out, H_out, W_out)
        b, c_out, h_out, w_out = output.shape
        # kernel shape: (C_out, C_in/groups, K_h, K_w)
        c_in = module.in_channels
        k_h, k_w = module.kernel_size
        groups = module.groups
        macs = b * c_out * h_out * w_out * (c_in // groups) * k_h * k_w
        flops += 2 * macs  # 1 MAC = 2 FLOPs (mul + add)

    def linear_hook(module, input, output):
        nonlocal flops
        # input shape: (B, in_features)
        b, in_f = input[0].shape
        out_f = module.out_features
        macs = b * in_f * out_f
        flops += 2 * macs

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    dummy_input = torch.randn(*input_size, device=device)
    with torch.no_grad():
        try:
            _ = model(dummy_input)
        except Exception:
            _ = model.get_embedding(dummy_input)

    for h in hooks:
        h.remove()

    return {
        "FLOPs": flops,
        "GFLOPs": flops / 1e9,
        "GMACs": (flops / 2) / 1e9,
    }


def benchmark_latency(model, input_size=(1, 3, 224, 224), device="cpu", num_warmup=10, num_runs=50):
    """
    Measures empirical inference latency in milliseconds (mean and std).
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(*input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model.get_embedding(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model.get_embedding(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)  # ms

    return {
        "Batch_Size": input_size[0],
        "Mean_Latency_ms": float(np.mean(timings)),
        "Median_Latency_ms": float(np.median(timings)),
        "Std_Latency_ms": float(np.std(timings)),
        "Throughput_fps": float(input_size[0] / (np.mean(timings) / 1000.0)),
    }


def compute_memory_footprint(embedding_dim=512, num_items=(1000, 10000, 100000, 1000000)):
    """Computes RAM/disk footprint for gallery index of embeddings."""
    bytes_per_float = 4  # float32
    results = {}
    for n in num_items:
        total_bytes = n * embedding_dim * bytes_per_float
        results[f"{n:,}_items"] = {
            "Bytes": total_bytes,
            "KB": total_bytes / 1024,
            "MB": total_bytes / (1024 ** 2),
            "GB": total_bytes / (1024 ** 3),
        }
    return results


def generate_full_efficiency_report(model, device="cpu"):
    """Generates a complete consolidated efficiency report."""
    params = count_parameters(model)
    flops = estimate_flops(model, (1, 3, 224, 224), device=device)
    lat_b1 = benchmark_latency(model, (1, 3, 224, 224), device=device, num_warmup=5, num_runs=20)
    lat_b32 = benchmark_latency(model, (32, 3, 224, 224), device=device, num_warmup=5, num_runs=10)
    mem = compute_memory_footprint(model.embedding_dim)

    return {
        "Parameters": params,
        "Computational_Complexity": flops,
        "Latency_Single_Image_ms": lat_b1["Mean_Latency_ms"],
        "Throughput_Single_Image_fps": lat_b1["Throughput_fps"],
        "Latency_Batch32_ms": lat_b32["Mean_Latency_ms"],
        "Throughput_Batch32_fps": lat_b32["Throughput_fps"],
        "Embedding_Dimension": model.embedding_dim,
        "Embedding_Size_Bytes": model.embedding_dim * 4,
        "Memory_Footprint": mem,
    }
