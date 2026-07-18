#!/usr/bin/env python3
"""Generate Conv2D reference values for C++ unit test validation.

Uses PyTorch (or falls back to NumPy) to compute ground-truth convolution
outputs. The generated values are hardcoded into tests/test_conv2d.cpp
to validate the C++ naive implementation against a known-correct framework.

Usage:
    python scripts/generate_conv2d_reference.py
"""

import numpy as np


def numpy_conv2d(input_data, kernel_data, bias_data,
                 input_shape, kernel_shape, stride=1, padding=0):
    """Pure NumPy implementation of 2D convolution (NCHW layout).

    This serves as an independent reference implementation.
    """
    N, C_in, H, W = input_shape
    C_out, C_in_k, kH, kW = kernel_shape

    assert C_in == C_in_k, "Input channels must match kernel input channels"
    assert len(bias_data) == C_out, "Bias size must match output channels"

    # Compute output dimensions
    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1

    # Reshape flat arrays to NCHW
    inp = np.array(input_data, dtype=np.float32).reshape(input_shape)
    kern = np.array(kernel_data, dtype=np.float32).reshape(kernel_shape)
    bias = np.array(bias_data, dtype=np.float32)

    # Pad input if needed
    if padding > 0:
        inp = np.pad(inp, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                     mode='constant', constant_values=0)

    # Compute convolution
    output = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
    for n in range(N):
        for oc in range(C_out):
            for oh in range(H_out):
                for ow in range(W_out):
                    val = bias[oc]
                    for ic in range(C_in):
                        for kh in range(kH):
                            for kw in range(kW):
                                ih = oh * stride + kh
                                iw = ow * stride + kw
                                val += inp[n, ic, ih, iw] * kern[oc, ic, kh, kw]
                    output[n, oc, oh, ow] = val

    return output


def format_cpp_array(values, name="expected", per_line=6):
    """Format numpy array as C++ vector initializer."""
    flat = values.flatten()
    lines = []
    lines.append(f"    // Shape: {list(values.shape)}, {len(flat)} elements")
    lines.append(f"    std::vector<float> {name} = {{")
    for i in range(0, len(flat), per_line):
        chunk = flat[i:i + per_line]
        formatted = ", ".join(f"{v:.6f}f" for v in chunk)
        suffix = "," if i + per_line < len(flat) else ""
        lines.append(f"        {formatted}{suffix}")
    lines.append("    };")
    return "\n".join(lines)


def generate_test_case_1():
    """Test case 1: 2-channel input, 2 output filters, 3x3 kernel, padding=1."""
    print("=" * 70)
    print("Test Case 1: Multi-channel conv with padding")
    print("  Input:  (1, 2, 4, 4)")
    print("  Kernel: (2, 2, 3, 3)")
    print("  Bias:   (2)")
    print("  Stride: 1, Padding: 1")
    print("=" * 70)

    np.random.seed(42)
    input_shape = (1, 2, 4, 4)
    kernel_shape = (2, 2, 3, 3)

    input_data = np.random.uniform(-1, 1, size=np.prod(input_shape)).astype(np.float32)
    kernel_data = np.random.uniform(-1, 1, size=np.prod(kernel_shape)).astype(np.float32)
    bias_data = np.array([0.1, -0.2], dtype=np.float32)

    output = numpy_conv2d(input_data.tolist(), kernel_data.tolist(), bias_data.tolist(),
                          input_shape, kernel_shape, stride=1, padding=1)

    print(f"\nOutput shape: {output.shape}")
    print(f"\n// --- C++ test data ---")
    print(format_cpp_array(input_data.reshape(input_shape), "input_data"))
    print(format_cpp_array(kernel_data.reshape(kernel_shape), "kernel_data"))
    print(f"    std::vector<float> bias_data = {{{bias_data[0]:.6f}f, {bias_data[1]:.6f}f}};")
    print(format_cpp_array(output, "expected"))
    print()

    return input_data, kernel_data, bias_data, output


def generate_test_case_2():
    """Test case 2: Network-like dims (1, 1, 8, 8) -> (4, 1, 3, 3), stride=1, pad=0."""
    print("=" * 70)
    print("Test Case 2: Single-channel to 4 filters, no padding")
    print("  Input:  (1, 1, 8, 8)")
    print("  Kernel: (4, 1, 3, 3)")
    print("  Bias:   (4)")
    print("  Stride: 1, Padding: 0")
    print("=" * 70)

    np.random.seed(123)
    input_shape = (1, 1, 8, 8)
    kernel_shape = (4, 1, 3, 3)

    input_data = np.random.uniform(0, 1, size=np.prod(input_shape)).astype(np.float32)
    kernel_data = np.random.uniform(-0.5, 0.5, size=np.prod(kernel_shape)).astype(np.float32)
    bias_data = np.array([0.0, 0.1, -0.1, 0.05], dtype=np.float32)

    output = numpy_conv2d(input_data.tolist(), kernel_data.tolist(), bias_data.tolist(),
                          input_shape, kernel_shape, stride=1, padding=0)

    print(f"\nOutput shape: {output.shape}")
    print(f"\n// --- C++ test data ---")
    print(format_cpp_array(input_data.reshape(input_shape), "input_data"))
    print(format_cpp_array(kernel_data.reshape(kernel_shape), "kernel_data"))
    bias_str = ", ".join(f"{b:.6f}f" for b in bias_data)
    print(f"    std::vector<float> bias_data = {{{bias_str}}};")
    print(format_cpp_array(output, "expected"))
    print()

    return input_data, kernel_data, bias_data, output


if __name__ == "__main__":
    print("Generating Conv2D reference values for C++ tests...\n")
    generate_test_case_1()
    generate_test_case_2()
    print("Done. Copy the generated arrays into tests/test_conv2d.cpp")
