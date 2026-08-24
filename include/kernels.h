#pragma once

// ============================================================================
// GPU Kernel Launch Wrappers — CUDA Kernel API
// ============================================================================
//
// High-level wrappers around CUDA kernel launches. Each function:
//   1. Extracts dimensions from input GpuTensors
//   2. Computes grid/block launch configuration
//   3. Launches the kernel
//   4. Returns the output GpuTensor
//
// These mirror the CPU layer API but operate on GpuTensors.
//
// ============================================================================

#include "gpu_tensor.h"

// ============================================================================
// Conv2D
// ============================================================================

/// GPU Conv2D forward pass (naive: one thread per output element).
///
/// @param input   Input tensor (N, C_in, H, W) on GPU
/// @param weights Kernel weights (C_out, C_in, KH, KW) on GPU
/// @param bias    Bias tensor (C_out) on GPU
/// @param stride  Convolution stride
/// @param padding Zero-padding amount
/// @return Output tensor (N, C_out, OH, OW) on GPU
///
GpuTensor conv2d_gpu(const GpuTensor& input, const GpuTensor& weights,
                      const GpuTensor& bias, int stride, int padding);

// ============================================================================
// ReLU
// ============================================================================

/// GPU ReLU activation: element-wise max(0, x).
/// @param input Any-shape tensor on GPU
/// @return Output tensor with same shape on GPU
GpuTensor relu_gpu(const GpuTensor& input);

// ============================================================================
// MaxPool2D
// ============================================================================

/// GPU MaxPool2D: downsample by taking max in pool_size×pool_size windows.
/// @param input   Input tensor (N, C, H, W) on GPU
/// @param pool_size Pooling window size (e.g., 2)
/// @param stride  Stride (default: pool_size for non-overlapping)
/// @return Output tensor (N, C, H/stride, W/stride) on GPU
GpuTensor maxpool2d_gpu(const GpuTensor& input, int pool_size, int stride = 0);

// ============================================================================
// Dense
// ============================================================================

/// GPU Dense (fully connected) forward pass: output = input * weights^T + bias.
/// @param input   Input tensor (N, in_features) on GPU
/// @param weights Weight matrix (out_features, in_features) on GPU
/// @param bias    Bias vector (out_features) on GPU
/// @return Output tensor (N, out_features) on GPU
GpuTensor dense_gpu(const GpuTensor& input, const GpuTensor& weights,
                     const GpuTensor& bias);

// ============================================================================
// Softmax
// ============================================================================

/// GPU Softmax: numerically stable per-row softmax with shared memory reductions.
/// @param input Input tensor (N, num_classes) on GPU
/// @return Output tensor (N, num_classes) on GPU, each row sums to 1.0
GpuTensor softmax_gpu(const GpuTensor& input);
