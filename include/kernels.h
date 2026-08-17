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
// ReLU (Commit 45)
// ============================================================================

// GpuTensor relu_gpu(const GpuTensor& input);

// ============================================================================
// MaxPool2D (Commit 47)
// ============================================================================

// GpuTensor maxpool2d_gpu(const GpuTensor& input, int pool_size, int stride = 0);

// ============================================================================
// Dense (Commit 49)
// ============================================================================

// GpuTensor dense_gpu(const GpuTensor& input, const GpuTensor& weights,
//                     const GpuTensor& bias);

// ============================================================================
// Softmax (Commit 51)
// ============================================================================

// GpuTensor softmax_gpu(const GpuTensor& input);
