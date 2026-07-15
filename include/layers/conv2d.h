#pragma once

// ============================================================================
// Conv2D — 2D Convolution Layer
// ============================================================================
// Performs 2D convolution on NCHW tensors with support for stride and padding.
// Used for spatial feature extraction in the CNN pipeline.
//
// CPU: Naive 6-loop implementation (batch, out_ch, out_h, out_w, in_ch, kh, kw)
// GPU: Custom CUDA kernels (naive + shared-memory tiled variants)
// ============================================================================

#include "tensor.h"

namespace cnn {

/// Compute 2D convolution: output = input * kernel + bias
///
/// @param input   Input tensor of shape (N, C_in, H, W)
/// @param kernel  Convolution kernels of shape (C_out, C_in, kH, kW)
/// @param bias    Bias vector of shape (C_out)
/// @param stride  Stride for the convolution (default: 1)
/// @param padding Zero-padding added to both sides of input (default: 0)
/// @return Output tensor of shape (N, C_out, H_out, W_out)
///
/// Output dimensions:
///   H_out = (H + 2*padding - kH) / stride + 1
///   W_out = (W + 2*padding - kW) / stride + 1
///
Tensor conv2d_forward(const Tensor& input, const Tensor& kernel, const Tensor& bias,
                      int stride = 1, int padding = 0);

}  // namespace cnn
