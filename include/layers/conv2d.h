#pragma once

// ============================================================================
// Conv2D — 2D Convolution Layer
// ============================================================================
// Performs 2D convolution on NCHW tensors with support for stride and padding.
// Used for spatial feature extraction in the CNN pipeline.
//
// CPU: Naive 7-loop implementation (batch, out_ch, out_h, out_w, in_ch, kh, kw)
// GPU: Custom CUDA kernels (naive + shared-memory tiled variants)
// ============================================================================

#include "tensor.h"

#include <string>

namespace cnn {

// ============================================================================
// Padding Utilities
// ============================================================================

/// Create a zero-padded copy of an NCHW input tensor.
/// Adds `padding` zeros to all four sides of the spatial dimensions (H, W).
/// Input shape: (N, C, H, W) -> Output shape: (N, C, H + 2*padding, W + 2*padding)
Tensor pad_tensor(const Tensor& input, int padding);

/// Compute the padding value needed for "same" output size.
/// For "same" convolution: output spatial dims == input spatial dims (when stride=1).
/// Formula: padding = (kernel_size - 1) / 2
/// Note: Only supports odd kernel sizes for symmetric padding.
int compute_same_padding(int kernel_size);

// ============================================================================
// Conv2D Forward Pass
// ============================================================================

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

/// Convenience overload with string-based padding mode.
///
/// @param padding_mode "valid" (no padding) or "same" (preserve spatial dims at stride=1)
///
/// Example:
///   conv2d_forward(input, kernel, bias, 1, "same")
///   // Automatically computes padding = (kH - 1) / 2
///
Tensor conv2d_forward(const Tensor& input, const Tensor& kernel, const Tensor& bias,
                      int stride, const std::string& padding_mode);

}  // namespace cnn
