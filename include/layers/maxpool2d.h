#pragma once

// ============================================================================
// MaxPool2D — 2D Max Pooling Layer
// ============================================================================
// Downsamples spatial dimensions by taking the maximum value in each
// pooling window. Reduces feature map size while preserving strongest
// activations.
//
// For our LeNet architecture: MaxPool(2x2, stride=2) halves H and W.
//
// CPU: 4 nested loops (batch, channel, pool_h, pool_w) + 2 inner loops
// GPU: One thread per output element, each scans its pooling window
// ============================================================================

#include "tensor.h"

namespace cnn {

/// Compute 2D max pooling on an NCHW tensor.
///
/// @param input     Input tensor of shape (N, C, H, W)
/// @param pool_size Size of the square pooling window (default: 2)
/// @param stride    Stride of the pooling window (default: same as pool_size)
/// @return Output tensor of shape (N, C, H_out, W_out)
///
/// Output dimensions:
///   H_out = (H - pool_size) / stride + 1
///   W_out = (W - pool_size) / stride + 1
///
/// Note: When stride == 0, it is set to pool_size (standard non-overlapping pooling).
///
Tensor maxpool2d_forward(const Tensor& input, int pool_size = 2, int stride = 0);

}  // namespace cnn
