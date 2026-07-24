#pragma once

// ============================================================================
// Flatten — Reshape Layer for Conv→Dense Transition
// ============================================================================
// Reshapes multi-dimensional feature maps into a flat vector suitable
// for input to Dense (fully connected) layers.
//
// In our LeNet pipeline:
//   Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool → **Flatten** → Dense
//   (N, 16, 5, 5) → (N, 400)
//
// This is a thin wrapper around Tensor::flatten() providing a consistent
// layer-function API: conv2d_forward → relu_forward → maxpool2d_forward
//   → flatten → dense_forward
// ============================================================================

#include "tensor.h"

namespace cnn {

/// Flatten all spatial and channel dimensions, keeping the batch dimension.
/// Input shape: (N, C, H, W) or (N, D1, D2, ...) → Output shape: (N, C*H*W)
///
/// For single samples (1D input), returns shape (1, num_elements).
///
/// @param input Tensor of any shape (>= 1D)
/// @return 2D tensor of shape (N, flattened_features)
///
Tensor flatten(const Tensor& input);

}  // namespace cnn
