#pragma once

// ============================================================================
// ReLU — Rectified Linear Unit Activation
// ============================================================================
// Element-wise activation: output = max(0, input)
// Applied after convolution and dense layers to introduce non-linearity.
//
// CPU: Simple loop with std::max
// GPU: Embarrassingly parallel — one thread per element
// ============================================================================

#include "tensor.h"

namespace cnn {

/// Apply ReLU activation element-wise: output[i] = max(0, input[i])
///
/// @param input Tensor of any shape
/// @return Output tensor of the same shape with all negative values zeroed
///
Tensor relu_forward(const Tensor& input);

/// Apply ReLU activation in-place: input[i] = max(0, input[i])
/// Avoids allocation when the input tensor is no longer needed.
///
void relu_forward_inplace(Tensor& input);

}  // namespace cnn
