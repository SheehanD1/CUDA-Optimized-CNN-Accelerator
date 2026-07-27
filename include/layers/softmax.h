#pragma once

// ============================================================================
// Softmax — Probability Normalization
// ============================================================================
// Converts raw logits from the final Dense layer into a probability
// distribution over classes. Uses the numerically stable formulation:
//
//   softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
//
// Applied independently per sample in the batch.
//
// CPU: Loop per sample with max subtraction for numerical stability
// GPU: Parallel reduction for max and sum, then element-wise normalize
// ============================================================================

#include "tensor.h"

namespace cnn {

/// Apply softmax along the last dimension (classes) for each sample.
///
/// @param input Logits tensor of shape (N, num_classes)
/// @return Output tensor of shape (N, num_classes) with values in [0, 1]
///         that sum to 1.0 for each sample.
///
/// Uses the max-subtraction trick for numerical stability.
///
Tensor softmax_forward(const Tensor& input);

}  // namespace cnn
