#pragma once

// ============================================================================
// Dense — Fully Connected Layer
// ============================================================================
// Computes output = input * weights^T + bias (matrix-vector multiply per sample).
// Used after the convolutional feature extractor to map flattened features
// to class logits.
//
// For our LeNet architecture:
//   Dense(16*5*5=400, 120) → ReLU → Dense(120, 10) → Softmax
//
// CPU: Naive triple-nested loop (batch, out_features, in_features)
// GPU: Can leverage cuBLAS or custom tiled matrix multiply kernel
// ============================================================================

#include "tensor.h"

namespace cnn {

/// Compute fully connected (dense) layer: output = input * weights^T + bias
///
/// @param input   Input tensor of shape (N, in_features)
/// @param weights Weight matrix of shape (out_features, in_features)
/// @param bias    Bias vector of shape (out_features)
/// @return Output tensor of shape (N, out_features)
///
/// Each output sample: out[n] = weights * in[n] + bias
///
Tensor dense_forward(const Tensor& input, const Tensor& weights, const Tensor& bias);

}  // namespace cnn
