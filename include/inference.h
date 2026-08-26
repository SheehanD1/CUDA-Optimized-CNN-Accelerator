#pragma once

// ============================================================================
// CPU Inference — Full Forward Pass Through All Layers
// ============================================================================
//
// Chains all layers in sequence to produce class probabilities from raw input:
//
//   Input (N, 1, 28, 28)
//     → Conv1(1→8, 3×3, pad=1) → ReLU → MaxPool(2×2)    → (N, 8, 14, 14)
//     → Conv2(8→16, 3×3, pad=1) → ReLU → MaxPool(2×2)   → (N, 16, 7, 7)
//     → Flatten                                           → (N, 784)
//     → Dense1(784→120) → ReLU                            → (N, 120)
//     → Dense2(120→10)                                    → (N, 10)
//     → Softmax                                           → (N, 10)
//   Output: probability distribution over 10 digit classes
//
// ============================================================================

#include "model.h"
#include "tensor.h"

/// Run full CPU inference pipeline.
///
/// @param model  Trained model with all weights and biases
/// @param input  Input tensor of shape (N, 1, 28, 28), values in [0, 1]
/// @return Output tensor of shape (N, 10) — softmax probabilities
///
Tensor cpu_inference(const Model& model, const Tensor& input);

/// Run CPU inference and return the predicted class for each sample.
///
/// @param model  Trained model with all weights and biases
/// @param input  Input tensor of shape (N, 1, 28, 28)
/// @return Vector of predicted class indices (0–9), one per sample
///
std::vector<int> cpu_predict(const Model& model, const Tensor& input);

// ============================================================================
// GPU Inference — Full Forward Pass Using CUDA Kernels
// ============================================================================
//
// Same architecture as CPU, but data stays on GPU between kernel launches.
// Only two H→D transfers at start (input + model weights) and one D→H
// at end (probabilities).
//
// ============================================================================

/// Run full GPU inference pipeline.
///
/// @param model  Trained model with all weights and biases (CPU)
/// @param input  Input tensor of shape (N, 1, 28, 28), values in [0, 1] (CPU)
/// @return Output tensor of shape (N, 10) — softmax probabilities (CPU)
///
Tensor gpu_inference(const Model& model, const Tensor& input);

/// Run GPU inference and return the predicted class for each sample.
///
/// @param model  Trained model with all weights and biases (CPU)
/// @param input  Input tensor of shape (N, 1, 28, 28) (CPU)
/// @return Vector of predicted class indices (0–9), one per sample
///
std::vector<int> gpu_predict(const Model& model, const Tensor& input);
