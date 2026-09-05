#pragma once

// ============================================================================
// GpuModel — Cached Model Weights on GPU
// ============================================================================
//
// Uploads model weights to GPU memory once, then reuses them across multiple
// inference calls. This eliminates the overhead of uploading 8 tensors
// (~380 KB for our LeNet architecture) on every inference call.
//
// Usage:
//   Model model;
//   model.load("weights.bin");
//   GpuModel gpu_model(model);  // Upload weights once
//
//   // Run inference many times without re-uploading weights
//   Tensor result1 = gpu_model.inference(input1);
//   Tensor result2 = gpu_model.inference(input2);
//
// ============================================================================

#include "gpu_tensor.h"
#include "model.h"
#include "tensor.h"

#include <vector>

class GpuModel {
public:
    /// Upload all model weights to GPU. Weights are cached for reuse.
    explicit GpuModel(const Model& model);

    /// Run full GPU inference pipeline using cached weights.
    /// @param input  Input tensor (N, 1, 28, 28) on CPU
    /// @return Softmax probabilities (N, 10) on CPU
    Tensor inference(const Tensor& input) const;

    /// Run GPU inference and return predicted class indices.
    /// @param input  Input tensor (N, 1, 28, 28) on CPU
    /// @return Vector of predicted classes (0-9), one per sample
    std::vector<int> predict(const Tensor& input) const;

    // Cached GPU weights (public for direct kernel access if needed)
    GpuTensor conv1_weights;
    GpuTensor conv1_bias;
    GpuTensor conv2_weights;
    GpuTensor conv2_bias;
    GpuTensor dense1_weights;
    GpuTensor dense1_bias;
    GpuTensor dense2_weights;
    GpuTensor dense2_bias;
};
