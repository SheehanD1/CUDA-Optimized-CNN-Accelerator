#include "gpu_model.h"

#include "kernels.h"

// ============================================================================
// GpuModel — Upload weights once, reuse across inference calls
// ============================================================================

GpuModel::GpuModel(const Model& model)
    : conv1_weights(model.conv1_weights),
      conv1_bias(model.conv1_bias),
      conv2_weights(model.conv2_weights),
      conv2_bias(model.conv2_bias),
      dense1_weights(model.dense1_weights),
      dense1_bias(model.dense1_bias),
      dense2_weights(model.dense2_weights),
      dense2_bias(model.dense2_bias) {
    // All 8 weight tensors are now resident on GPU
}

Tensor GpuModel::inference(const Tensor& input) const {
    // ========================================================================
    // Upload only the input (weights are already on GPU)
    // ========================================================================

    GpuTensor x(input);

    // ========================================================================
    // Block 1: Conv1 → ReLU → MaxPool
    // ========================================================================

    x = conv2d_tiled_gpu(x, conv1_weights, conv1_bias, 1, 1);
    x = relu_gpu(x);
    x = maxpool2d_gpu(x, 2);

    // ========================================================================
    // Block 2: Conv2 → ReLU → MaxPool
    // ========================================================================

    x = conv2d_tiled_gpu(x, conv2_weights, conv2_bias, 1, 1);
    x = relu_gpu(x);
    x = maxpool2d_gpu(x, 2);

    // ========================================================================
    // Flatten: (N, 16, 7, 7) → (N, 784)
    // ========================================================================

    int batch_size = x.dim(0);
    int flat_size = x.num_elements() / batch_size;
    x.reshape({batch_size, flat_size});

    // ========================================================================
    // Dense Block: Dense1 → ReLU → Dense2
    // ========================================================================

    x = dense_gpu(x, dense1_weights, dense1_bias);
    x = relu_gpu(x);
    x = dense_gpu(x, dense2_weights, dense2_bias);

    // ========================================================================
    // Softmax
    // ========================================================================

    x = softmax_gpu(x);

    // ========================================================================
    // Download result
    // ========================================================================

    return x.download();
}

std::vector<int> GpuModel::predict(const Tensor& input) const {
    Tensor probabilities = inference(input);
    return probabilities.argmax_per_row();
}
