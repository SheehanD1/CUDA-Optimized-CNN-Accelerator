#include "inference.h"

#include "layers/conv2d.h"
#include "layers/dense.h"
#include "layers/flatten.h"
#include "layers/maxpool2d.h"
#include "layers/relu.h"
#include "layers/softmax.h"

using cnn::conv2d_forward;
using cnn::dense_forward;
using cnn::flatten;
using cnn::maxpool2d_forward;
using cnn::relu_forward;
using cnn::softmax_forward;

Tensor cpu_inference(const Model& model, const Tensor& input) {
    // ========================================================================
    // Block 1: Conv1 → ReLU → MaxPool
    // ========================================================================
    // Input:  (N, 1, 28, 28)
    // Conv1:  (N, 8, 28, 28)  — 3×3 kernel, padding=1 preserves spatial dims
    // ReLU:   (N, 8, 28, 28)
    // Pool:   (N, 8, 14, 14)  — 2×2 pool, stride 2

    Tensor x = conv2d_forward(input, model.conv1_weights, model.conv1_bias, 1, 1);
    x = relu_forward(x);
    x = maxpool2d_forward(x, 2);

    // ========================================================================
    // Block 2: Conv2 → ReLU → MaxPool
    // ========================================================================
    // Input:  (N, 8, 14, 14)
    // Conv2:  (N, 16, 14, 14)
    // ReLU:   (N, 16, 14, 14)
    // Pool:   (N, 16, 7, 7)

    x = conv2d_forward(x, model.conv2_weights, model.conv2_bias, 1, 1);
    x = relu_forward(x);
    x = maxpool2d_forward(x, 2);

    // ========================================================================
    // Flatten: (N, 16, 7, 7) → (N, 784)
    // ========================================================================

    x = flatten(x);

    // ========================================================================
    // Dense Block: Dense1 → ReLU → Dense2
    // ========================================================================
    // Dense1: (N, 784) → (N, 120)
    // ReLU:   (N, 120)
    // Dense2: (N, 120) → (N, 10)

    x = dense_forward(x, model.dense1_weights, model.dense1_bias);
    x = relu_forward(x);
    x = dense_forward(x, model.dense2_weights, model.dense2_bias);

    // ========================================================================
    // Softmax: (N, 10) → (N, 10) probabilities
    // ========================================================================

    x = softmax_forward(x);

    return x;
}

std::vector<int> cpu_predict(const Model& model, const Tensor& input) {
    Tensor probabilities = cpu_inference(model, input);
    return probabilities.argmax_per_row();
}
