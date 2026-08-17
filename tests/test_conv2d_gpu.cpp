#include "gpu_tensor.h"
#include "kernels.h"
#include "layers/conv2d.h"
#include "tensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using cnn::conv2d_forward;

// ============================================================================
// Helper: run conv2d on both CPU and GPU, compare results
// ============================================================================

static void verify_conv2d_gpu_vs_cpu(
    const Tensor& input, const Tensor& weights, const Tensor& bias,
    int stride, int padding, float atol = 1e-5f
) {
    // CPU reference
    Tensor cpu_output = conv2d_forward(input, weights, bias, stride, padding);

    // GPU computation
    GpuTensor gpu_input(input);
    GpuTensor gpu_weights(weights);
    GpuTensor gpu_bias(bias);
    GpuTensor gpu_output = conv2d_gpu(gpu_input, gpu_weights, gpu_bias, stride, padding);

    // Download and compare
    Tensor gpu_result = gpu_output.download();

    EXPECT_EQ(cpu_output.shape(), gpu_result.shape())
        << "Shape mismatch between CPU and GPU output";

    EXPECT_TRUE(cpu_output.allclose(gpu_result, atol))
        << "GPU output differs from CPU reference. Max diff: "
        << cpu_output.max_diff(gpu_result);
}

// ============================================================================
// Basic Correctness Tests
// ============================================================================

TEST(Conv2dGPU, SingleChannelNoPadding) {
    // 1 batch, 1 channel, 4x4 input, 1 output channel, 3x3 kernel
    Tensor input({1, 1, 4, 4});
    for (int i = 0; i < 16; ++i) input[i] = static_cast<float>(i + 1);

    Tensor weights({1, 1, 3, 3});
    for (int i = 0; i < 9; ++i) weights[i] = 1.0f;

    Tensor bias({1}, {0.0f});

    verify_conv2d_gpu_vs_cpu(input, weights, bias, 1, 0);
}

TEST(Conv2dGPU, SingleChannelWithPadding) {
    Tensor input({1, 1, 4, 4});
    for (int i = 0; i < 16; ++i) input[i] = static_cast<float>(i + 1);

    Tensor weights({1, 1, 3, 3});
    for (int i = 0; i < 9; ++i) weights[i] = 1.0f;

    Tensor bias({1}, {0.0f});

    // padding=1 preserves spatial dimensions
    verify_conv2d_gpu_vs_cpu(input, weights, bias, 1, 1);
}

TEST(Conv2dGPU, WithBias) {
    Tensor input({1, 1, 3, 3});
    for (int i = 0; i < 9; ++i) input[i] = static_cast<float>(i);

    Tensor weights({2, 1, 2, 2});
    for (int i = 0; i < 8; ++i) weights[i] = 0.5f;

    Tensor bias({2}, {1.0f, -1.0f});

    verify_conv2d_gpu_vs_cpu(input, weights, bias, 1, 0);
}

// ============================================================================
// Multi-Channel Tests
// ============================================================================

TEST(Conv2dGPU, MultiInputChannels) {
    // 3 input channels, 2 output channels
    Tensor input = Tensor::rand({1, 3, 5, 5}, 42);
    Tensor weights = Tensor::rand({2, 3, 3, 3}, 43);
    Tensor bias = Tensor::rand({2}, 44);

    verify_conv2d_gpu_vs_cpu(input, weights, bias, 1, 1);
}

TEST(Conv2dGPU, ManyOutputChannels) {
    // 1→8 channels, like our Conv1 layer
    Tensor input = Tensor::rand({1, 1, 8, 8}, 42);
    Tensor weights = Tensor::rand({8, 1, 3, 3}, 43);
    Tensor bias = Tensor::rand({8}, 44);

    verify_conv2d_gpu_vs_cpu(input, weights, bias, 1, 1);
}

// ============================================================================
// Architecture-Relevant Sizes
// ============================================================================

TEST(Conv2dGPU, Conv1_28x28) {
    // Conv1: (1, 1, 28, 28) → (1, 8, 28, 28) with pad=1
    Tensor input = Tensor::rand({1, 1, 28, 28}, 42);
    Tensor weights = Tensor::rand({8, 1, 3, 3}, 43);
    Tensor bias = Tensor::rand({8}, 44);

    verify_conv2d_gpu_vs_cpu(input, weights, bias, 1, 1);
}

TEST(Conv2dGPU, Conv2_14x14) {
    // Conv2: (1, 8, 14, 14) → (1, 16, 14, 14) with pad=1
    Tensor input = Tensor::rand({1, 8, 14, 14}, 42);
    Tensor weights = Tensor::rand({16, 8, 3, 3}, 43);
    Tensor bias = Tensor::rand({16}, 44);

    verify_conv2d_gpu_vs_cpu(input, weights, bias, 1, 1);
}

// ============================================================================
// Batch Tests
// ============================================================================

TEST(Conv2dGPU, BatchOf4) {
    Tensor input = Tensor::rand({4, 1, 28, 28}, 42);
    Tensor weights = Tensor::rand({8, 1, 3, 3}, 43);
    Tensor bias = Tensor::rand({8}, 44);

    verify_conv2d_gpu_vs_cpu(input, weights, bias, 1, 1);
}

TEST(Conv2dGPU, BatchOf8Conv2) {
    Tensor input = Tensor::rand({8, 8, 14, 14}, 42);
    Tensor weights = Tensor::rand({16, 8, 3, 3}, 43);
    Tensor bias = Tensor::rand({16}, 44);

    verify_conv2d_gpu_vs_cpu(input, weights, bias, 1, 1);
}

// ============================================================================
// Stride Tests
// ============================================================================

TEST(Conv2dGPU, Stride2) {
    Tensor input = Tensor::rand({1, 1, 8, 8}, 42);
    Tensor weights = Tensor::rand({4, 1, 3, 3}, 43);
    Tensor bias = Tensor::rand({4}, 44);

    verify_conv2d_gpu_vs_cpu(input, weights, bias, 2, 1);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(Conv2dGPU, AllZeroInput) {
    Tensor input = Tensor::zeros({1, 1, 5, 5});
    Tensor weights = Tensor::rand({2, 1, 3, 3}, 42);
    Tensor bias({2}, {3.0f, -1.0f});

    // Output should just be bias values
    verify_conv2d_gpu_vs_cpu(input, weights, bias, 1, 0);
}

TEST(Conv2dGPU, AllZeroWeights) {
    Tensor input = Tensor::rand({1, 1, 5, 5}, 42);
    Tensor weights = Tensor::zeros({2, 1, 3, 3});
    Tensor bias({2}, {5.0f, -2.0f});

    // Output should just be bias values
    verify_conv2d_gpu_vs_cpu(input, weights, bias, 1, 0);
}

TEST(Conv2dGPU, IdentityKernel1x1) {
    // 1x1 convolution: essentially a per-pixel channel transform
    Tensor input = Tensor::rand({1, 3, 5, 5}, 42);
    Tensor weights = Tensor::rand({4, 3, 1, 1}, 43);
    Tensor bias = Tensor::rand({4}, 44);

    verify_conv2d_gpu_vs_cpu(input, weights, bias, 1, 0);
}
