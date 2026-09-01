#include "gpu_tensor.h"
#include "kernels.h"
#include "layers/conv2d.h"
#include "tensor.h"

#include <gtest/gtest.h>

#include <vector>

using cnn::conv2d_forward;

// ============================================================================
// Tiled Conv2D Correctness Validation
// ============================================================================
//
// Verifies that conv2d_tiled_gpu() produces identical output to:
//   1. conv2d_gpu()  — naive GPU kernel (already validated against CPU)
//   2. conv2d_forward() — CPU reference
//
// ============================================================================

static void verify_tiled_vs_naive_and_cpu(
    const Tensor& input, const Tensor& weights, const Tensor& bias,
    int stride, int padding, float atol = 1e-4f
) {
    // CPU reference
    Tensor cpu_output = conv2d_forward(input, weights, bias, stride, padding);

    // GPU naive
    GpuTensor gpu_input(input);
    GpuTensor gpu_weights(weights);
    GpuTensor gpu_bias(bias);
    GpuTensor naive_output = conv2d_gpu(gpu_input, gpu_weights, gpu_bias, stride, padding);
    Tensor naive_result = naive_output.download();

    // GPU tiled
    GpuTensor tiled_output = conv2d_tiled_gpu(gpu_input, gpu_weights, gpu_bias, stride, padding);
    Tensor tiled_result = tiled_output.download();

    // Verify shapes match
    EXPECT_EQ(cpu_output.shape(), tiled_result.shape())
        << "Shape mismatch: tiled vs CPU";
    EXPECT_EQ(naive_result.shape(), tiled_result.shape())
        << "Shape mismatch: tiled vs naive";

    // Verify tiled matches CPU
    EXPECT_TRUE(cpu_output.allclose(tiled_result, atol))
        << "Tiled differs from CPU. Max diff: "
        << cpu_output.max_diff(tiled_result);

    // Verify tiled matches naive
    EXPECT_TRUE(naive_result.allclose(tiled_result, atol))
        << "Tiled differs from naive. Max diff: "
        << naive_result.max_diff(tiled_result);
}

// ============================================================================
// Basic Correctness
// ============================================================================

TEST(Conv2dTiled, SingleChannel_NoPadding) {
    Tensor input({1, 1, 4, 4});
    for (int i = 0; i < 16; ++i) input[i] = static_cast<float>(i + 1);
    Tensor weights({1, 1, 3, 3});
    for (int i = 0; i < 9; ++i) weights[i] = 1.0f;
    Tensor bias({1}, {0.0f});

    verify_tiled_vs_naive_and_cpu(input, weights, bias, 1, 0);
}

TEST(Conv2dTiled, SingleChannel_WithPadding) {
    Tensor input({1, 1, 4, 4});
    for (int i = 0; i < 16; ++i) input[i] = static_cast<float>(i + 1);
    Tensor weights({1, 1, 3, 3});
    for (int i = 0; i < 9; ++i) weights[i] = 1.0f;
    Tensor bias({1}, {0.0f});

    verify_tiled_vs_naive_and_cpu(input, weights, bias, 1, 1);
}

TEST(Conv2dTiled, WithBias) {
    Tensor input({1, 1, 3, 3});
    for (int i = 0; i < 9; ++i) input[i] = static_cast<float>(i);
    Tensor weights({2, 1, 2, 2});
    for (int i = 0; i < 8; ++i) weights[i] = 0.5f;
    Tensor bias({2}, {1.0f, -1.0f});

    verify_tiled_vs_naive_and_cpu(input, weights, bias, 1, 0);
}

// ============================================================================
// Multi-Channel
// ============================================================================

TEST(Conv2dTiled, MultiInputChannels) {
    Tensor input = Tensor::rand({1, 3, 5, 5}, 42);
    Tensor weights = Tensor::rand({2, 3, 3, 3}, 43);
    Tensor bias = Tensor::rand({2}, 44);

    verify_tiled_vs_naive_and_cpu(input, weights, bias, 1, 1);
}

TEST(Conv2dTiled, ManyOutputChannels) {
    Tensor input = Tensor::rand({1, 1, 8, 8}, 42);
    Tensor weights = Tensor::rand({8, 1, 3, 3}, 43);
    Tensor bias = Tensor::rand({8}, 44);

    verify_tiled_vs_naive_and_cpu(input, weights, bias, 1, 1);
}

// ============================================================================
// Architecture-Relevant Sizes
// ============================================================================

TEST(Conv2dTiled, Conv1_28x28) {
    // Conv1: (1, 1, 28, 28) → (1, 8, 28, 28) with pad=1
    Tensor input = Tensor::rand({1, 1, 28, 28}, 42);
    Tensor weights = Tensor::rand({8, 1, 3, 3}, 43);
    Tensor bias = Tensor::rand({8}, 44);

    verify_tiled_vs_naive_and_cpu(input, weights, bias, 1, 1);
}

TEST(Conv2dTiled, Conv2_14x14) {
    // Conv2: (1, 8, 14, 14) → (1, 16, 14, 14) with pad=1
    Tensor input = Tensor::rand({1, 8, 14, 14}, 42);
    Tensor weights = Tensor::rand({16, 8, 3, 3}, 43);
    Tensor bias = Tensor::rand({16}, 44);

    verify_tiled_vs_naive_and_cpu(input, weights, bias, 1, 1);
}

// ============================================================================
// Batch Tests
// ============================================================================

TEST(Conv2dTiled, BatchOf4_Conv1) {
    Tensor input = Tensor::rand({4, 1, 28, 28}, 42);
    Tensor weights = Tensor::rand({8, 1, 3, 3}, 43);
    Tensor bias = Tensor::rand({8}, 44);

    verify_tiled_vs_naive_and_cpu(input, weights, bias, 1, 1);
}

TEST(Conv2dTiled, BatchOf8_Conv2) {
    Tensor input = Tensor::rand({8, 8, 14, 14}, 42);
    Tensor weights = Tensor::rand({16, 8, 3, 3}, 43);
    Tensor bias = Tensor::rand({16}, 44);

    verify_tiled_vs_naive_and_cpu(input, weights, bias, 1, 1);
}

// ============================================================================
// Stride Tests
// ============================================================================

TEST(Conv2dTiled, Stride2) {
    Tensor input = Tensor::rand({1, 1, 8, 8}, 42);
    Tensor weights = Tensor::rand({4, 1, 3, 3}, 43);
    Tensor bias = Tensor::rand({4}, 44);

    verify_tiled_vs_naive_and_cpu(input, weights, bias, 2, 1);
}

// ============================================================================
// Non-Tile-Aligned Sizes (edge case for tiling)
// ============================================================================

TEST(Conv2dTiled, NonAligned_7x7) {
    // 7x7 is not a multiple of TILE_W=16, tests boundary handling
    Tensor input = Tensor::rand({1, 8, 7, 7}, 42);
    Tensor weights = Tensor::rand({16, 8, 3, 3}, 43);
    Tensor bias = Tensor::rand({16}, 44);

    verify_tiled_vs_naive_and_cpu(input, weights, bias, 1, 1);
}

TEST(Conv2dTiled, NonAligned_13x13) {
    // 13x13 output, not aligned to 16
    Tensor input = Tensor::rand({1, 4, 13, 13}, 42);
    Tensor weights = Tensor::rand({8, 4, 3, 3}, 43);
    Tensor bias = Tensor::rand({8}, 44);

    verify_tiled_vs_naive_and_cpu(input, weights, bias, 1, 1);
}

TEST(Conv2dTiled, SmallInput_3x3) {
    // Very small: output is 3x3 (smaller than tile size)
    Tensor input = Tensor::rand({1, 1, 3, 3}, 42);
    Tensor weights = Tensor::rand({2, 1, 3, 3}, 43);
    Tensor bias = Tensor::rand({2}, 44);

    verify_tiled_vs_naive_and_cpu(input, weights, bias, 1, 1);
}

// ============================================================================
// 1x1 Convolution
// ============================================================================

TEST(Conv2dTiled, Conv1x1) {
    Tensor input = Tensor::rand({1, 3, 8, 8}, 42);
    Tensor weights = Tensor::rand({4, 3, 1, 1}, 43);
    Tensor bias = Tensor::rand({4}, 44);

    verify_tiled_vs_naive_and_cpu(input, weights, bias, 1, 0);
}
