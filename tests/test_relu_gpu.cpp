#include "gpu_tensor.h"
#include "kernels.h"
#include "layers/relu.h"
#include "tensor.h"

#include <gtest/gtest.h>

#include <vector>

using cnn::relu_forward;

// ============================================================================
// Helper: run ReLU on both CPU and GPU, compare results
// ============================================================================

static void verify_relu_gpu_vs_cpu(const Tensor& input, float atol = 1e-6f) {
    Tensor cpu_output = relu_forward(input);

    GpuTensor gpu_input(input);
    GpuTensor gpu_output = relu_gpu(gpu_input);
    Tensor gpu_result = gpu_output.download();

    EXPECT_EQ(cpu_output.shape(), gpu_result.shape());
    EXPECT_TRUE(cpu_output.allclose(gpu_result, atol))
        << "Max diff: " << cpu_output.max_diff(gpu_result);
}

// ============================================================================
// Basic Correctness
// ============================================================================

TEST(ReluGPU, AllPositive) {
    Tensor input({6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    verify_relu_gpu_vs_cpu(input);
}

TEST(ReluGPU, AllNegative) {
    Tensor input({5}, {-1.0f, -2.0f, -3.0f, -0.5f, -100.0f});
    verify_relu_gpu_vs_cpu(input);
}

TEST(ReluGPU, Mixed) {
    Tensor input({6}, {-2.0f, 0.0f, 3.0f, -1.0f, 5.0f, -0.001f});
    verify_relu_gpu_vs_cpu(input);
}

TEST(ReluGPU, AllZeros) {
    Tensor input = Tensor::zeros({10});
    verify_relu_gpu_vs_cpu(input);
}

// ============================================================================
// Multi-Dimensional
// ============================================================================

TEST(ReluGPU, Tensor2D) {
    Tensor input({3, 4}, {-1, 2, -3, 4, 5, -6, 7, -8, 0, 1, -2, 3});
    verify_relu_gpu_vs_cpu(input);
}

TEST(ReluGPU, Tensor4D_NCHW) {
    Tensor input = Tensor::rand({2, 3, 4, 4}, 42);
    // Shift to have negative values: rand is [0,1], shift to [-0.5, 0.5]
    for (int i = 0; i < input.num_elements(); ++i) {
        input[i] -= 0.5f;
    }
    verify_relu_gpu_vs_cpu(input);
}

// ============================================================================
// Architecture-Relevant Sizes
// ============================================================================

TEST(ReluGPU, AfterConv1_Shape) {
    // After Conv1: (1, 8, 28, 28)
    Tensor input = Tensor::rand({1, 8, 28, 28}, 42);
    for (int i = 0; i < input.num_elements(); ++i) input[i] -= 0.5f;
    verify_relu_gpu_vs_cpu(input);
}

TEST(ReluGPU, AfterConv2_Shape) {
    // After Conv2: (1, 16, 14, 14)
    Tensor input = Tensor::rand({1, 16, 14, 14}, 42);
    for (int i = 0; i < input.num_elements(); ++i) input[i] -= 0.5f;
    verify_relu_gpu_vs_cpu(input);
}

TEST(ReluGPU, AfterDense1_Shape) {
    // After Dense1: (1, 120)
    Tensor input = Tensor::rand({1, 120}, 42);
    for (int i = 0; i < input.num_elements(); ++i) input[i] -= 0.5f;
    verify_relu_gpu_vs_cpu(input);
}

// ============================================================================
// Batch Tests
// ============================================================================

TEST(ReluGPU, BatchOf8) {
    Tensor input = Tensor::rand({8, 8, 28, 28}, 42);
    for (int i = 0; i < input.num_elements(); ++i) input[i] -= 0.5f;
    verify_relu_gpu_vs_cpu(input);
}

// ============================================================================
// Non-Power-of-2 Sizes
// ============================================================================

TEST(ReluGPU, OddSize) {
    // 1023 elements — not a multiple of block size (256)
    Tensor input = Tensor::rand({1023}, 42);
    for (int i = 0; i < input.num_elements(); ++i) input[i] -= 0.5f;
    verify_relu_gpu_vs_cpu(input);
}

TEST(ReluGPU, PrimeSize) {
    // 997 elements — prime number
    Tensor input = Tensor::rand({997}, 42);
    for (int i = 0; i < input.num_elements(); ++i) input[i] -= 0.5f;
    verify_relu_gpu_vs_cpu(input);
}

TEST(ReluGPU, SingleElement) {
    Tensor input({1}, {-5.0f});
    verify_relu_gpu_vs_cpu(input);
}
