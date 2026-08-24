#include "gpu_tensor.h"
#include "kernels.h"
#include "layers/dense.h"
#include "tensor.h"

#include <gtest/gtest.h>

#include <vector>

using cnn::dense_forward;

// ============================================================================
// Helper: run Dense on both CPU and GPU, compare results
// ============================================================================

static void verify_dense_gpu_vs_cpu(
    const Tensor& input, const Tensor& weights, const Tensor& bias,
    float atol = 1e-4f
) {
    Tensor cpu_output = dense_forward(input, weights, bias);

    GpuTensor gpu_input(input);
    GpuTensor gpu_weights(weights);
    GpuTensor gpu_bias(bias);
    GpuTensor gpu_output = dense_gpu(gpu_input, gpu_weights, gpu_bias);
    Tensor gpu_result = gpu_output.download();

    EXPECT_EQ(cpu_output.shape(), gpu_result.shape())
        << "Shape mismatch between CPU and GPU output";

    EXPECT_TRUE(cpu_output.allclose(gpu_result, atol))
        << "GPU output differs from CPU reference. Max diff: "
        << cpu_output.max_diff(gpu_result);
}

// ============================================================================
// Basic Correctness
// ============================================================================

TEST(DenseGPU, Simple3to2) {
    Tensor input({1, 3}, {1.0f, 2.0f, 3.0f});
    Tensor weights({2, 3}, {1.0f, 0.0f, 0.0f,
                             0.0f, 1.0f, 0.0f});
    Tensor bias({2}, {0.5f, -0.5f});

    verify_dense_gpu_vs_cpu(input, weights, bias);
}

TEST(DenseGPU, IdentityWeights) {
    Tensor input({1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor weights({4, 4}, {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1});
    Tensor bias = Tensor::zeros({4});

    verify_dense_gpu_vs_cpu(input, weights, bias);
}

TEST(DenseGPU, WithBias) {
    Tensor input({1, 3}, {1.0f, 1.0f, 1.0f});
    Tensor weights = Tensor::zeros({2, 3});
    Tensor bias({2}, {5.0f, -3.0f});

    // Output should just be bias
    verify_dense_gpu_vs_cpu(input, weights, bias);
}

// ============================================================================
// Architecture-Relevant Sizes
// ============================================================================

TEST(DenseGPU, Dense1_784to120) {
    // Dense1: flatten output (1, 784) → (1, 120)
    Tensor input = Tensor::rand({1, 784}, 42);
    Tensor weights = Tensor::rand({120, 784}, 43);
    Tensor bias = Tensor::rand({120}, 44);

    verify_dense_gpu_vs_cpu(input, weights, bias);
}

TEST(DenseGPU, Dense2_120to10) {
    // Dense2: (1, 120) → (1, 10)
    Tensor input = Tensor::rand({1, 120}, 42);
    Tensor weights = Tensor::rand({10, 120}, 43);
    Tensor bias = Tensor::rand({10}, 44);

    verify_dense_gpu_vs_cpu(input, weights, bias);
}

// ============================================================================
// Batch Tests
// ============================================================================

TEST(DenseGPU, BatchOf4_Dense1) {
    Tensor input = Tensor::rand({4, 784}, 42);
    Tensor weights = Tensor::rand({120, 784}, 43);
    Tensor bias = Tensor::rand({120}, 44);

    verify_dense_gpu_vs_cpu(input, weights, bias);
}

TEST(DenseGPU, BatchOf8_Dense2) {
    Tensor input = Tensor::rand({8, 120}, 42);
    Tensor weights = Tensor::rand({10, 120}, 43);
    Tensor bias = Tensor::rand({10}, 44);

    verify_dense_gpu_vs_cpu(input, weights, bias);
}

TEST(DenseGPU, BatchOf16) {
    Tensor input = Tensor::rand({16, 784}, 42);
    Tensor weights = Tensor::rand({120, 784}, 43);
    Tensor bias = Tensor::rand({120}, 44);

    verify_dense_gpu_vs_cpu(input, weights, bias);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(DenseGPU, AllZeroInput) {
    Tensor input = Tensor::zeros({1, 50});
    Tensor weights = Tensor::rand({10, 50}, 42);
    Tensor bias({10});
    for (int i = 0; i < 10; ++i) bias[i] = static_cast<float>(i);

    // Output should be bias only
    verify_dense_gpu_vs_cpu(input, weights, bias);
}

TEST(DenseGPU, AllZeroWeights) {
    Tensor input = Tensor::rand({1, 50}, 42);
    Tensor weights = Tensor::zeros({10, 50});
    Tensor bias({10});
    for (int i = 0; i < 10; ++i) bias[i] = static_cast<float>(i) * 0.1f;

    verify_dense_gpu_vs_cpu(input, weights, bias);
}

TEST(DenseGPU, SingleOutputFeature) {
    Tensor input = Tensor::rand({1, 100}, 42);
    Tensor weights = Tensor::rand({1, 100}, 43);
    Tensor bias({1}, {0.0f});

    verify_dense_gpu_vs_cpu(input, weights, bias);
}

TEST(DenseGPU, LargeInFeatures) {
    // Stress test: large in_features
    Tensor input = Tensor::rand({1, 2048}, 42);
    Tensor weights = Tensor::rand({64, 2048}, 43);
    Tensor bias = Tensor::rand({64}, 44);

    verify_dense_gpu_vs_cpu(input, weights, bias, 1e-3f);
}
