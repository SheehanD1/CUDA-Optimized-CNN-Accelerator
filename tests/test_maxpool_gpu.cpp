#include "gpu_tensor.h"
#include "kernels.h"
#include "layers/maxpool2d.h"
#include "tensor.h"

#include <gtest/gtest.h>

#include <vector>

using cnn::maxpool2d_forward;

// ============================================================================
// Helper: run MaxPool2D on both CPU and GPU, compare results
// ============================================================================

static void verify_maxpool_gpu_vs_cpu(
    const Tensor& input, int pool_size, int stride = 0, float atol = 1e-6f
) {
    // CPU uses stride=pool_size when stride=0 is passed as default
    Tensor cpu_output = maxpool2d_forward(input, pool_size,
                                          (stride <= 0) ? pool_size : stride);

    GpuTensor gpu_input(input);
    GpuTensor gpu_output = maxpool2d_gpu(gpu_input, pool_size, stride);
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

TEST(MaxPoolGPU, Pool2x2_Simple) {
    Tensor input({1, 1, 4, 4});
    for (int i = 0; i < 16; ++i) input[i] = static_cast<float>(i + 1);
    verify_maxpool_gpu_vs_cpu(input, 2);
}

TEST(MaxPoolGPU, Pool2x2_SingleChannel) {
    Tensor input = Tensor::rand({1, 1, 6, 6}, 42);
    verify_maxpool_gpu_vs_cpu(input, 2);
}

TEST(MaxPoolGPU, Pool3x3) {
    Tensor input = Tensor::rand({1, 1, 9, 9}, 42);
    verify_maxpool_gpu_vs_cpu(input, 3);
}

// ============================================================================
// Multi-Channel Tests
// ============================================================================

TEST(MaxPoolGPU, MultiChannel) {
    Tensor input = Tensor::rand({1, 4, 8, 8}, 42);
    verify_maxpool_gpu_vs_cpu(input, 2);
}

TEST(MaxPoolGPU, ManyChannels) {
    Tensor input = Tensor::rand({1, 16, 8, 8}, 42);
    verify_maxpool_gpu_vs_cpu(input, 2);
}

// ============================================================================
// Architecture-Relevant Sizes
// ============================================================================

TEST(MaxPoolGPU, AfterConv1_28x28) {
    // After Conv1+ReLU: (1, 8, 28, 28) → (1, 8, 14, 14)
    Tensor input = Tensor::rand({1, 8, 28, 28}, 42);
    verify_maxpool_gpu_vs_cpu(input, 2);
}

TEST(MaxPoolGPU, AfterConv2_14x14) {
    // After Conv2+ReLU: (1, 16, 14, 14) → (1, 16, 7, 7)
    Tensor input = Tensor::rand({1, 16, 14, 14}, 42);
    verify_maxpool_gpu_vs_cpu(input, 2);
}

// ============================================================================
// Batch Tests
// ============================================================================

TEST(MaxPoolGPU, BatchOf4) {
    Tensor input = Tensor::rand({4, 8, 28, 28}, 42);
    verify_maxpool_gpu_vs_cpu(input, 2);
}

TEST(MaxPoolGPU, BatchOf8) {
    Tensor input = Tensor::rand({8, 16, 14, 14}, 42);
    verify_maxpool_gpu_vs_cpu(input, 2);
}

// ============================================================================
// Stride Tests
// ============================================================================

TEST(MaxPoolGPU, ExplicitStride) {
    // Explicit stride=2 (same as pool_size=2 default)
    Tensor input = Tensor::rand({1, 4, 8, 8}, 42);
    verify_maxpool_gpu_vs_cpu(input, 2, 2);
}

TEST(MaxPoolGPU, OverlappingStride1) {
    // Overlapping pools: pool=2, stride=1
    Tensor input = Tensor::rand({1, 2, 6, 6}, 42);
    verify_maxpool_gpu_vs_cpu(input, 2, 1);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(MaxPoolGPU, NegativeValues) {
    // Max of all negatives should still find the least negative
    Tensor input({1, 1, 4, 4});
    for (int i = 0; i < 16; ++i) {
        input[i] = -static_cast<float>(16 - i);  // -16, -15, ..., -1
    }
    verify_maxpool_gpu_vs_cpu(input, 2);
}

TEST(MaxPoolGPU, UniformValues) {
    // All same value — max should be that value
    Tensor input({1, 1, 4, 4});
    for (int i = 0; i < 16; ++i) input[i] = 42.0f;
    verify_maxpool_gpu_vs_cpu(input, 2);
}
