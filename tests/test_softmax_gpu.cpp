#include "gpu_tensor.h"
#include "kernels.h"
#include "layers/softmax.h"
#include "tensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using cnn::softmax_forward;

// ============================================================================
// Helper: run Softmax on both CPU and GPU, compare results
// ============================================================================

static void verify_softmax_gpu_vs_cpu(const Tensor& input, float atol = 1e-5f) {
    Tensor cpu_output = softmax_forward(input);

    GpuTensor gpu_input(input);
    GpuTensor gpu_output = softmax_gpu(gpu_input);
    Tensor gpu_result = gpu_output.download();

    EXPECT_EQ(cpu_output.shape(), gpu_result.shape());

    EXPECT_TRUE(cpu_output.allclose(gpu_result, atol))
        << "GPU output differs from CPU reference. Max diff: "
        << cpu_output.max_diff(gpu_result);
}

// Helper: verify softmax properties (all in [0,1], rows sum to 1)
static void verify_softmax_properties(const Tensor& output, int batch_size,
                                       int num_classes) {
    for (int n = 0; n < batch_size; ++n) {
        float sum = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            float val = output.at(n, j);
            EXPECT_GE(val, 0.0f) << "Negative probability at (" << n << "," << j << ")";
            EXPECT_LE(val, 1.0f) << "Probability > 1 at (" << n << "," << j << ")";
            sum += val;
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f)
            << "Row " << n << " does not sum to 1.0";
    }
}

// ============================================================================
// Basic Correctness
// ============================================================================

TEST(SoftmaxGPU, SimpleValues) {
    Tensor input({1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
    verify_softmax_gpu_vs_cpu(input);
}

TEST(SoftmaxGPU, AllEqual) {
    // All equal inputs → uniform distribution (1/N each)
    Tensor input({1, 5}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    verify_softmax_gpu_vs_cpu(input);

    GpuTensor gpu_in(input);
    Tensor result = softmax_gpu(gpu_in).download();
    for (int j = 0; j < 5; ++j) {
        EXPECT_NEAR(result.at(0, j), 0.2f, 1e-5f);
    }
}

TEST(SoftmaxGPU, AllZeros) {
    Tensor input = Tensor::zeros({1, 10});
    verify_softmax_gpu_vs_cpu(input);

    GpuTensor gpu_in(input);
    Tensor result = softmax_gpu(gpu_in).download();
    for (int j = 0; j < 10; ++j) {
        EXPECT_NEAR(result.at(0, j), 0.1f, 1e-5f);
    }
}

// ============================================================================
// Numerical Stability
// ============================================================================

TEST(SoftmaxGPU, LargePositiveValues) {
    // Large values that would overflow without max-subtraction
    Tensor input({1, 4}, {1000.0f, 1001.0f, 1002.0f, 1003.0f});
    verify_softmax_gpu_vs_cpu(input);

    GpuTensor gpu_in(input);
    Tensor result = softmax_gpu(gpu_in).download();
    verify_softmax_properties(result, 1, 4);
}

TEST(SoftmaxGPU, LargeNegativeValues) {
    Tensor input({1, 4}, {-1000.0f, -999.0f, -998.0f, -997.0f});
    verify_softmax_gpu_vs_cpu(input);

    GpuTensor gpu_in(input);
    Tensor result = softmax_gpu(gpu_in).download();
    verify_softmax_properties(result, 1, 4);
}

TEST(SoftmaxGPU, MixedExtremeValues) {
    Tensor input({1, 5}, {-100.0f, 0.0f, 100.0f, -50.0f, 50.0f});
    verify_softmax_gpu_vs_cpu(input);
}

// ============================================================================
// Architecture-Relevant: 10 Classes
// ============================================================================

TEST(SoftmaxGPU, TenClasses) {
    Tensor input = Tensor::rand({1, 10}, 42);
    verify_softmax_gpu_vs_cpu(input);

    GpuTensor gpu_in(input);
    Tensor result = softmax_gpu(gpu_in).download();
    verify_softmax_properties(result, 1, 10);
}

TEST(SoftmaxGPU, TenClasses_OneDominant) {
    // One logit much larger — should get probability near 1.0
    Tensor input = Tensor::zeros({1, 10});
    input[5] = 100.0f;
    verify_softmax_gpu_vs_cpu(input);

    GpuTensor gpu_in(input);
    Tensor result = softmax_gpu(gpu_in).download();
    EXPECT_GT(result.at(0, 5), 0.99f);
}

// ============================================================================
// Batch Tests
// ============================================================================

TEST(SoftmaxGPU, BatchOf4) {
    Tensor input = Tensor::rand({4, 10}, 42);
    verify_softmax_gpu_vs_cpu(input);

    GpuTensor gpu_in(input);
    Tensor result = softmax_gpu(gpu_in).download();
    verify_softmax_properties(result, 4, 10);
}

TEST(SoftmaxGPU, BatchOf16) {
    Tensor input = Tensor::rand({16, 10}, 42);
    verify_softmax_gpu_vs_cpu(input);

    GpuTensor gpu_in(input);
    Tensor result = softmax_gpu(gpu_in).download();
    verify_softmax_properties(result, 16, 10);
}

TEST(SoftmaxGPU, BatchOf64) {
    Tensor input = Tensor::rand({64, 10}, 42);
    verify_softmax_gpu_vs_cpu(input);
}

// ============================================================================
// Larger Class Counts
// ============================================================================

TEST(SoftmaxGPU, HundredClasses) {
    Tensor input = Tensor::rand({1, 100}, 42);
    verify_softmax_gpu_vs_cpu(input);

    GpuTensor gpu_in(input);
    Tensor result = softmax_gpu(gpu_in).download();
    verify_softmax_properties(result, 1, 100);
}

TEST(SoftmaxGPU, ThousandClasses) {
    Tensor input = Tensor::rand({4, 1000}, 42);
    verify_softmax_gpu_vs_cpu(input);
}
