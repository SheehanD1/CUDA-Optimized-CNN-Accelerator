#include "inference.h"
#include "model.h"
#include "tensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

// ============================================================================
// GPU vs CPU End-to-End Pipeline Validation
// ============================================================================
//
// Verifies that gpu_inference() produces the same output as cpu_inference()
// for the full 11-layer pipeline. This is the ultimate correctness check —
// if individual kernel tests pass but this fails, there's a wiring bug.
//
// ============================================================================

class GpuPipelineTest : public ::testing::Test {
protected:
    Model model;

    void SetUp() override {
        // Use deterministic Xavier initialization with fixed seed
        model.initialize_xavier(42);
    }
};

// ============================================================================
// Helper: compare full pipeline outputs
// ============================================================================

static void verify_pipeline_gpu_vs_cpu(
    const Model& model, const Tensor& input, float atol = 1e-3f
) {
    Tensor cpu_output = cpu_inference(model, input);
    Tensor gpu_output = gpu_inference(model, input);

    EXPECT_EQ(cpu_output.shape(), gpu_output.shape())
        << "Shape mismatch: CPU " << cpu_output.shape().size()
        << "D vs GPU " << gpu_output.shape().size() << "D";

    EXPECT_TRUE(cpu_output.allclose(gpu_output, atol))
        << "GPU pipeline output differs from CPU. Max diff: "
        << cpu_output.max_diff(gpu_output);

    // Verify softmax properties on GPU output
    int batch_size = gpu_output.dim(0);
    int num_classes = gpu_output.dim(1);
    for (int n = 0; n < batch_size; ++n) {
        float sum = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            float val = gpu_output.at(n, j);
            EXPECT_GE(val, 0.0f);
            EXPECT_LE(val, 1.0f);
            sum += val;
        }
        EXPECT_NEAR(sum, 1.0f, 1e-4f)
            << "Row " << n << " probabilities don't sum to 1.0";
    }
}

// ============================================================================
// Single Image Tests
// ============================================================================

TEST_F(GpuPipelineTest, SingleImage_Zeros) {
    Tensor input = Tensor::zeros({1, 1, 28, 28});
    verify_pipeline_gpu_vs_cpu(model, input);
}

TEST_F(GpuPipelineTest, SingleImage_Ones) {
    Tensor input({1, 1, 28, 28});
    for (int i = 0; i < input.num_elements(); ++i) input[i] = 1.0f;
    verify_pipeline_gpu_vs_cpu(model, input);
}

TEST_F(GpuPipelineTest, SingleImage_Random) {
    Tensor input = Tensor::rand({1, 1, 28, 28}, 100);
    verify_pipeline_gpu_vs_cpu(model, input);
}

TEST_F(GpuPipelineTest, SingleImage_Normalized) {
    // Simulate MNIST-like normalized input [0, 1]
    Tensor input = Tensor::rand({1, 1, 28, 28}, 200);
    verify_pipeline_gpu_vs_cpu(model, input);
}

// ============================================================================
// Prediction Agreement
// ============================================================================

TEST_F(GpuPipelineTest, PredictionsMatch) {
    // Run multiple random images and verify CPU/GPU predict same class
    for (int seed = 0; seed < 10; ++seed) {
        Tensor input = Tensor::rand({1, 1, 28, 28}, seed);

        std::vector<int> cpu_preds = cpu_predict(model, input);
        std::vector<int> gpu_preds = gpu_predict(model, input);

        EXPECT_EQ(cpu_preds.size(), gpu_preds.size());
        EXPECT_EQ(cpu_preds[0], gpu_preds[0])
            << "Prediction mismatch on seed " << seed
            << ": CPU=" << cpu_preds[0] << " GPU=" << gpu_preds[0];
    }
}

// ============================================================================
// Batch Tests
// ============================================================================

TEST_F(GpuPipelineTest, BatchOf2) {
    Tensor input = Tensor::rand({2, 1, 28, 28}, 42);
    verify_pipeline_gpu_vs_cpu(model, input);
}

TEST_F(GpuPipelineTest, BatchOf4) {
    Tensor input = Tensor::rand({4, 1, 28, 28}, 42);
    verify_pipeline_gpu_vs_cpu(model, input);
}

TEST_F(GpuPipelineTest, BatchOf8) {
    Tensor input = Tensor::rand({8, 1, 28, 28}, 42);
    verify_pipeline_gpu_vs_cpu(model, input);
}

// ============================================================================
// Different Model Weights
// ============================================================================

TEST(GpuPipelineSeedTest, DifferentWeightSeed) {
    Model model2;
    model2.initialize_xavier(99);

    Tensor input = Tensor::rand({1, 1, 28, 28}, 42);
    verify_pipeline_gpu_vs_cpu(model2, input);
}

TEST(GpuPipelineSeedTest, AnotherWeightSeed) {
    Model model3;
    model3.initialize_xavier(1234);

    Tensor input = Tensor::rand({2, 1, 28, 28}, 55);
    verify_pipeline_gpu_vs_cpu(model3, input);
}

// ============================================================================
// Output Shape Verification
// ============================================================================

TEST_F(GpuPipelineTest, OutputShape_Single) {
    Tensor input = Tensor::rand({1, 1, 28, 28}, 42);
    Tensor output = gpu_inference(model, input);

    EXPECT_EQ(output.ndim(), 2);
    EXPECT_EQ(output.dim(0), 1);
    EXPECT_EQ(output.dim(1), 10);
}

TEST_F(GpuPipelineTest, OutputShape_Batch) {
    Tensor input = Tensor::rand({4, 1, 28, 28}, 42);
    Tensor output = gpu_inference(model, input);

    EXPECT_EQ(output.ndim(), 2);
    EXPECT_EQ(output.dim(0), 4);
    EXPECT_EQ(output.dim(1), 10);
}
