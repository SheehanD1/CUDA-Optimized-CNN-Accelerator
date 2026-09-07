#include "cuda_stream.h"
#include "gpu_model.h"
#include "inference.h"
#include "model.h"
#include "tensor.h"

#include <gtest/gtest.h>

#include <vector>

// ============================================================================
// Optimized Pipeline End-to-End Validation
// ============================================================================
//
// Validates all optimizations produce correct results:
//   1. GpuModel (cached weights) matches cpu_inference()
//   2. GpuModel matches gpu_inference() (non-cached)
//   3. Async transfers produce correct results
//   4. CudaStream RAII works correctly
//
// ============================================================================

class OptimizedPipelineTest : public ::testing::Test {
protected:
    Model model;

    void SetUp() override {
        model.initialize_xavier(42);
    }
};

// ============================================================================
// GpuModel vs CPU
// ============================================================================

TEST_F(OptimizedPipelineTest, GpuModel_MatchesCPU_Single) {
    GpuModel gpu_model(model);
    Tensor input = Tensor::rand({1, 1, 28, 28}, 42);

    Tensor cpu_output = cpu_inference(model, input);
    Tensor gpu_output = gpu_model.inference(input);

    EXPECT_EQ(cpu_output.shape(), gpu_output.shape());
    EXPECT_TRUE(cpu_output.allclose(gpu_output, 1e-3f))
        << "GpuModel differs from CPU. Max diff: "
        << cpu_output.max_diff(gpu_output);
}

TEST_F(OptimizedPipelineTest, GpuModel_MatchesCPU_Batch4) {
    GpuModel gpu_model(model);
    Tensor input = Tensor::rand({4, 1, 28, 28}, 42);

    Tensor cpu_output = cpu_inference(model, input);
    Tensor gpu_output = gpu_model.inference(input);

    EXPECT_TRUE(cpu_output.allclose(gpu_output, 1e-3f));
}

TEST_F(OptimizedPipelineTest, GpuModel_MatchesCPU_Batch8) {
    GpuModel gpu_model(model);
    Tensor input = Tensor::rand({8, 1, 28, 28}, 42);

    Tensor cpu_output = cpu_inference(model, input);
    Tensor gpu_output = gpu_model.inference(input);

    EXPECT_TRUE(cpu_output.allclose(gpu_output, 1e-3f));
}

// ============================================================================
// GpuModel vs gpu_inference (non-cached)
// ============================================================================

TEST_F(OptimizedPipelineTest, GpuModel_MatchesGpuInference) {
    GpuModel gpu_model(model);
    Tensor input = Tensor::rand({1, 1, 28, 28}, 100);

    Tensor cached_output = gpu_model.inference(input);
    Tensor uncached_output = gpu_inference(model, input);

    EXPECT_TRUE(cached_output.allclose(uncached_output, 1e-5f))
        << "Cached vs uncached differ. Max diff: "
        << cached_output.max_diff(uncached_output);
}

// ============================================================================
// GpuModel Prediction Agreement
// ============================================================================

TEST_F(OptimizedPipelineTest, GpuModel_PredictionsMatch) {
    GpuModel gpu_model(model);

    for (int seed = 0; seed < 10; ++seed) {
        Tensor input = Tensor::rand({1, 1, 28, 28}, seed);

        std::vector<int> cpu_preds = cpu_predict(model, input);
        std::vector<int> gpu_preds = gpu_model.predict(input);

        EXPECT_EQ(cpu_preds[0], gpu_preds[0])
            << "Prediction mismatch on seed " << seed;
    }
}

// ============================================================================
// GpuModel Repeated Inference (weight reuse)
// ============================================================================

TEST_F(OptimizedPipelineTest, GpuModel_RepeatedInference) {
    GpuModel gpu_model(model);

    // Run the same input multiple times — results should be identical
    Tensor input = Tensor::rand({1, 1, 28, 28}, 42);
    Tensor first = gpu_model.inference(input);

    for (int i = 0; i < 5; ++i) {
        Tensor result = gpu_model.inference(input);
        EXPECT_TRUE(first.allclose(result, 1e-6f))
            << "Repeated inference #" << i << " differs. Max diff: "
            << first.max_diff(result);
    }
}

TEST_F(OptimizedPipelineTest, GpuModel_DifferentInputs) {
    GpuModel gpu_model(model);

    // Different inputs should produce different outputs
    Tensor input1 = Tensor::rand({1, 1, 28, 28}, 42);
    Tensor input2 = Tensor::rand({1, 1, 28, 28}, 99);

    Tensor out1 = gpu_model.inference(input1);
    Tensor out2 = gpu_model.inference(input2);

    // Outputs should differ (extremely unlikely to be identical for random inputs)
    bool same = out1.allclose(out2, 1e-6f);
    EXPECT_FALSE(same) << "Different inputs produced identical outputs";
}

// ============================================================================
// CudaStream RAII
// ============================================================================

TEST(CudaStreamTest, CreateAndDestroy) {
    // Should not throw
    CudaStream stream;
    EXPECT_NE(stream.get(), nullptr);
}

TEST(CudaStreamTest, MoveSemantics) {
    CudaStream stream1;
    cudaStream_t raw = stream1.get();

    CudaStream stream2 = std::move(stream1);
    EXPECT_EQ(stream2.get(), raw);
    EXPECT_EQ(stream1.get(), nullptr);
}

TEST(CudaStreamTest, Synchronize) {
    CudaStream stream;
    // Synchronize on empty stream should succeed immediately
    stream.synchronize();
    EXPECT_TRUE(stream.is_complete());
}

// ============================================================================
// Async Transfers
// ============================================================================

TEST(AsyncTransferTest, UploadDownloadRoundTrip) {
    Tensor original = Tensor::rand({1, 1, 28, 28}, 42);
    GpuTensor gpu_tensor(original.shape());

    CudaStream stream;
    gpu_tensor.upload_async(original, stream);
    Tensor result = gpu_tensor.download_async(stream);
    stream.synchronize();

    EXPECT_TRUE(original.allclose(result, 1e-7f));
}

TEST(AsyncTransferTest, MultipleStreams) {
    Tensor t1 = Tensor::rand({1, 1, 28, 28}, 42);
    Tensor t2 = Tensor::rand({1, 1, 28, 28}, 99);

    GpuTensor g1(t1.shape());
    GpuTensor g2(t2.shape());

    CudaStream stream1;
    CudaStream stream2;

    // Concurrent uploads on different streams
    g1.upload_async(t1, stream1);
    g2.upload_async(t2, stream2);

    // Concurrent downloads
    Tensor r1 = g1.download_async(stream1);
    Tensor r2 = g2.download_async(stream2);

    stream1.synchronize();
    stream2.synchronize();

    EXPECT_TRUE(t1.allclose(r1, 1e-7f));
    EXPECT_TRUE(t2.allclose(r2, 1e-7f));
}

// ============================================================================
// Softmax Properties on Optimized Pipeline
// ============================================================================

TEST_F(OptimizedPipelineTest, SoftmaxProperties) {
    GpuModel gpu_model(model);
    Tensor input = Tensor::rand({4, 1, 28, 28}, 42);
    Tensor output = gpu_model.inference(input);

    EXPECT_EQ(output.dim(0), 4);
    EXPECT_EQ(output.dim(1), 10);

    for (int n = 0; n < 4; ++n) {
        float sum = 0.0f;
        for (int j = 0; j < 10; ++j) {
            float val = output.at(n, j);
            EXPECT_GE(val, 0.0f);
            EXPECT_LE(val, 1.0f);
            sum += val;
        }
        EXPECT_NEAR(sum, 1.0f, 1e-4f);
    }
}
