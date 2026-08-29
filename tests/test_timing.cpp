#include "cuda_timer.h"
#include "device_info.h"
#include "inference.h"
#include "model.h"
#include "tensor.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <vector>

// ============================================================================
// CPU vs GPU Timing Comparison
// ============================================================================
//
// Benchmarks the full inference pipeline on both CPU and GPU, reporting
// per-image latency and speedup. Uses CudaTimer for hardware-accurate
// GPU measurement and std::chrono for CPU.
//
// These are not pass/fail correctness tests — they report performance
// metrics and verify basic sanity (GPU should finish in finite time).
//
// ============================================================================

class TimingBenchmark : public ::testing::Test {
protected:
    Model model;
    static constexpr int WARMUP_ITERS = 3;
    static constexpr int BENCH_ITERS = 10;

    void SetUp() override {
        model.initialize_xavier(42);
    }
};

// ============================================================================
// Helper: measure CPU inference time (milliseconds)
// ============================================================================

static double bench_cpu(const Model& model, const Tensor& input,
                        int warmup, int iters) {
    // Warm up
    for (int i = 0; i < warmup; ++i) {
        Tensor out = cpu_inference(model, input);
        (void)out;
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        Tensor out = cpu_inference(model, input);
        (void)out;
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return total_ms / static_cast<double>(iters);
}

// ============================================================================
// Helper: measure GPU inference time (milliseconds)
// ============================================================================

static double bench_gpu(const Model& model, const Tensor& input,
                        int warmup, int iters) {
    // Warm up (includes CUDA context initialization, JIT compilation, etc.)
    for (int i = 0; i < warmup; ++i) {
        Tensor out = gpu_inference(model, input);
        (void)out;
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        Tensor out = gpu_inference(model, input);
        (void)out;
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return total_ms / static_cast<double>(iters);
}

// ============================================================================
// Single Image Benchmark
// ============================================================================

TEST_F(TimingBenchmark, SingleImage) {
    Tensor input = Tensor::rand({1, 1, 28, 28}, 42);

    double cpu_ms = bench_cpu(model, input, WARMUP_ITERS, BENCH_ITERS);
    double gpu_ms = bench_gpu(model, input, WARMUP_ITERS, BENCH_ITERS);
    double speedup = cpu_ms / gpu_ms;

    std::printf("\n");
    std::printf("  ┌─────────────────────────────────────────────┐\n");
    std::printf("  │  Single Image Benchmark (28×28)             │\n");
    std::printf("  ├─────────────────────────────────────────────┤\n");
    std::printf("  │  CPU:     %8.2f ms/image                 │\n", cpu_ms);
    std::printf("  │  GPU:     %8.2f ms/image                 │\n", gpu_ms);
    std::printf("  │  Speedup: %8.2fx                         │\n", speedup);
    std::printf("  └─────────────────────────────────────────────┘\n\n");

    // Sanity: both should complete in reasonable time
    EXPECT_GT(cpu_ms, 0.0);
    EXPECT_GT(gpu_ms, 0.0);
    EXPECT_LT(cpu_ms, 10000.0);  // < 10 seconds
    EXPECT_LT(gpu_ms, 10000.0);
}

// ============================================================================
// Batch Benchmarks — Scaling Behavior
// ============================================================================

TEST_F(TimingBenchmark, BatchScaling) {
    std::vector<int> batch_sizes = {1, 2, 4, 8, 16};

    std::printf("\n");
    std::printf("  ┌──────────┬────────────┬────────────┬──────────┐\n");
    std::printf("  │  Batch   │  CPU (ms)  │  GPU (ms)  │  Speedup │\n");
    std::printf("  ├──────────┼────────────┼────────────┼──────────┤\n");

    for (int batch : batch_sizes) {
        Tensor input = Tensor::rand({batch, 1, 28, 28}, 42);

        double cpu_ms = bench_cpu(model, input, WARMUP_ITERS, BENCH_ITERS);
        double gpu_ms = bench_gpu(model, input, WARMUP_ITERS, BENCH_ITERS);
        double speedup = cpu_ms / gpu_ms;

        std::printf("  │  %4d    │  %8.2f  │  %8.2f  │  %6.2fx │\n",
                    batch, cpu_ms, gpu_ms, speedup);

        EXPECT_GT(cpu_ms, 0.0);
        EXPECT_GT(gpu_ms, 0.0);
    }

    std::printf("  └──────────┴────────────┴────────────┴──────────┘\n\n");
}

// ============================================================================
// Per-Layer Timing (GPU)
// ============================================================================

TEST_F(TimingBenchmark, DeviceInfo) {
    // Print device info for context on benchmark results
    std::printf("\n");
    print_device_info();
    std::printf("\n");

    DeviceInfo info = get_device_info();
    EXPECT_GT(info.num_sms, 0);
    EXPECT_GT(info.global_memory_bytes, 0u);
    EXPECT_FALSE(info.name.empty());
}
