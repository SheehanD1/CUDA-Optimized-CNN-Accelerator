#include "cuda_timer.h"
#include "device_info.h"
#include "gpu_model.h"
#include "inference.h"
#include "model.h"
#include "tensor.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <vector>

// ============================================================================
// Final Performance Benchmark Suite
// ============================================================================
//
// Comprehensive benchmark comparing all pipeline variants:
//   1. CPU inference
//   2. GPU inference (naive conv, per-call weight upload)
//   3. GPU optimized (tiled conv, cached weights via GpuModel)
//
// Reports per-image latency, throughput, and speedup at multiple batch sizes.
//
// ============================================================================

static constexpr int WARMUP = 5;
static constexpr int ITERS = 20;

// ============================================================================
// Timing Helpers
// ============================================================================

static double bench_cpu(const Model& model, const Tensor& input, int iters) {
    for (int i = 0; i < WARMUP; ++i) {
        Tensor out = cpu_inference(model, input);
        (void)out;
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        Tensor out = cpu_inference(model, input);
        (void)out;
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() /
           static_cast<double>(iters);
}

static double bench_gpu_naive(const Model& model, const Tensor& input, int iters) {
    for (int i = 0; i < WARMUP; ++i) {
        Tensor out = gpu_inference(model, input);
        (void)out;
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        Tensor out = gpu_inference(model, input);
        (void)out;
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() /
           static_cast<double>(iters);
}

static double bench_gpu_optimized(const GpuModel& gpu_model, const Tensor& input,
                                   int iters) {
    for (int i = 0; i < WARMUP; ++i) {
        Tensor out = gpu_model.inference(input);
        (void)out;
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        Tensor out = gpu_model.inference(input);
        (void)out;
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() /
           static_cast<double>(iters);
}

// ============================================================================
// Full Pipeline Comparison
// ============================================================================

TEST(FinalBenchmark, PipelineComparison) {
    Model model;
    model.initialize_xavier(42);
    GpuModel gpu_model(model);

    std::printf("\n");
    print_device_info();

    std::printf("\n  ══════════════════════════════════════════════════════════════════\n");
    std::printf("  Final Performance Benchmark — %d warmup, %d iterations\n", WARMUP, ITERS);
    std::printf("  ══════════════════════════════════════════════════════════════════\n\n");

    std::printf("  ┌───────┬────────────┬────────────┬────────────┬──────────┬──────────┐\n");
    std::printf("  │ Batch │  CPU (ms)  │ GPU Naive  │ GPU Optim  │ Naive ×  │ Optim ×  │\n");
    std::printf("  ├───────┼────────────┼────────────┼────────────┼──────────┼──────────┤\n");

    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32};

    for (int batch : batch_sizes) {
        Tensor input = Tensor::rand({batch, 1, 28, 28}, 42);

        double cpu_ms = bench_cpu(model, input, ITERS);
        double naive_ms = bench_gpu_naive(model, input, ITERS);
        double optim_ms = bench_gpu_optimized(gpu_model, input, ITERS);

        double naive_speedup = cpu_ms / naive_ms;
        double optim_speedup = cpu_ms / optim_ms;

        std::printf("  │ %5d │ %8.2f   │ %8.2f   │ %8.2f   │ %6.2fx  │ %6.2fx  │\n",
                    batch, cpu_ms, naive_ms, optim_ms, naive_speedup, optim_speedup);

        // Sanity
        EXPECT_GT(cpu_ms, 0.0);
        EXPECT_GT(naive_ms, 0.0);
        EXPECT_GT(optim_ms, 0.0);
    }

    std::printf("  └───────┴────────────┴────────────┴────────────┴──────────┴──────────┘\n");

    // ========================================================================
    // Throughput Summary
    // ========================================================================

    Tensor input16 = Tensor::rand({16, 1, 28, 28}, 42);

    double cpu_ms = bench_cpu(model, input16, ITERS);
    double naive_ms = bench_gpu_naive(model, input16, ITERS);
    double optim_ms = bench_gpu_optimized(gpu_model, input16, ITERS);

    double cpu_ips = 16000.0 / cpu_ms;
    double naive_ips = 16000.0 / naive_ms;
    double optim_ips = 16000.0 / optim_ms;

    std::printf("\n  Throughput (images/sec at batch=16):\n");
    std::printf("  ┌────────────────────┬─────────────────┐\n");
    std::printf("  │ Pipeline           │  Images/sec     │\n");
    std::printf("  ├────────────────────┼─────────────────┤\n");
    std::printf("  │ CPU                │ %11.0f     │\n", cpu_ips);
    std::printf("  │ GPU Naive          │ %11.0f     │\n", naive_ips);
    std::printf("  │ GPU Optimized      │ %11.0f     │\n", optim_ips);
    std::printf("  └────────────────────┴─────────────────┘\n");

    // ========================================================================
    // Optimization Breakdown
    // ========================================================================

    std::printf("\n  Optimization Breakdown (batch=16):\n");
    std::printf("  ┌────────────────────────────────┬───────────────┐\n");
    std::printf("  │ Optimization                   │ Improvement   │\n");
    std::printf("  ├────────────────────────────────┼───────────────┤\n");
    std::printf("  │ Naive GPU vs CPU               │ %8.2fx      │\n", cpu_ms / naive_ms);
    std::printf("  │ Tiled Conv + Cache vs Naive    │ %8.2fx      │\n", naive_ms / optim_ms);
    std::printf("  │ Total: Optimized GPU vs CPU    │ %8.2fx      │\n", cpu_ms / optim_ms);
    std::printf("  └────────────────────────────────┴───────────────┘\n\n");
}

// ============================================================================
// Per-Image Latency Comparison
// ============================================================================

TEST(FinalBenchmark, PerImageLatency) {
    Model model;
    model.initialize_xavier(42);
    GpuModel gpu_model(model);

    Tensor input = Tensor::rand({1, 1, 28, 28}, 42);

    double cpu_ms = bench_cpu(model, input, ITERS);
    double naive_ms = bench_gpu_naive(model, input, ITERS);
    double optim_ms = bench_gpu_optimized(gpu_model, input, ITERS);

    std::printf("\n  Single Image Latency:\n");
    std::printf("    CPU:           %8.3f ms\n", cpu_ms);
    std::printf("    GPU Naive:     %8.3f ms\n", naive_ms);
    std::printf("    GPU Optimized: %8.3f ms\n", optim_ms);
    std::printf("    Speedup:       %8.2fx (optimized vs CPU)\n\n", cpu_ms / optim_ms);

    EXPECT_GT(cpu_ms, 0.0);
    EXPECT_GT(optim_ms, 0.0);
}
