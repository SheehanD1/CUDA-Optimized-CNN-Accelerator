#include "cuda_timer.h"
#include "device_info.h"
#include "gpu_tensor.h"
#include "kernels.h"
#include "tensor.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <vector>

// ============================================================================
// Naive vs Tiled Conv2D Performance Comparison
// ============================================================================
//
// Benchmarks both naive and tiled Conv2D kernels at architecture-relevant
// sizes, reporting latency and speedup. Uses CudaTimer for hardware-accurate
// GPU timing with warmup iterations.
//
// ============================================================================

static constexpr int WARMUP = 5;
static constexpr int ITERS = 20;

struct ConvConfig {
    const char* name;
    int batch, in_c, in_h, in_w;
    int out_c, kern_h, kern_w;
    int stride, padding;
};

static double bench_naive(const GpuTensor& input, const GpuTensor& weights,
                           const GpuTensor& bias, int stride, int padding) {
    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        GpuTensor out = conv2d_gpu(input, weights, bias, stride, padding);
        (void)out;
    }

    CudaTimer timer;
    timer.start();
    for (int i = 0; i < ITERS; ++i) {
        GpuTensor out = conv2d_gpu(input, weights, bias, stride, padding);
        (void)out;
    }
    timer.stop();
    return timer.elapsed_ms() / static_cast<double>(ITERS);
}

static double bench_tiled(const GpuTensor& input, const GpuTensor& weights,
                           const GpuTensor& bias, int stride, int padding) {
    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        GpuTensor out = conv2d_tiled_gpu(input, weights, bias, stride, padding);
        (void)out;
    }

    CudaTimer timer;
    timer.start();
    for (int i = 0; i < ITERS; ++i) {
        GpuTensor out = conv2d_tiled_gpu(input, weights, bias, stride, padding);
        (void)out;
    }
    timer.stop();
    return timer.elapsed_ms() / static_cast<double>(ITERS);
}

static void run_comparison(const ConvConfig& cfg) {
    Tensor cpu_input = Tensor::rand({cfg.batch, cfg.in_c, cfg.in_h, cfg.in_w}, 42);
    Tensor cpu_weights = Tensor::rand({cfg.out_c, cfg.in_c, cfg.kern_h, cfg.kern_w}, 43);
    Tensor cpu_bias = Tensor::rand({cfg.out_c}, 44);

    GpuTensor input(cpu_input);
    GpuTensor weights(cpu_weights);
    GpuTensor bias(cpu_bias);

    double naive_ms = bench_naive(input, weights, bias, cfg.stride, cfg.padding);
    double tiled_ms = bench_tiled(input, weights, bias, cfg.stride, cfg.padding);
    double speedup = naive_ms / tiled_ms;

    std::printf("  │ %-20s │ %8.3f ms │ %8.3f ms │ %6.2fx │\n",
                cfg.name, naive_ms, tiled_ms, speedup);

    // Sanity: both should produce results
    EXPECT_GT(naive_ms, 0.0);
    EXPECT_GT(tiled_ms, 0.0);
}

// ============================================================================
// Full Comparison Table
// ============================================================================

TEST(Conv2dPerf, NaiveVsTiled) {
    std::printf("\n");
    print_device_info();

    std::printf("\n  Naive vs Tiled Conv2D — %d warmup, %d iterations, CudaTimer\n\n",
                WARMUP, ITERS);

    std::printf("  ┌──────────────────────┬─────────────┬─────────────┬─────────┐\n");
    std::printf("  │ Configuration        │   Naive     │   Tiled     │ Speedup │\n");
    std::printf("  ├──────────────────────┼─────────────┼─────────────┼─────────┤\n");

    // Architecture-relevant configurations
    std::vector<ConvConfig> configs = {
        // Conv1: small input, few channels
        {"Conv1 B=1",  1, 1, 28, 28,  8, 3, 3, 1, 1},
        {"Conv1 B=4",  4, 1, 28, 28,  8, 3, 3, 1, 1},
        {"Conv1 B=16", 16, 1, 28, 28,  8, 3, 3, 1, 1},
        {"Conv1 B=64", 64, 1, 28, 28,  8, 3, 3, 1, 1},

        // Conv2: more channels, smaller spatial
        {"Conv2 B=1",  1, 8, 14, 14, 16, 3, 3, 1, 1},
        {"Conv2 B=4",  4, 8, 14, 14, 16, 3, 3, 1, 1},
        {"Conv2 B=16", 16, 8, 14, 14, 16, 3, 3, 1, 1},
        {"Conv2 B=64", 64, 8, 14, 14, 16, 3, 3, 1, 1},

        // Larger scenarios (stress test)
        {"Large 32ch",  1, 32, 28, 28, 64, 3, 3, 1, 1},
        {"Large batch",  32, 16, 14, 14, 32, 3, 3, 1, 1},
    };

    for (const auto& cfg : configs) {
        run_comparison(cfg);
    }

    std::printf("  └──────────────────────┴─────────────┴─────────────┴─────────┘\n\n");
}
