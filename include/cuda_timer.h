#pragma once

// ============================================================================
// CUDA Timer — cudaEvent-based High-Resolution GPU Timing
// ============================================================================
//
// Uses cudaEvent pairs to measure GPU operations with hardware-level
// precision. Suitable for timing kernel launches, memory transfers,
// and full pipeline stages.
//
// Usage:
//   CudaTimer timer;
//   timer.start();
//   // ... GPU operations ...
//   timer.stop();
//   float ms = timer.elapsed_ms();
//   printf("Elapsed: %.3f ms\n", ms);
//
// Or for quick one-shot timing:
//   auto [result, ms] = timed_upload(cpu_tensor);
//   printf("Upload took %.3f ms\n", ms);
//
// ============================================================================

#include <cuda_runtime.h>

/// RAII wrapper around a pair of cudaEvents for GPU timing.
class CudaTimer {
public:
    CudaTimer();
    ~CudaTimer();

    // Non-copyable, non-movable (owns CUDA events)
    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;
    CudaTimer(CudaTimer&&) = delete;
    CudaTimer& operator=(CudaTimer&&) = delete;

    /// Record the start event on the current CUDA stream.
    void start();

    /// Record the stop event and synchronize.
    void stop();

    /// Returns the elapsed time in milliseconds between start and stop.
    /// Must call stop() before calling this.
    float elapsed_ms() const;

private:
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
};

// ============================================================================
// Timed Transfer Utilities
// ============================================================================

#include "gpu_tensor.h"
#include "tensor.h"

/// Upload a CPU tensor to GPU and return the elapsed time.
/// @return pair of {GpuTensor, elapsed_ms}
struct TimedUploadResult {
    GpuTensor gpu_tensor;
    float elapsed_ms;
};

TimedUploadResult timed_upload(const Tensor& cpu_tensor);

/// Download a GPU tensor to CPU and return the elapsed time.
/// @return pair of {Tensor, elapsed_ms}
struct TimedDownloadResult {
    Tensor cpu_tensor;
    float elapsed_ms;
};

TimedDownloadResult timed_download(const GpuTensor& gpu_tensor);

/// Print transfer statistics for a given data size and elapsed time.
void print_transfer_stats(const char* direction, size_t bytes, float elapsed_ms);
