#include "cuda_timer.h"
#include "cuda_utils.h"
#include "gpu_memory.h"

#include <cstdio>

// ============================================================================
// CudaTimer Implementation
// ============================================================================

CudaTimer::CudaTimer() {
    CUDA_CHECK(cudaEventCreate(&start_event_));
    CUDA_CHECK(cudaEventCreate(&stop_event_));
}

CudaTimer::~CudaTimer() {
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
}

void CudaTimer::start() {
    CUDA_CHECK(cudaEventRecord(start_event_, 0));
}

void CudaTimer::stop() {
    CUDA_CHECK(cudaEventRecord(stop_event_, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));
}

float CudaTimer::elapsed_ms() const {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_event_, stop_event_));
    return ms;
}

// ============================================================================
// Timed Transfer Utilities
// ============================================================================

TimedUploadResult timed_upload(const Tensor& cpu_tensor) {
    CudaTimer timer;

    // Allocate device memory
    GpuTensor gpu(std::vector<int>(cpu_tensor.shape()));

    // Time only the actual data transfer
    timer.start();
    host_to_device(gpu.data(), cpu_tensor.data(),
                   static_cast<size_t>(cpu_tensor.num_elements()));
    timer.stop();

    return {std::move(gpu), timer.elapsed_ms()};
}

TimedDownloadResult timed_download(const GpuTensor& gpu_tensor) {
    CudaTimer timer;

    Tensor cpu(gpu_tensor.shape());

    // Time only the actual data transfer
    timer.start();
    device_to_host(cpu.data(), gpu_tensor.data(),
                   static_cast<size_t>(gpu_tensor.num_elements()));
    timer.stop();

    return {std::move(cpu), timer.elapsed_ms()};
}

// ============================================================================
// Transfer Statistics Printer
// ============================================================================

void print_transfer_stats(const char* direction, size_t bytes, float elapsed_ms) {
    double gb = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    double seconds = static_cast<double>(elapsed_ms) / 1000.0;
    double gbps = (seconds > 0.0) ? (gb / seconds) : 0.0;

    std::printf("  %-20s %8zu bytes  %8.3f ms  %6.2f GB/s\n",
                direction, bytes, static_cast<double>(elapsed_ms), gbps);
}
