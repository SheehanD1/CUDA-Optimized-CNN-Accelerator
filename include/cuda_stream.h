#pragma once

// ============================================================================
// CudaStream — RAII Wrapper for cudaStream_t
// ============================================================================
//
// Manages the lifecycle of a CUDA stream, enabling:
//   - Overlapped H↔D transfers with kernel execution
//   - Concurrent kernel launches on different streams
//   - Ordered execution within a single stream
//
// Usage:
//   CudaStream stream;
//   kernel<<<grid, block, 0, stream.get()>>>(...);
//   cudaMemcpyAsync(dst, src, size, kind, stream.get());
//   stream.synchronize();  // Wait for all work on this stream
//
// ============================================================================

#include "cuda_utils.h"

#include <cuda_runtime.h>

class CudaStream {
public:
    /// Create a new CUDA stream.
    CudaStream() {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    /// Destroy the CUDA stream.
    ~CudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    // Move semantics
    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }

    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_ != nullptr) {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    // No copies
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    /// Get the underlying cudaStream_t for API calls.
    cudaStream_t get() const { return stream_; }

    /// Implicit conversion to cudaStream_t for convenience.
    operator cudaStream_t() const { return stream_; }

    /// Block the host until all operations on this stream complete.
    void synchronize() const {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    /// Returns true if all operations on this stream have completed.
    bool is_complete() const {
        cudaError_t err = cudaStreamQuery(stream_);
        if (err == cudaSuccess) return true;
        if (err == cudaErrorNotReady) return false;
        CUDA_CHECK(err);  // Throw on real errors
        return false;
    }

private:
    cudaStream_t stream_ = nullptr;
};
