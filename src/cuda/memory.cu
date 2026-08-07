#include "gpu_memory.h"
#include "cuda_utils.h"

#include <cuda_runtime.h>

#include <cstring>

// ============================================================================
// GPU Memory Allocation / Deallocation
// ============================================================================

template <typename T>
T* gpu_alloc(size_t count) {
    T* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ptr), count * sizeof(T)));
    return d_ptr;
}

void gpu_free(void* d_ptr) {
    if (d_ptr != nullptr) {
        CUDA_CHECK(cudaFree(d_ptr));
    }
}

// ============================================================================
// Host ↔ Device Transfers
// ============================================================================

template <typename T>
void host_to_device(T* d_dst, const T* h_src, size_t count) {
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, count * sizeof(T),
                          cudaMemcpyHostToDevice));
}

template <typename T>
void device_to_host(T* h_dst, const T* d_src, size_t count) {
    CUDA_CHECK(cudaMemcpy(h_dst, d_src, count * sizeof(T),
                          cudaMemcpyDeviceToHost));
}

template <typename T>
void device_to_device(T* d_dst, const T* d_src, size_t count) {
    CUDA_CHECK(cudaMemcpy(d_dst, d_src, count * sizeof(T),
                          cudaMemcpyDeviceToDevice));
}

// ============================================================================
// Device Memset
// ============================================================================

template <typename T>
void gpu_memset_zero(T* d_ptr, size_t count) {
    CUDA_CHECK(cudaMemset(d_ptr, 0, count * sizeof(T)));
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================
// We instantiate for float (the only type used in our CNN) to allow
// the implementation to live in this .cu file rather than the header.

template float* gpu_alloc<float>(size_t count);
template void host_to_device<float>(float* d_dst, const float* h_src, size_t count);
template void device_to_host<float>(float* h_dst, const float* d_src, size_t count);
template void device_to_device<float>(float* d_dst, const float* d_src, size_t count);
template void gpu_memset_zero<float>(float* d_ptr, size_t count);
