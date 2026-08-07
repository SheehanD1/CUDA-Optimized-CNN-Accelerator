#pragma once

// ============================================================================
// GPU Memory Manager — CUDA Device Memory Operations
// ============================================================================
//
// Thin wrappers around CUDA memory API calls with automatic error checking.
// All functions use CUDA_CHECK to catch errors with file/line diagnostics.
//
// Usage:
//   float* d_ptr = gpu_alloc<float>(1024);    // Allocate 1024 floats on GPU
//   host_to_device(d_ptr, h_ptr, 1024);       // Copy 1024 floats H→D
//   device_to_host(h_ptr, d_ptr, 1024);       // Copy 1024 floats D→H
//   gpu_free(d_ptr);                          // Free GPU memory
//
// ============================================================================

#include <cstddef>

/// Allocate device memory for `count` elements of type T.
/// Returns a device pointer. Aborts on allocation failure.
template <typename T>
T* gpu_alloc(size_t count);

/// Free device memory.
void gpu_free(void* d_ptr);

/// Copy `count` elements from host to device.
/// dst must be a device pointer, src must be a host pointer.
template <typename T>
void host_to_device(T* d_dst, const T* h_src, size_t count);

/// Copy `count` elements from device to host.
/// dst must be a host pointer, src must be a device pointer.
template <typename T>
void device_to_host(T* h_dst, const T* d_src, size_t count);

/// Copy `count` elements from device to device.
/// Both pointers must be device pointers.
template <typename T>
void device_to_device(T* d_dst, const T* d_src, size_t count);

/// Set `count` elements on device to zero.
template <typename T>
void gpu_memset_zero(T* d_ptr, size_t count);
