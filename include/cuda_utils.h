#pragma once

// ============================================================================
// CUDA Utility Macros — Error Checking and Diagnostics
// ============================================================================
//
// Every CUDA API call and kernel launch should be wrapped with these macros
// to catch errors early with helpful diagnostics (file, line, error string).
//
// Usage:
//
//   // Wrap CUDA API calls:
//   CUDA_CHECK(cudaMalloc(&ptr, size));
//   CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
//
//   // After kernel launches:
//   my_kernel<<<grid, block>>>(args...);
//   KERNEL_CHECK();
//
//   // Synchronize and check:
//   CUDA_SYNC_CHECK();
//
// ============================================================================

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>

// ============================================================================
// CUDA_CHECK — Wrap CUDA API calls
// ============================================================================
// Checks the return value of a CUDA runtime API call. If the call fails,
// prints the error with file/line info and aborts.
//
// Example:
//   CUDA_CHECK(cudaMalloc(&ptr, num_bytes));

#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t error = (call);                                            \
        if (error != cudaSuccess) {                                            \
            std::fprintf(stderr,                                               \
                "CUDA Error at %s:%d\n"                                        \
                "  Code:    %d\n"                                              \
                "  String:  %s\n"                                              \
                "  Call:    %s\n",                                              \
                __FILE__, __LINE__,                                            \
                static_cast<int>(error),                                       \
                cudaGetErrorString(error),                                     \
                #call);                                                        \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// ============================================================================
// KERNEL_CHECK — Check for errors after kernel launch
// ============================================================================
// CUDA kernel launches are asynchronous and don't return an error code.
// This macro checks for:
//   1. Launch configuration errors (cudaGetLastError)
//   2. Does NOT synchronize — use CUDA_SYNC_CHECK for that.
//
// Place immediately after every kernel<<<...>>>(...) call.
//
// Example:
//   relu_kernel<<<grid, block>>>(data, n);
//   KERNEL_CHECK();

#define KERNEL_CHECK()                                                         \
    do {                                                                       \
        cudaError_t error = cudaGetLastError();                                \
        if (error != cudaSuccess) {                                            \
            std::fprintf(stderr,                                               \
                "CUDA Kernel Launch Error at %s:%d\n"                          \
                "  Code:    %d\n"                                              \
                "  String:  %s\n",                                             \
                __FILE__, __LINE__,                                            \
                static_cast<int>(error),                                       \
                cudaGetErrorString(error));                                     \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// ============================================================================
// CUDA_SYNC_CHECK — Synchronize device and check for execution errors
// ============================================================================
// Use this after kernel launches when you need to verify the kernel
// completed successfully (e.g., in debug builds or after the final kernel
// in a pipeline stage).
//
// This is expensive (blocks until all GPU work completes), so use sparingly
// in production code.
//
// Example:
//   conv2d_kernel<<<grid, block>>>(args...);
//   CUDA_SYNC_CHECK();  // Blocks until kernel finishes, checks for errors

#define CUDA_SYNC_CHECK()                                                      \
    do {                                                                       \
        cudaError_t error = cudaDeviceSynchronize();                           \
        if (error != cudaSuccess) {                                            \
            std::fprintf(stderr,                                               \
                "CUDA Sync Error at %s:%d\n"                                   \
                "  Code:    %d\n"                                              \
                "  String:  %s\n",                                             \
                __FILE__, __LINE__,                                            \
                static_cast<int>(error),                                       \
                cudaGetErrorString(error));                                     \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// ============================================================================
// Utility: compute grid dimensions for 1D kernels
// ============================================================================

/// Compute the number of blocks needed for n elements with given block size.
/// ceil(n / block_size)
inline int cuda_grid_size(int n, int block_size) {
    return (n + block_size - 1) / block_size;
}

// ============================================================================
// Utility: stringify CUDA error code
// ============================================================================

/// Convert a cudaError_t to a human-readable string.
inline std::string cuda_error_string(cudaError_t error) {
    return std::string(cudaGetErrorName(error)) + ": " +
           std::string(cudaGetErrorString(error));
}
