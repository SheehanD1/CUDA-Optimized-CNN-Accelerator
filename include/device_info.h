#pragma once

// ============================================================================
// GPU Device Info — Query and Display CUDA Device Properties
// ============================================================================
//
// Queries the CUDA runtime for hardware capabilities of the active GPU.
// Useful for build diagnostics, performance tuning, and documenting the
// test environment in benchmarks.
//
// Usage:
//   print_device_info();         // Print full device summary to stdout
//   auto info = get_device_info(); // Get structured device properties
//
// ============================================================================

#include <string>

/// Structured GPU device properties.
struct DeviceInfo {
    std::string name;                   ///< Device name (e.g., "NVIDIA GeForce RTX 4090")
    int compute_major;                  ///< Compute capability major version
    int compute_minor;                  ///< Compute capability minor version
    size_t global_memory_bytes;         ///< Total global memory in bytes
    size_t shared_memory_per_block;     ///< Shared memory per block in bytes
    int max_threads_per_block;          ///< Maximum threads per block
    int warp_size;                      ///< Threads per warp (typically 32)
    int num_sms;                        ///< Number of Streaming Multiprocessors
    int max_blocks_per_sm;              ///< Max blocks per SM
    int max_threads_per_sm;             ///< Max threads per SM
    int clock_rate_mhz;                 ///< GPU clock rate in MHz
    int memory_clock_mhz;              ///< Memory clock rate in MHz
    int memory_bus_width;               ///< Memory bus width in bits
    int l2_cache_size;                  ///< L2 cache size in bytes

    /// Returns compute capability as a string (e.g., "8.9").
    std::string compute_capability() const;

    /// Returns global memory in human-readable format (e.g., "24.0 GB").
    std::string global_memory_string() const;

    /// Returns theoretical memory bandwidth in GB/s.
    double memory_bandwidth_gbps() const;
};

/// Query device properties for the currently active CUDA device.
/// @param device_id  CUDA device index (default: 0)
DeviceInfo get_device_info(int device_id = 0);

/// Print a formatted summary of GPU device properties to stdout.
/// @param device_id  CUDA device index (default: 0)
void print_device_info(int device_id = 0);

/// Returns the number of CUDA-capable devices.
int get_cuda_device_count();
