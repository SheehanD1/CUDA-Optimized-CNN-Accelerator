#include "device_info.h"
#include "cuda_utils.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <string>

// ============================================================================
// DeviceInfo Helper Methods
// ============================================================================

std::string DeviceInfo::compute_capability() const {
    return std::to_string(compute_major) + "." + std::to_string(compute_minor);
}

std::string DeviceInfo::global_memory_string() const {
    double gb = static_cast<double>(global_memory_bytes) / (1024.0 * 1024.0 * 1024.0);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f GB", gb);
    return std::string(buf);
}

double DeviceInfo::memory_bandwidth_gbps() const {
    // Bandwidth = memory_clock (Hz) * bus_width (bits) * 2 (DDR) / 8 (bits/byte)
    // memory_clock_mhz is in MHz, so multiply by 1e6 to get Hz
    double clock_hz = static_cast<double>(memory_clock_mhz) * 1e6;
    double bus_bytes = static_cast<double>(memory_bus_width) / 8.0;
    double bandwidth_bytes = clock_hz * bus_bytes * 2.0;  // DDR = double data rate
    return bandwidth_bytes / (1024.0 * 1024.0 * 1024.0);  // Convert to GB/s
}

// ============================================================================
// Query Device Properties
// ============================================================================

DeviceInfo get_device_info(int device_id) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    DeviceInfo info;
    info.name = std::string(prop.name);
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    info.global_memory_bytes = prop.totalGlobalMem;
    info.shared_memory_per_block = prop.sharedMemPerBlock;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.warp_size = prop.warpSize;
    info.num_sms = prop.multiProcessorCount;
    info.max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;
    info.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    info.clock_rate_mhz = prop.clockRate / 1000;          // clockRate is in kHz
    info.memory_clock_mhz = prop.memoryClockRate / 1000;  // memoryClockRate is in kHz
    info.memory_bus_width = prop.memoryBusWidth;
    info.l2_cache_size = prop.l2CacheSize;

    return info;
}

int get_cuda_device_count() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

// ============================================================================
// Print Device Summary
// ============================================================================

void print_device_info(int device_id) {
    DeviceInfo info = get_device_info(device_id);

    std::printf("====================================================\n");
    std::printf("  CUDA Device %d: %s\n", device_id, info.name.c_str());
    std::printf("====================================================\n");
    std::printf("  Compute Capability:    %s\n",
                info.compute_capability().c_str());
    std::printf("  Global Memory:         %s (%zu bytes)\n",
                info.global_memory_string().c_str(),
                info.global_memory_bytes);
    std::printf("  Shared Memory/Block:   %zu bytes\n",
                info.shared_memory_per_block);
    std::printf("  L2 Cache Size:         %d bytes\n",
                info.l2_cache_size);
    std::printf("  Streaming Multiprocs:  %d\n", info.num_sms);
    std::printf("  Max Threads/Block:     %d\n", info.max_threads_per_block);
    std::printf("  Max Threads/SM:        %d\n", info.max_threads_per_sm);
    std::printf("  Max Blocks/SM:         %d\n", info.max_blocks_per_sm);
    std::printf("  Warp Size:             %d\n", info.warp_size);
    std::printf("  GPU Clock:             %d MHz\n", info.clock_rate_mhz);
    std::printf("  Memory Clock:          %d MHz\n", info.memory_clock_mhz);
    std::printf("  Memory Bus Width:      %d bits\n", info.memory_bus_width);
    std::printf("  Memory Bandwidth:      %.1f GB/s (theoretical)\n",
                info.memory_bandwidth_gbps());
    std::printf("====================================================\n");
}
