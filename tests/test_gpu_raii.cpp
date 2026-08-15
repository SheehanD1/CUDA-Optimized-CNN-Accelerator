#include "gpu_tensor.h"
#include "tensor.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <utility>
#include <vector>

// ============================================================================
// Helper: query free GPU memory
// ============================================================================

static size_t get_free_gpu_memory() {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    return free_bytes;
}

// ============================================================================
// RAII — Destructor Frees Memory
// ============================================================================

TEST(GpuTensorRAII, DestructorFreesMemory) {
    // Allocate a large-ish tensor, check that memory is reclaimed
    size_t free_before = get_free_gpu_memory();

    {
        // 1 million floats = 4 MB
        GpuTensor gpu(std::vector<int>{1000000});
        size_t free_during = get_free_gpu_memory();

        // Memory should decrease while tensor exists
        EXPECT_LT(free_during, free_before);
    }
    // After destruction, synchronize to ensure deallocation completes
    cudaDeviceSynchronize();

    size_t free_after = get_free_gpu_memory();

    // Memory should be reclaimed (within some margin for allocator overhead)
    // free_after should be close to free_before
    EXPECT_GT(free_after, free_before - 1024);  // Allow 1KB allocator overhead
}

TEST(GpuTensorRAII, MultipleAllocDealloc) {
    size_t free_baseline = get_free_gpu_memory();

    // Allocate and destroy several tensors in sequence
    for (int i = 0; i < 10; ++i) {
        GpuTensor gpu(std::vector<int>{100000});
        // tensor destroyed here each iteration
    }
    cudaDeviceSynchronize();

    size_t free_after = get_free_gpu_memory();

    // No cumulative leak — memory should be back to baseline
    EXPECT_GT(free_after, free_baseline - 1024);
}

TEST(GpuTensorRAII, ScopedLifetime) {
    // Verify tensor can be used within a scope and is cleaned up after
    Tensor result({0});  // placeholder

    {
        Tensor cpu_in({3}, {10.0f, 20.0f, 30.0f});
        GpuTensor gpu(cpu_in);
        result = gpu.download();
    }  // gpu destroyed here

    // Data should have been downloaded before destruction
    EXPECT_EQ(result.shape(), (std::vector<int>{3}));
    EXPECT_FLOAT_EQ(result[0], 10.0f);
    EXPECT_FLOAT_EQ(result[1], 20.0f);
    EXPECT_FLOAT_EQ(result[2], 30.0f);
}

// ============================================================================
// Move Semantics — Ownership Transfer
// ============================================================================

TEST(GpuTensorRAII, MoveConstructionTransfersOwnership) {
    Tensor cpu_data({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    GpuTensor original(cpu_data);

    EXPECT_TRUE(original.is_valid());
    const float* original_ptr = original.data();

    // Move construct — ownership transfers
    GpuTensor moved(std::move(original));

    EXPECT_FALSE(original.is_valid());   // Source invalidated
    EXPECT_TRUE(moved.is_valid());       // Destination owns memory
    EXPECT_EQ(moved.data(), original_ptr);  // Same device pointer

    // Data integrity preserved
    Tensor downloaded = moved.download();
    EXPECT_TRUE(downloaded == cpu_data);
}

TEST(GpuTensorRAII, MoveAssignmentFreesOldMemory) {
    size_t free_before = get_free_gpu_memory();

    GpuTensor gpu1(std::vector<int>{500000});  // 2 MB
    GpuTensor gpu2(std::vector<int>{100000});  // 0.4 MB

    size_t free_with_both = get_free_gpu_memory();

    // Move-assign gpu2 into gpu1: gpu1's old 2MB should be freed
    gpu1 = std::move(gpu2);

    cudaDeviceSynchronize();
    size_t free_after_move = get_free_gpu_memory();

    // After move, we should have more free memory than when both existed
    // (gpu1's original 2MB was freed, only 0.4MB remains)
    EXPECT_GT(free_after_move, free_with_both);

    EXPECT_FALSE(gpu2.is_valid());
    EXPECT_TRUE(gpu1.is_valid());
    EXPECT_EQ(gpu1.num_elements(), 100000);
}

TEST(GpuTensorRAII, SelfMoveAssignmentSafe) {
    // Self-move-assignment should not cause issues
    GpuTensor gpu(std::vector<int>{10});
    float* ptr = gpu.data();

    // Technically UB in C++, but our implementation handles it safely
    gpu = std::move(gpu);

    // Should still be valid (our impl checks this != &other)
    EXPECT_TRUE(gpu.is_valid());
    EXPECT_EQ(gpu.data(), ptr);
}

// ============================================================================
// Move Into Container (vector, etc.)
// ============================================================================

TEST(GpuTensorRAII, MoveIntoVector) {
    std::vector<GpuTensor> tensors;

    // Move tensors into a vector
    for (int i = 0; i < 5; ++i) {
        Tensor cpu({3}, {static_cast<float>(i), static_cast<float>(i + 1),
                         static_cast<float>(i + 2)});
        tensors.push_back(GpuTensor(cpu));
    }

    // Verify all tensors are valid and have correct data
    EXPECT_EQ(tensors.size(), 5u);
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(tensors[static_cast<size_t>(i)].is_valid());
        Tensor downloaded = tensors[static_cast<size_t>(i)].download();
        EXPECT_FLOAT_EQ(downloaded[0], static_cast<float>(i));
    }
}

TEST(GpuTensorRAII, VectorDestructionFreesAll) {
    size_t free_before = get_free_gpu_memory();

    {
        std::vector<GpuTensor> tensors;
        for (int i = 0; i < 10; ++i) {
            tensors.push_back(GpuTensor(std::vector<int>{100000}));
        }
        // All 10 tensors alive — ~4 MB allocated
        size_t free_during = get_free_gpu_memory();
        EXPECT_LT(free_during, free_before);
    }
    // Vector destroyed, all GpuTensors freed
    cudaDeviceSynchronize();

    size_t free_after = get_free_gpu_memory();
    EXPECT_GT(free_after, free_before - 1024);
}

// ============================================================================
// No Double-Free Tests
// ============================================================================

TEST(GpuTensorRAII, MovedFromTensorSafeToDestroy) {
    // After move, the source should be safely destroyable (no double-free)
    GpuTensor gpu(std::vector<int>{100});
    GpuTensor moved(std::move(gpu));

    // gpu is now invalid, but its destructor should not crash
    EXPECT_FALSE(gpu.is_valid());

    // moved should work fine
    Tensor downloaded = moved.download();
    EXPECT_EQ(downloaded.num_elements(), 100);
}

TEST(GpuTensorRAII, ChainedMoves) {
    Tensor cpu_data({3}, {7.0f, 8.0f, 9.0f});
    GpuTensor a(cpu_data);

    GpuTensor b(std::move(a));
    EXPECT_FALSE(a.is_valid());

    GpuTensor c(std::move(b));
    EXPECT_FALSE(b.is_valid());

    // Only c is valid, and it has the original data
    EXPECT_TRUE(c.is_valid());
    Tensor result = c.download();
    EXPECT_TRUE(result == cpu_data);
}
