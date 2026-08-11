#include "gpu_tensor.h"
#include "tensor.h"

#include <gtest/gtest.h>

#include <vector>

// ============================================================================
// GPU Memory Round-Trip Tests
// ============================================================================
// Verify that data survives a CPU → GPU → CPU transfer cycle intact.

TEST(GpuTensor, RoundTripSimple) {
    // Small tensor: upload → download → verify exact match
    Tensor cpu_in({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    GpuTensor gpu(cpu_in);
    Tensor cpu_out = gpu.download();

    EXPECT_TRUE(cpu_in == cpu_out);
}

TEST(GpuTensor, RoundTrip2D) {
    Tensor cpu_in({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    GpuTensor gpu(cpu_in);
    Tensor cpu_out = gpu.download();

    EXPECT_TRUE(cpu_in == cpu_out);
}

TEST(GpuTensor, RoundTrip4D_NCHW) {
    // Simulate a small image batch: (2, 1, 3, 3)
    std::vector<float> data(2 * 1 * 3 * 3);
    for (int i = 0; i < static_cast<int>(data.size()); ++i) {
        data[static_cast<size_t>(i)] = static_cast<float>(i) * 0.1f;
    }
    Tensor cpu_in({2, 1, 3, 3}, data);
    GpuTensor gpu(cpu_in);
    Tensor cpu_out = gpu.download();

    EXPECT_TRUE(cpu_in == cpu_out);
}

TEST(GpuTensor, RoundTripLarger) {
    // Realistic size: single MNIST image (1, 1, 28, 28)
    Tensor cpu_in = Tensor::rand({1, 1, 28, 28}, 42);
    GpuTensor gpu(cpu_in);
    Tensor cpu_out = gpu.download();

    EXPECT_TRUE(cpu_in == cpu_out);
}

TEST(GpuTensor, RoundTripNegativeValues) {
    Tensor cpu_in({5}, {-3.14f, -1.0f, 0.0f, 1.0f, 2.71f});
    GpuTensor gpu(cpu_in);
    Tensor cpu_out = gpu.download();

    EXPECT_TRUE(cpu_in == cpu_out);
}

TEST(GpuTensor, RoundTripAllZeros) {
    Tensor cpu_in = Tensor::zeros({3, 3});
    GpuTensor gpu(cpu_in);
    Tensor cpu_out = gpu.download();

    EXPECT_TRUE(cpu_in == cpu_out);
}

// ============================================================================
// Shape Preservation Tests
// ============================================================================

TEST(GpuTensor, ShapePreserved1D) {
    GpuTensor gpu(std::vector<int>{10});
    EXPECT_EQ(gpu.shape(), (std::vector<int>{10}));
    EXPECT_EQ(gpu.ndim(), 1);
    EXPECT_EQ(gpu.dim(0), 10);
    EXPECT_EQ(gpu.num_elements(), 10);
}

TEST(GpuTensor, ShapePreserved4D) {
    GpuTensor gpu(std::vector<int>{2, 8, 14, 14});
    EXPECT_EQ(gpu.shape(), (std::vector<int>{2, 8, 14, 14}));
    EXPECT_EQ(gpu.ndim(), 4);
    EXPECT_EQ(gpu.dim(0), 2);
    EXPECT_EQ(gpu.dim(1), 8);
    EXPECT_EQ(gpu.dim(2), 14);
    EXPECT_EQ(gpu.dim(3), 14);
    EXPECT_EQ(gpu.num_elements(), 2 * 8 * 14 * 14);
}

TEST(GpuTensor, ShapeFromCpuTensor) {
    Tensor cpu({3, 4, 5});
    GpuTensor gpu(cpu);
    EXPECT_EQ(gpu.shape(), cpu.shape());
    EXPECT_EQ(gpu.num_elements(), cpu.num_elements());
}

// ============================================================================
// Zero Initialization Test
// ============================================================================

TEST(GpuTensor, ShapeConstructorZeroInitialized) {
    // Construct from shape only → should be all zeros
    GpuTensor gpu(std::vector<int>{100});
    Tensor cpu = gpu.download();

    Tensor expected = Tensor::zeros({100});
    EXPECT_TRUE(cpu == expected);
}

// ============================================================================
// Upload Overwrite Test
// ============================================================================

TEST(GpuTensor, UploadOverwritesData) {
    Tensor first({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor second({4}, {10.0f, 20.0f, 30.0f, 40.0f});

    GpuTensor gpu(first);

    // Overwrite with second tensor
    gpu.upload(second);
    Tensor result = gpu.download();

    EXPECT_TRUE(result == second);
    EXPECT_FALSE(result == first);
}

TEST(GpuTensor, UploadMismatchThrows) {
    GpuTensor gpu(std::vector<int>{4});
    Tensor wrong_size({6});
    EXPECT_THROW(gpu.upload(wrong_size), std::invalid_argument);
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST(GpuTensor, MoveConstruction) {
    Tensor cpu_in({3}, {1.0f, 2.0f, 3.0f});
    GpuTensor original(cpu_in);

    // Move construct
    GpuTensor moved(std::move(original));

    // Moved-from should be invalid
    EXPECT_FALSE(original.is_valid());

    // Moved-to should contain the data
    EXPECT_TRUE(moved.is_valid());
    EXPECT_EQ(moved.shape(), (std::vector<int>{3}));

    Tensor cpu_out = moved.download();
    EXPECT_TRUE(cpu_out == cpu_in);
}

TEST(GpuTensor, MoveAssignment) {
    Tensor data1({2}, {1.0f, 2.0f});
    Tensor data2({3}, {10.0f, 20.0f, 30.0f});

    GpuTensor gpu1(data1);
    GpuTensor gpu2(data2);

    // Move assign gpu2 into gpu1 (gpu1's old memory should be freed)
    gpu1 = std::move(gpu2);

    EXPECT_FALSE(gpu2.is_valid());
    EXPECT_TRUE(gpu1.is_valid());
    EXPECT_EQ(gpu1.shape(), (std::vector<int>{3}));

    Tensor result = gpu1.download();
    EXPECT_TRUE(result == data2);
}

// ============================================================================
// SizeBytes Test
// ============================================================================

TEST(GpuTensor, SizeBytes) {
    GpuTensor gpu(std::vector<int>{10, 20});
    EXPECT_EQ(gpu.size_bytes(), 10u * 20u * sizeof(float));
}

// ============================================================================
// Validity Test
// ============================================================================

TEST(GpuTensor, IsValidAfterConstruction) {
    GpuTensor gpu(std::vector<int>{5});
    EXPECT_TRUE(gpu.is_valid());
    EXPECT_NE(gpu.data(), nullptr);
}
