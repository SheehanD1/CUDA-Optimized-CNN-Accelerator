#include "tensor.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

using cnn::Tensor;

// ============================================================================
// Construction Tests
// ============================================================================

TEST(TensorConstruction, DefaultConstructor) {
    Tensor t;
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.num_elements(), 0);
    EXPECT_EQ(t.ndim(), 0);
}

TEST(TensorConstruction, FromShapeVector) {
    Tensor t(std::vector<int>{2, 3, 4});
    EXPECT_EQ(t.ndim(), 3);
    EXPECT_EQ(t.num_elements(), 24);
    EXPECT_EQ(t.shape(), (std::vector<int>{2, 3, 4}));
    EXPECT_FALSE(t.empty());
}

TEST(TensorConstruction, FromInitializerList) {
    Tensor t({1, 8, 28, 28});
    EXPECT_EQ(t.ndim(), 4);
    EXPECT_EQ(t.num_elements(), 1 * 8 * 28 * 28);
    EXPECT_EQ(t.shape(), (std::vector<int>{1, 8, 28, 28}));
}

TEST(TensorConstruction, ZeroInitialized) {
    Tensor t({2, 3});
    for (int i = 0; i < t.num_elements(); ++i) {
        EXPECT_FLOAT_EQ(t[i], 0.0f);
    }
}

TEST(TensorConstruction, FromShapeAndData) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor t({2, 3}, data);
    EXPECT_EQ(t.num_elements(), 6);
    EXPECT_FLOAT_EQ(t[0], 1.0f);
    EXPECT_FLOAT_EQ(t[5], 6.0f);
}

TEST(TensorConstruction, MismatchedDataThrows) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    EXPECT_THROW(Tensor({2, 3}, data), std::invalid_argument);
}

TEST(TensorConstruction, NegativeDimensionThrows) {
    EXPECT_THROW(Tensor({2, -1, 3}), std::invalid_argument);
}

TEST(TensorConstruction, ZeroDimensionThrows) {
    EXPECT_THROW(Tensor({2, 0, 3}), std::invalid_argument);
}

TEST(TensorConstruction, OneDimensional) {
    Tensor t({10});
    EXPECT_EQ(t.ndim(), 1);
    EXPECT_EQ(t.num_elements(), 10);
    EXPECT_EQ(t.dim(0), 10);
}

TEST(TensorConstruction, CopyConstruction) {
    Tensor original({2, 3});
    original[0] = 42.0f;
    Tensor copy = original;
    EXPECT_FLOAT_EQ(copy[0], 42.0f);

    // Verify deep copy — modifying copy doesn't affect original
    copy[0] = 99.0f;
    EXPECT_FLOAT_EQ(original[0], 42.0f);
    EXPECT_FLOAT_EQ(copy[0], 99.0f);
}

TEST(TensorConstruction, MoveConstruction) {
    Tensor original({2, 3});
    original[0] = 42.0f;
    Tensor moved = std::move(original);
    EXPECT_FLOAT_EQ(moved[0], 42.0f);
    EXPECT_EQ(moved.num_elements(), 6);
}

// ============================================================================
// Strides Tests
// ============================================================================

TEST(TensorStrides, NCHW4D) {
    Tensor t({2, 3, 4, 5});
    auto strides = t.strides();
    EXPECT_EQ(strides[3], 1);          // W stride
    EXPECT_EQ(strides[2], 5);          // H stride = W
    EXPECT_EQ(strides[1], 4 * 5);     // C stride = H*W
    EXPECT_EQ(strides[0], 3 * 4 * 5); // N stride = C*H*W
}

TEST(TensorStrides, TwoDimensional) {
    Tensor t({3, 4});
    auto strides = t.strides();
    EXPECT_EQ(strides[0], 4);
    EXPECT_EQ(strides[1], 1);
}

// ============================================================================
// Dimension Query Tests
// ============================================================================

TEST(TensorDim, PositiveIndex) {
    Tensor t({2, 3, 4, 5});
    EXPECT_EQ(t.dim(0), 2);
    EXPECT_EQ(t.dim(1), 3);
    EXPECT_EQ(t.dim(2), 4);
    EXPECT_EQ(t.dim(3), 5);
}

TEST(TensorDim, NegativeIndex) {
    Tensor t({2, 3, 4, 5});
    EXPECT_EQ(t.dim(-1), 5);
    EXPECT_EQ(t.dim(-2), 4);
    EXPECT_EQ(t.dim(-3), 3);
    EXPECT_EQ(t.dim(-4), 2);
}

TEST(TensorDim, OutOfRangeThrows) {
    Tensor t({2, 3});
    EXPECT_THROW(t.dim(2), std::out_of_range);
    EXPECT_THROW(t.dim(-3), std::out_of_range);
}

// ============================================================================
// Element Access Tests — 4D NCHW
// ============================================================================

TEST(TensorAccess4D, ReadWrite) {
    Tensor t({1, 2, 3, 4});
    t.at(0, 0, 0, 0) = 1.0f;
    t.at(0, 1, 2, 3) = 99.0f;
    EXPECT_FLOAT_EQ(t.at(0, 0, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(t.at(0, 1, 2, 3), 99.0f);
}

TEST(TensorAccess4D, NCHWLayout) {
    // Verify NCHW memory layout: W varies fastest
    Tensor t({1, 1, 2, 3});
    // Set values using at()
    float val = 0.0f;
    for (int h = 0; h < 2; ++h) {
        for (int w = 0; w < 3; ++w) {
            t.at(0, 0, h, w) = val;
            val += 1.0f;
        }
    }
    // Verify flat memory is row-major (W varies fastest)
    EXPECT_FLOAT_EQ(t[0], 0.0f);  // (0,0,0,0)
    EXPECT_FLOAT_EQ(t[1], 1.0f);  // (0,0,0,1)
    EXPECT_FLOAT_EQ(t[2], 2.0f);  // (0,0,0,2)
    EXPECT_FLOAT_EQ(t[3], 3.0f);  // (0,0,1,0)
    EXPECT_FLOAT_EQ(t[4], 4.0f);  // (0,0,1,1)
    EXPECT_FLOAT_EQ(t[5], 5.0f);  // (0,0,1,2)
}

TEST(TensorAccess4D, MultiChannel) {
    Tensor t({1, 2, 2, 2});
    t.at(0, 0, 0, 0) = 10.0f;
    t.at(0, 1, 0, 0) = 20.0f;
    // Channel 0 data is in first 4 elements, channel 1 in next 4
    EXPECT_FLOAT_EQ(t[0], 10.0f);  // C=0, H=0, W=0
    EXPECT_FLOAT_EQ(t[4], 20.0f);  // C=1, H=0, W=0
}

// ============================================================================
// Element Access Tests — 2D
// ============================================================================

TEST(TensorAccess2D, ReadWrite) {
    Tensor t({3, 4});
    t.at(0, 0) = 1.0f;
    t.at(2, 3) = 99.0f;
    EXPECT_FLOAT_EQ(t.at(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(t.at(2, 3), 99.0f);
}

TEST(TensorAccess2D, RowMajorLayout) {
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    Tensor t({2, 3}, data);
    EXPECT_FLOAT_EQ(t.at(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(t.at(0, 2), 3.0f);
    EXPECT_FLOAT_EQ(t.at(1, 0), 4.0f);
    EXPECT_FLOAT_EQ(t.at(1, 2), 6.0f);
}

// ============================================================================
// Element Access Tests — Generic (vector indices)
// ============================================================================

TEST(TensorAccessGeneric, ThreeDimensional) {
    Tensor t({2, 3, 4});
    t.at(std::vector<int>{1, 2, 3}) = 42.0f;
    EXPECT_FLOAT_EQ(t.at(std::vector<int>{1, 2, 3}), 42.0f);
}

// ============================================================================
// Element Access Tests — Flat
// ============================================================================

TEST(TensorAccessFlat, ReadWrite) {
    Tensor t({6});
    t[0] = 10.0f;
    t[5] = 50.0f;
    EXPECT_FLOAT_EQ(t[0], 10.0f);
    EXPECT_FLOAT_EQ(t[5], 50.0f);
}

TEST(TensorAccessFlat, ConstAccess) {
    std::vector<float> data = {1, 2, 3, 4};
    const Tensor t({4}, data);
    EXPECT_FLOAT_EQ(t[0], 1.0f);
    EXPECT_FLOAT_EQ(t[3], 4.0f);
}

// ============================================================================
// NCHW Convenience Accessors
// ============================================================================

TEST(TensorNCHW, ConvenienceAccessors) {
    Tensor t({2, 8, 14, 14});
    EXPECT_EQ(t.batch_size(), 2);
    EXPECT_EQ(t.channels(), 8);
    EXPECT_EQ(t.height(), 14);
    EXPECT_EQ(t.width(), 14);
}

// ============================================================================
// Fill and Zero Tests
// ============================================================================

TEST(TensorFill, FillValue) {
    Tensor t({3, 4});
    t.fill(7.5f);
    for (int i = 0; i < t.num_elements(); ++i) {
        EXPECT_FLOAT_EQ(t[i], 7.5f);
    }
}

TEST(TensorFill, Zero) {
    Tensor t({2, 3});
    t.fill(42.0f);
    t.zero();
    for (int i = 0; i < t.num_elements(); ++i) {
        EXPECT_FLOAT_EQ(t[i], 0.0f);
    }
}

// ============================================================================
// Data Pointer Tests
// ============================================================================

TEST(TensorData, RawPointer) {
    Tensor t({4});
    float* ptr = t.data();
    ptr[0] = 100.0f;
    EXPECT_FLOAT_EQ(t[0], 100.0f);
}

TEST(TensorData, ConstRawPointer) {
    std::vector<float> data = {1, 2, 3};
    const Tensor t({3}, data);
    const float* ptr = t.data();
    EXPECT_FLOAT_EQ(ptr[0], 1.0f);
    EXPECT_FLOAT_EQ(ptr[2], 3.0f);
}

// ============================================================================
// Debug Printing Tests
// ============================================================================

TEST(TensorPrint, ShapeString) {
    Tensor t({1, 8, 28, 28});
    EXPECT_EQ(t.shape_string(), "(1, 8, 28, 28)");
}

TEST(TensorPrint, ShapeStringOneDim) {
    Tensor t({10});
    EXPECT_EQ(t.shape_string(), "(10)");
}

TEST(TensorPrint, ToStringContainsShape) {
    Tensor t({2, 3});
    std::string s = t.to_string();
    EXPECT_NE(s.find("(2, 3)"), std::string::npos);
    EXPECT_NE(s.find("6 elements"), std::string::npos);
}

TEST(TensorPrint, ToStringTruncation) {
    // Create a tensor with 100 elements — should be truncated with max_elements=10
    Tensor t({100});
    std::string s = t.to_string(10);
    EXPECT_NE(s.find("..."), std::string::npos);
}

TEST(TensorPrint, ToStringNoTruncation) {
    // Small tensor should not be truncated
    Tensor t({4});
    std::string s = t.to_string(10);
    EXPECT_EQ(s.find("..."), std::string::npos);
}
