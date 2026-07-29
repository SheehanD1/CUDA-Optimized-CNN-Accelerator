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

// ============================================================================
// Factory Method Tests — zeros, ones, full
// ============================================================================

TEST(TensorFactory, Zeros) {
    auto t = Tensor::zeros({2, 3, 4});
    EXPECT_EQ(t.shape(), (std::vector<int>{2, 3, 4}));
    EXPECT_EQ(t.num_elements(), 24);
    for (int i = 0; i < t.num_elements(); ++i) {
        EXPECT_FLOAT_EQ(t[i], 0.0f);
    }
}

TEST(TensorFactory, Ones) {
    auto t = Tensor::ones({5, 3});
    EXPECT_EQ(t.shape(), (std::vector<int>{5, 3}));
    for (int i = 0; i < t.num_elements(); ++i) {
        EXPECT_FLOAT_EQ(t[i], 1.0f);
    }
}

TEST(TensorFactory, Full) {
    auto t = Tensor::full({2, 2}, 3.14f);
    EXPECT_EQ(t.num_elements(), 4);
    for (int i = 0; i < t.num_elements(); ++i) {
        EXPECT_FLOAT_EQ(t[i], 3.14f);
    }
}

// ============================================================================
// Factory Method Tests — rand, randn
// ============================================================================

TEST(TensorFactory, RandShape) {
    auto t = Tensor::rand({1, 1, 28, 28}, 42);
    EXPECT_EQ(t.shape(), (std::vector<int>{1, 1, 28, 28}));
    EXPECT_EQ(t.num_elements(), 784);
}

TEST(TensorFactory, RandRange) {
    auto t = Tensor::rand({1000}, 42);
    for (int i = 0; i < t.num_elements(); ++i) {
        EXPECT_GE(t[i], 0.0f);
        EXPECT_LT(t[i], 1.0f);
    }
}

TEST(TensorFactory, RandReproducibleWithSeed) {
    auto t1 = Tensor::rand({10}, 42);
    auto t2 = Tensor::rand({10}, 42);
    for (int i = 0; i < t1.num_elements(); ++i) {
        EXPECT_FLOAT_EQ(t1[i], t2[i]);
    }
}

TEST(TensorFactory, RandDifferentSeeds) {
    auto t1 = Tensor::rand({100}, 1);
    auto t2 = Tensor::rand({100}, 2);
    // With different seeds, at least some values should differ
    bool any_different = false;
    for (int i = 0; i < t1.num_elements(); ++i) {
        if (t1[i] != t2[i]) {
            any_different = true;
            break;
        }
    }
    EXPECT_TRUE(any_different);
}

TEST(TensorFactory, RandnShape) {
    auto t = Tensor::randn({8, 1, 3, 3}, 0.0f, 0.1f, 42);
    EXPECT_EQ(t.shape(), (std::vector<int>{8, 1, 3, 3}));
    EXPECT_EQ(t.num_elements(), 72);
}

TEST(TensorFactory, RandnReproducible) {
    auto t1 = Tensor::randn({20}, 0.0f, 1.0f, 123);
    auto t2 = Tensor::randn({20}, 0.0f, 1.0f, 123);
    for (int i = 0; i < t1.num_elements(); ++i) {
        EXPECT_FLOAT_EQ(t1[i], t2[i]);
    }
}

TEST(TensorFactory, RandnMeanStddev) {
    // With enough samples, mean should be close to specified mean
    auto t = Tensor::randn({10000}, 5.0f, 0.5f, 42);
    float sum = 0.0f;
    for (int i = 0; i < t.num_elements(); ++i) {
        sum += t[i];
    }
    float mean = sum / static_cast<float>(t.num_elements());
    EXPECT_NEAR(mean, 5.0f, 0.05f);  // Within 0.05 of target mean
}

// ============================================================================
// Reshape Tests
// ============================================================================

TEST(TensorReshape, BasicReshape) {
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    Tensor t({2, 3}, data);
    Tensor reshaped = t.reshape({3, 2});
    EXPECT_EQ(reshaped.shape(), (std::vector<int>{3, 2}));
    EXPECT_EQ(reshaped.num_elements(), 6);
    // Data order is preserved
    EXPECT_FLOAT_EQ(reshaped[0], 1.0f);
    EXPECT_FLOAT_EQ(reshaped[5], 6.0f);
}

TEST(TensorReshape, InferDimension) {
    Tensor t({2, 3, 4});
    Tensor reshaped = t.reshape({-1, 4});
    EXPECT_EQ(reshaped.shape(), (std::vector<int>{6, 4}));
    EXPECT_EQ(reshaped.num_elements(), 24);
}

TEST(TensorReshape, InferFirstDimension) {
    Tensor t({24});
    Tensor reshaped = t.reshape({-1, 2, 3});
    EXPECT_EQ(reshaped.shape(), (std::vector<int>{4, 2, 3}));
}

TEST(TensorReshape, InferLastDimension) {
    Tensor t({2, 3, 4});
    Tensor reshaped = t.reshape({6, -1});
    EXPECT_EQ(reshaped.shape(), (std::vector<int>{6, 4}));
}

TEST(TensorReshape, PreservesData) {
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor t({2, 4}, data);
    Tensor reshaped = t.reshape({4, 2});
    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(reshaped[i], static_cast<float>(i + 1));
    }
}

TEST(TensorReshape, SameShape) {
    Tensor t({2, 3});
    Tensor reshaped = t.reshape({2, 3});
    EXPECT_EQ(reshaped.shape(), t.shape());
}

TEST(TensorReshape, MultipleNegOneThrows) {
    Tensor t({24});
    EXPECT_THROW(t.reshape({-1, -1, 4}), std::invalid_argument);
}

TEST(TensorReshape, MismatchedElementCountThrows) {
    Tensor t({2, 3});
    EXPECT_THROW(t.reshape({2, 4}), std::invalid_argument);
}

TEST(TensorReshape, ZeroDimensionThrows) {
    Tensor t({6});
    EXPECT_THROW(t.reshape({0, 6}), std::invalid_argument);
}

TEST(TensorReshape, IndivisibleInferThrows) {
    Tensor t({7});
    EXPECT_THROW(t.reshape({-1, 3}), std::invalid_argument);
}

TEST(TensorReshape, DoesNotModifyOriginal) {
    Tensor original({2, 3});
    original[0] = 42.0f;
    Tensor reshaped = original.reshape({3, 2});
    reshaped[0] = 99.0f;
    // Original should be unchanged (reshape copies data)
    EXPECT_FLOAT_EQ(original[0], 42.0f);
}

// ============================================================================
// Flatten Tests
// ============================================================================

TEST(TensorFlatten, NCHW) {
    Tensor t({2, 8, 14, 14});
    Tensor flat = t.flatten();
    EXPECT_EQ(flat.shape(), (std::vector<int>{2, 8 * 14 * 14}));
    EXPECT_EQ(flat.num_elements(), t.num_elements());
}

TEST(TensorFlatten, ThreeDimensional) {
    Tensor t({3, 4, 5});
    Tensor flat = t.flatten();
    EXPECT_EQ(flat.shape(), (std::vector<int>{3, 20}));
}

TEST(TensorFlatten, OneDimensional) {
    Tensor t({10});
    Tensor flat = t.flatten();
    EXPECT_EQ(flat.shape(), (std::vector<int>{1, 10}));
}

TEST(TensorFlatten, FlattenAll) {
    Tensor t({2, 3, 4});
    Tensor flat = t.flatten_all();
    EXPECT_EQ(flat.shape(), (std::vector<int>{24}));
    EXPECT_EQ(flat.ndim(), 1);
}

TEST(TensorFlatten, PreservesData) {
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    Tensor t({1, 2, 3}, data);
    Tensor flat = t.flatten();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(flat[i], static_cast<float>(i + 1));
    }
}

// ============================================================================
// Allclose Tests
// ============================================================================

TEST(TensorAllclose, IdenticalTensors) {
    auto t1 = Tensor::ones({3, 4});
    auto t2 = Tensor::ones({3, 4});
    EXPECT_TRUE(t1.allclose(t2));
}

TEST(TensorAllclose, WithinTolerance) {
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> data2 = {1.000001f, 2.000001f, 3.000001f};
    Tensor t1({3}, data1);
    Tensor t2({3}, data2);
    EXPECT_TRUE(t1.allclose(t2, 1e-5f));
}

TEST(TensorAllclose, OutsideTolerance) {
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> data2 = {1.0f, 2.0f, 3.1f};
    Tensor t1({3}, data1);
    Tensor t2({3}, data2);
    EXPECT_FALSE(t1.allclose(t2, 1e-5f));
}

TEST(TensorAllclose, ShapeMismatch) {
    auto t1 = Tensor::ones({2, 3});
    auto t2 = Tensor::ones({3, 2});
    EXPECT_FALSE(t1.allclose(t2));
}

TEST(TensorAllclose, CustomAtol) {
    std::vector<float> data1 = {1.0f, 2.0f};
    std::vector<float> data2 = {1.05f, 2.05f};
    Tensor t1({2}, data1);
    Tensor t2({2}, data2);
    EXPECT_FALSE(t1.allclose(t2, 0.01f));   // Too tight
    EXPECT_TRUE(t1.allclose(t2, 0.1f));     // Relaxed enough
}

TEST(TensorAllclose, RelativeTolerance) {
    // For large values, rtol matters more
    std::vector<float> data1 = {1000.0f};
    std::vector<float> data2 = {1000.01f};
    Tensor t1({1}, data1);
    Tensor t2({1}, data2);
    // atol=0 but rtol=1e-4 → tol = 0 + 1e-4 * 1000.01 = 0.100001
    EXPECT_TRUE(t1.allclose(t2, 0.0f, 1e-4f));
}

TEST(TensorAllclose, ZeroTensors) {
    auto t1 = Tensor::zeros({100});
    auto t2 = Tensor::zeros({100});
    EXPECT_TRUE(t1.allclose(t2));
}

TEST(TensorAllclose, Symmetric) {
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> data2 = {1.001f, 2.001f, 3.001f};
    Tensor t1({3}, data1);
    Tensor t2({3}, data2);
    // allclose should give same result in both directions
    EXPECT_EQ(t1.allclose(t2, 0.01f), t2.allclose(t1, 0.01f));
}

// ============================================================================
// MaxDiff Tests
// ============================================================================

TEST(TensorMaxDiff, IdenticalTensors) {
    auto t1 = Tensor::ones({3, 4});
    auto t2 = Tensor::ones({3, 4});
    EXPECT_FLOAT_EQ(t1.max_diff(t2), 0.0f);
}

TEST(TensorMaxDiff, KnownDifference) {
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> data2 = {1.0f, 2.5f, 3.0f};
    Tensor t1({3}, data1);
    Tensor t2({3}, data2);
    EXPECT_FLOAT_EQ(t1.max_diff(t2), 0.5f);
}

TEST(TensorMaxDiff, ShapeMismatchReturnsNegative) {
    auto t1 = Tensor::ones({2, 3});
    auto t2 = Tensor::ones({3, 2});
    EXPECT_FLOAT_EQ(t1.max_diff(t2), -1.0f);
}

// ============================================================================
// Equality Operator Tests
// ============================================================================

TEST(TensorEquality, EqualTensors) {
    std::vector<float> data = {1, 2, 3, 4};
    Tensor t1({2, 2}, data);
    Tensor t2({2, 2}, data);
    EXPECT_TRUE(t1 == t2);
    EXPECT_FALSE(t1 != t2);
}

TEST(TensorEquality, DifferentData) {
    Tensor t1 = Tensor::ones({3});
    Tensor t2 = Tensor::zeros({3});
    EXPECT_FALSE(t1 == t2);
    EXPECT_TRUE(t1 != t2);
}

TEST(TensorEquality, DifferentShape) {
    Tensor t1 = Tensor::zeros({2, 3});
    Tensor t2 = Tensor::zeros({3, 2});
    EXPECT_FALSE(t1 == t2);
}

// ============================================================================
// Argmax Tests
// ============================================================================

TEST(TensorArgmax, Simple) {
    Tensor t({5}, {1.0f, 3.0f, 2.0f, 5.0f, 4.0f});
    EXPECT_EQ(t.argmax(), 3);
}

TEST(TensorArgmax, FirstElement) {
    Tensor t({3}, {10.0f, 5.0f, 1.0f});
    EXPECT_EQ(t.argmax(), 0);
}

TEST(TensorArgmax, LastElement) {
    Tensor t({3}, {1.0f, 5.0f, 10.0f});
    EXPECT_EQ(t.argmax(), 2);
}

TEST(TensorArgmax, AllNegative) {
    Tensor t({4}, {-5.0f, -1.0f, -3.0f, -2.0f});
    EXPECT_EQ(t.argmax(), 1);  // -1 is the largest
}

TEST(TensorArgmax, SingleElement) {
    Tensor t({1}, {42.0f});
    EXPECT_EQ(t.argmax(), 0);
}

// ============================================================================
// MaxValue Tests
// ============================================================================

TEST(TensorMaxValue, Simple) {
    Tensor t({5}, {1.0f, 3.0f, 2.0f, 5.0f, 4.0f});
    EXPECT_FLOAT_EQ(t.max_value(), 5.0f);
}

TEST(TensorMaxValue, AllNegative) {
    Tensor t({3}, {-10.0f, -5.0f, -1.0f});
    EXPECT_FLOAT_EQ(t.max_value(), -1.0f);
}

// ============================================================================
// ArgmaxPerRow Tests
// ============================================================================

TEST(TensorArgmaxPerRow, SingleRow) {
    Tensor t({1, 4}, {0.1f, 0.7f, 0.1f, 0.1f});
    auto result = t.argmax_per_row();
    EXPECT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], 1);
}

TEST(TensorArgmaxPerRow, BatchOf3) {
    // 3 samples, 4 classes each
    Tensor t({3, 4}, {
        0.1f, 0.7f, 0.1f, 0.1f,  // max at index 1
        0.5f, 0.1f, 0.3f, 0.1f,  // max at index 0
        0.1f, 0.1f, 0.1f, 0.7f   // max at index 3
    });
    auto result = t.argmax_per_row();
    EXPECT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0], 1);
    EXPECT_EQ(result[1], 0);
    EXPECT_EQ(result[2], 3);
}

TEST(TensorArgmaxPerRow, TenClasses) {
    // Simulate softmax output for MNIST: class 7 has highest prob
    Tensor t({1, 10}, {0.01f, 0.01f, 0.02f, 0.01f, 0.01f,
                        0.01f, 0.02f, 0.85f, 0.03f, 0.03f});
    auto result = t.argmax_per_row();
    EXPECT_EQ(result[0], 7);
}

TEST(TensorArgmaxPerRow, NonTwoDimThrows) {
    Tensor t({2, 3, 4});
    EXPECT_THROW(t.argmax_per_row(), std::invalid_argument);
}
