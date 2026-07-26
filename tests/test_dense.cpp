#include "layers/dense.h"
#include "layers/flatten.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

using cnn::Tensor;
using cnn::dense_forward;
using cnn::flatten;

// ============================================================================
// Dense Forward — Basic Correctness Tests
// ============================================================================

TEST(DenseForward, SingleNeuron) {
    // Input: (1, 3) = [1, 2, 3]
    // Weights: (1, 3) = [1, 1, 1]
    // Bias: (1) = [0]
    // Output: 1*1 + 2*1 + 3*1 + 0 = 6
    Tensor input({1, 3}, {1.0f, 2.0f, 3.0f});
    Tensor weights({1, 3}, {1.0f, 1.0f, 1.0f});
    Tensor bias({1}, {0.0f});

    Tensor output = dense_forward(input, weights, bias);
    EXPECT_EQ(output.shape(), (std::vector<int>{1, 1}));
    EXPECT_FLOAT_EQ(output.at(0, 0), 6.0f);
}

TEST(DenseForward, SingleNeuronWithBias) {
    // Same as above but bias = 10
    // Output: 6 + 10 = 16
    Tensor input({1, 3}, {1.0f, 2.0f, 3.0f});
    Tensor weights({1, 3}, {1.0f, 1.0f, 1.0f});
    Tensor bias({1}, {10.0f});

    Tensor output = dense_forward(input, weights, bias);
    EXPECT_FLOAT_EQ(output.at(0, 0), 16.0f);
}

TEST(DenseForward, IdentityWeights) {
    // 3x3 identity weight matrix → output == input + bias
    Tensor input({1, 3}, {5.0f, 10.0f, 15.0f});
    Tensor weights({3, 3}, {1, 0, 0, 0, 1, 0, 0, 0, 1});
    Tensor bias({3}, {0.0f, 0.0f, 0.0f});

    Tensor output = dense_forward(input, weights, bias);
    EXPECT_EQ(output.shape(), (std::vector<int>{1, 3}));
    EXPECT_FLOAT_EQ(output.at(0, 0), 5.0f);
    EXPECT_FLOAT_EQ(output.at(0, 1), 10.0f);
    EXPECT_FLOAT_EQ(output.at(0, 2), 15.0f);
}

TEST(DenseForward, HandComputed) {
    // Input: (1, 2) = [2, 3]
    // Weights: (3, 2) = [[1, 0], [0, 1], [1, 1]]
    // Bias: (3) = [0.5, -0.5, 0]
    //
    // out[0] = 2*1 + 3*0 + 0.5 = 2.5
    // out[1] = 2*0 + 3*1 - 0.5 = 2.5
    // out[2] = 2*1 + 3*1 + 0   = 5.0
    Tensor input({1, 2}, {2.0f, 3.0f});
    Tensor weights({3, 2}, {1, 0, 0, 1, 1, 1});
    Tensor bias({3}, {0.5f, -0.5f, 0.0f});

    Tensor output = dense_forward(input, weights, bias);
    EXPECT_EQ(output.shape(), (std::vector<int>{1, 3}));
    EXPECT_FLOAT_EQ(output.at(0, 0), 2.5f);
    EXPECT_FLOAT_EQ(output.at(0, 1), 2.5f);
    EXPECT_FLOAT_EQ(output.at(0, 2), 5.0f);
}

// ============================================================================
// Dense Forward — Batch Tests
// ============================================================================

TEST(DenseForward, BatchSize2) {
    // Two samples, same weights
    // Sample 0: [1, 0] → out = [1*2 + 0*1 + 0] = [2]
    // Sample 1: [0, 1] → out = [0*2 + 1*1 + 0] = [1]
    Tensor input({2, 2}, {1, 0, 0, 1});
    Tensor weights({1, 2}, {2.0f, 1.0f});
    Tensor bias({1}, {0.0f});

    Tensor output = dense_forward(input, weights, bias);
    EXPECT_EQ(output.shape(), (std::vector<int>{2, 1}));
    EXPECT_FLOAT_EQ(output.at(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(output.at(1, 0), 1.0f);
}

TEST(DenseForward, BatchSize4) {
    // 4 samples, 2 in_features → 3 out_features
    Tensor input({4, 2}, {1, 0, 0, 1, 1, 1, 2, 3});
    Tensor weights({3, 2}, {1, 0, 0, 1, 1, 1});
    Tensor bias({3}, {0, 0, 0});

    Tensor output = dense_forward(input, weights, bias);
    EXPECT_EQ(output.shape(), (std::vector<int>{4, 3}));

    // Sample 0: [1,0] → [1, 0, 1]
    EXPECT_FLOAT_EQ(output.at(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(output.at(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(output.at(0, 2), 1.0f);

    // Sample 3: [2,3] → [2, 3, 5]
    EXPECT_FLOAT_EQ(output.at(3, 0), 2.0f);
    EXPECT_FLOAT_EQ(output.at(3, 1), 3.0f);
    EXPECT_FLOAT_EQ(output.at(3, 2), 5.0f);
}

// ============================================================================
// Dense Forward — Architecture-Relevant Dimensions
// ============================================================================

TEST(DenseForward, OutputDims400to120) {
    // First dense layer in our LeNet: 400 → 120
    Tensor input({1, 400});
    Tensor weights({120, 400});
    Tensor bias({120});

    Tensor output = dense_forward(input, weights, bias);
    EXPECT_EQ(output.shape(), (std::vector<int>{1, 120}));
}

TEST(DenseForward, OutputDims120to10) {
    // Second dense layer: 120 → 10 (class logits)
    Tensor input({1, 120});
    Tensor weights({10, 120});
    Tensor bias({10});

    Tensor output = dense_forward(input, weights, bias);
    EXPECT_EQ(output.shape(), (std::vector<int>{1, 10}));
}

// ============================================================================
// Dense Forward — Does Not Modify Input
// ============================================================================

TEST(DenseForward, DoesNotModifyInput) {
    Tensor input({1, 3}, {1, 2, 3});
    Tensor input_copy = input;
    Tensor weights({2, 3});
    Tensor bias({2});

    dense_forward(input, weights, bias);
    EXPECT_TRUE(input == input_copy);
}

// ============================================================================
// Dense Forward — Validation Tests
// ============================================================================

TEST(DenseValidation, Input3DThrows) {
    Tensor input({1, 3, 3});
    Tensor weights({2, 9});
    Tensor bias({2});
    EXPECT_THROW(dense_forward(input, weights, bias), std::invalid_argument);
}

TEST(DenseValidation, FeatureMismatchThrows) {
    Tensor input({1, 5});     // 5 features
    Tensor weights({3, 10});  // expects 10 features
    Tensor bias({3});
    EXPECT_THROW(dense_forward(input, weights, bias), std::invalid_argument);
}

TEST(DenseValidation, BiasSizeMismatchThrows) {
    Tensor input({1, 5});
    Tensor weights({3, 5});   // 3 output features
    Tensor bias({4});         // bias has 4 (should be 3)
    EXPECT_THROW(dense_forward(input, weights, bias), std::invalid_argument);
}

TEST(DenseValidation, Weights3DThrows) {
    Tensor input({1, 5});
    Tensor weights({3, 5, 1});
    Tensor bias({3});
    EXPECT_THROW(dense_forward(input, weights, bias), std::invalid_argument);
}

// ============================================================================
// Dense + Flatten Integration Test
// ============================================================================

TEST(DenseFlatten, ConvOutputToClassLogits) {
    // Simulate: conv output (1, 2, 3, 3) → flatten → (1, 18) → dense → (1, 5)
    Tensor conv_output({1, 2, 3, 3});
    conv_output.fill(1.0f);

    // Flatten: (1, 2, 3, 3) → (1, 18)
    Tensor flat = flatten(conv_output);
    EXPECT_EQ(flat.shape(), (std::vector<int>{1, 18}));

    // Dense: (1, 18) → (1, 5)
    Tensor weights = Tensor::ones({5, 18});
    Tensor bias({5}, {0, 0, 0, 0, 0});

    Tensor logits = dense_forward(flat, weights, bias);
    EXPECT_EQ(logits.shape(), (std::vector<int>{1, 5}));

    // Each output = sum of 18 ones = 18
    for (int j = 0; j < 5; ++j) {
        EXPECT_FLOAT_EQ(logits.at(0, j), 18.0f);
    }
}

// ============================================================================
// Dense — NumPy Reference Test
// ============================================================================

TEST(DenseReference, RandomWeights) {
    // Hand-computed with known values
    // Input: [1, 2, 3, 4]
    // Weights: [[0.1, 0.2, 0.3, 0.4],
    //           [0.5, 0.6, 0.7, 0.8]]
    // Bias: [0.1, 0.2]
    // out[0] = 0.1*1 + 0.2*2 + 0.3*3 + 0.4*4 + 0.1 = 0.1+0.4+0.9+1.6+0.1 = 3.1
    // out[1] = 0.5*1 + 0.6*2 + 0.7*3 + 0.8*4 + 0.2 = 0.5+1.2+2.1+3.2+0.2 = 7.2
    Tensor input({1, 4}, {1, 2, 3, 4});
    Tensor weights({2, 4}, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f});
    Tensor bias({2}, {0.1f, 0.2f});

    Tensor output = dense_forward(input, weights, bias);
    EXPECT_NEAR(output.at(0, 0), 3.1f, 1e-5f);
    EXPECT_NEAR(output.at(0, 1), 7.2f, 1e-5f);
}
