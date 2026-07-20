#include "layers/relu.h"

#include <gtest/gtest.h>

#include <vector>

using cnn::Tensor;
using cnn::relu_forward;
using cnn::relu_forward_inplace;

// ============================================================================
// ReLU Forward — Basic Tests
// ============================================================================

TEST(ReLUForward, AllPositive) {
    Tensor input({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor output = relu_forward(input);
    EXPECT_EQ(output.shape(), input.shape());
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(output[i], input[i]);
    }
}

TEST(ReLUForward, AllNegative) {
    Tensor input({4}, {-1.0f, -2.0f, -3.0f, -4.0f});
    Tensor output = relu_forward(input);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(output[i], 0.0f);
    }
}

TEST(ReLUForward, MixedValues) {
    Tensor input({6}, {-3.0f, -1.0f, 0.0f, 0.5f, 2.0f, -0.1f});
    Tensor output = relu_forward(input);
    EXPECT_FLOAT_EQ(output[0], 0.0f);
    EXPECT_FLOAT_EQ(output[1], 0.0f);
    EXPECT_FLOAT_EQ(output[2], 0.0f);
    EXPECT_FLOAT_EQ(output[3], 0.5f);
    EXPECT_FLOAT_EQ(output[4], 2.0f);
    EXPECT_FLOAT_EQ(output[5], 0.0f);
}

TEST(ReLUForward, ZeroInput) {
    Tensor input({3}, {0.0f, 0.0f, 0.0f});
    Tensor output = relu_forward(input);
    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(output[i], 0.0f);
    }
}

TEST(ReLUForward, SingleElement) {
    Tensor pos({1}, {5.0f});
    EXPECT_FLOAT_EQ(relu_forward(pos)[0], 5.0f);

    Tensor neg({1}, {-5.0f});
    EXPECT_FLOAT_EQ(relu_forward(neg)[0], 0.0f);
}

// ============================================================================
// ReLU Forward — Shape Preservation
// ============================================================================

TEST(ReLUForward, PreservesShape4D) {
    Tensor input({1, 8, 28, 28});
    Tensor output = relu_forward(input);
    EXPECT_EQ(output.shape(), (std::vector<int>{1, 8, 28, 28}));
}

TEST(ReLUForward, PreservesShape2D) {
    Tensor input({120, 10});
    Tensor output = relu_forward(input);
    EXPECT_EQ(output.shape(), (std::vector<int>{120, 10}));
}

TEST(ReLUForward, PreservesShape1D) {
    Tensor input({100});
    Tensor output = relu_forward(input);
    EXPECT_EQ(output.shape(), (std::vector<int>{100}));
}

// ============================================================================
// ReLU Forward — Does Not Modify Input
// ============================================================================

TEST(ReLUForward, DoesNotModifyInput) {
    Tensor input({4}, {-1.0f, 2.0f, -3.0f, 4.0f});
    Tensor output = relu_forward(input);

    // Input should be unchanged
    EXPECT_FLOAT_EQ(input[0], -1.0f);
    EXPECT_FLOAT_EQ(input[2], -3.0f);
}

// ============================================================================
// ReLU Forward — NCHW Tensor
// ============================================================================

TEST(ReLUForward, NCHWTensor) {
    // Simulate a post-convolution output with mixed positive/negative values
    Tensor input({1, 2, 2, 2}, {-0.5f, 1.2f, -3.0f, 0.0f,
                                  2.5f, -0.1f, 0.8f, -1.0f});
    Tensor output = relu_forward(input);
    EXPECT_EQ(output.shape(), (std::vector<int>{1, 2, 2, 2}));

    // Channel 0
    EXPECT_FLOAT_EQ(output.at(0, 0, 0, 0), 0.0f);
    EXPECT_FLOAT_EQ(output.at(0, 0, 0, 1), 1.2f);
    EXPECT_FLOAT_EQ(output.at(0, 0, 1, 0), 0.0f);
    EXPECT_FLOAT_EQ(output.at(0, 0, 1, 1), 0.0f);

    // Channel 1
    EXPECT_FLOAT_EQ(output.at(0, 1, 0, 0), 2.5f);
    EXPECT_FLOAT_EQ(output.at(0, 1, 0, 1), 0.0f);
    EXPECT_FLOAT_EQ(output.at(0, 1, 1, 0), 0.8f);
    EXPECT_FLOAT_EQ(output.at(0, 1, 1, 1), 0.0f);
}

// ============================================================================
// ReLU In-Place Tests
// ============================================================================

TEST(ReLUInPlace, ModifiesInput) {
    Tensor input({6}, {-3.0f, -1.0f, 0.0f, 0.5f, 2.0f, -0.1f});
    relu_forward_inplace(input);

    EXPECT_FLOAT_EQ(input[0], 0.0f);
    EXPECT_FLOAT_EQ(input[1], 0.0f);
    EXPECT_FLOAT_EQ(input[2], 0.0f);
    EXPECT_FLOAT_EQ(input[3], 0.5f);
    EXPECT_FLOAT_EQ(input[4], 2.0f);
    EXPECT_FLOAT_EQ(input[5], 0.0f);
}

TEST(ReLUInPlace, AllPositiveUnchanged) {
    Tensor input({3}, {1.0f, 2.0f, 3.0f});
    relu_forward_inplace(input);
    EXPECT_FLOAT_EQ(input[0], 1.0f);
    EXPECT_FLOAT_EQ(input[1], 2.0f);
    EXPECT_FLOAT_EQ(input[2], 3.0f);
}

TEST(ReLUInPlace, MatchesForward) {
    // In-place and allocating versions should produce identical results
    Tensor original({8}, {-2.0f, 3.0f, -1.0f, 0.0f, 5.0f, -0.5f, 1.0f, -4.0f});
    Tensor copy = original;

    Tensor output = relu_forward(original);
    relu_forward_inplace(copy);

    EXPECT_TRUE(output == copy);
}

// ============================================================================
// ReLU — Large Tensor Test
// ============================================================================

TEST(ReLUForward, LargeTensor) {
    // Test with a realistic-sized tensor
    auto input = Tensor::randn({1, 16, 14, 14}, 0.0f, 1.0f, 42);
    Tensor output = relu_forward(input);

    EXPECT_EQ(output.shape(), input.shape());

    // All output values should be >= 0
    for (int i = 0; i < output.num_elements(); ++i) {
        EXPECT_GE(output[i], 0.0f) << "Negative value at index " << i;
    }

    // Positive input values should be unchanged
    for (int i = 0; i < input.num_elements(); ++i) {
        if (input[i] > 0.0f) {
            EXPECT_FLOAT_EQ(output[i], input[i]);
        }
    }
}
