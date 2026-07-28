#include "layers/softmax.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

using cnn::Tensor;
using cnn::softmax_forward;

// ============================================================================
// Softmax — Basic Correctness Tests
// ============================================================================

TEST(Softmax, UniformInput) {
    // Equal logits → uniform distribution
    Tensor input({1, 4}, {1.0f, 1.0f, 1.0f, 1.0f});
    Tensor output = softmax_forward(input);

    EXPECT_EQ(output.shape(), (std::vector<int>{1, 4}));
    for (int j = 0; j < 4; ++j) {
        EXPECT_NEAR(output.at(0, j), 0.25f, 1e-6f);
    }
}

TEST(Softmax, OneDominant) {
    // One large logit → that class gets nearly all probability
    Tensor input({1, 3}, {10.0f, 0.0f, 0.0f});
    Tensor output = softmax_forward(input);

    EXPECT_GT(output.at(0, 0), 0.99f);
    EXPECT_LT(output.at(0, 1), 0.01f);
    EXPECT_LT(output.at(0, 2), 0.01f);
}

TEST(Softmax, TwoClasses) {
    // softmax([0, 0]) = [0.5, 0.5]
    Tensor input({1, 2}, {0.0f, 0.0f});
    Tensor output = softmax_forward(input);

    EXPECT_NEAR(output.at(0, 0), 0.5f, 1e-6f);
    EXPECT_NEAR(output.at(0, 1), 0.5f, 1e-6f);
}

TEST(Softmax, HandComputed) {
    // softmax([1, 2, 3])
    // exp(1-3) = exp(-2), exp(2-3) = exp(-1), exp(3-3) = exp(0)
    // = [0.13534, 0.36788, 1.0]
    // sum = 1.50321
    // = [0.09003, 0.24473, 0.66524]
    Tensor input({1, 3}, {1.0f, 2.0f, 3.0f});
    Tensor output = softmax_forward(input);

    EXPECT_NEAR(output.at(0, 0), 0.09003f, 1e-4f);
    EXPECT_NEAR(output.at(0, 1), 0.24473f, 1e-4f);
    EXPECT_NEAR(output.at(0, 2), 0.66524f, 1e-4f);
}

// ============================================================================
// Softmax — Sum-to-One Property
// ============================================================================

TEST(Softmax, SumsToOne) {
    Tensor input({1, 10}, {0.1f, -0.5f, 1.2f, 3.0f, -2.0f,
                            0.5f, 0.0f, -1.0f, 2.5f, 1.0f});
    Tensor output = softmax_forward(input);

    float sum = 0.0f;
    for (int j = 0; j < 10; ++j) {
        sum += output.at(0, j);
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(Softmax, AllOutputsNonNegative) {
    Tensor input({1, 5}, {-100.0f, -50.0f, 0.0f, 50.0f, 100.0f});
    Tensor output = softmax_forward(input);

    for (int j = 0; j < 5; ++j) {
        EXPECT_GE(output.at(0, j), 0.0f);
    }
}

TEST(Softmax, AllOutputsLeOne) {
    Tensor input({1, 5}, {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor output = softmax_forward(input);

    for (int j = 0; j < 5; ++j) {
        EXPECT_LE(output.at(0, j), 1.0f);
    }
}

// ============================================================================
// Softmax — Numerical Stability
// ============================================================================

TEST(Softmax, LargePositiveValues) {
    // Without max subtraction, exp(1000) would overflow
    Tensor input({1, 3}, {1000.0f, 1000.0f, 1000.0f});
    Tensor output = softmax_forward(input);

    // Should still produce valid uniform distribution
    for (int j = 0; j < 3; ++j) {
        EXPECT_NEAR(output.at(0, j), 1.0f / 3.0f, 1e-5f);
    }
}

TEST(Softmax, LargeNegativeValues) {
    Tensor input({1, 3}, {-1000.0f, -999.0f, -1000.0f});
    Tensor output = softmax_forward(input);

    // Middle value is largest → gets most probability
    float sum = 0.0f;
    for (int j = 0; j < 3; ++j) {
        EXPECT_GE(output.at(0, j), 0.0f);
        sum += output.at(0, j);
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
    EXPECT_GT(output.at(0, 1), output.at(0, 0));
}

TEST(Softmax, MixedExtremeValues) {
    // One very large, others very small
    Tensor input({1, 4}, {-500.0f, 500.0f, -500.0f, -500.0f});
    Tensor output = softmax_forward(input);

    // Class 1 should get essentially all probability
    EXPECT_NEAR(output.at(0, 1), 1.0f, 1e-5f);
    float sum = 0.0f;
    for (int j = 0; j < 4; ++j) sum += output.at(0, j);
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

// ============================================================================
// Softmax — Batch Tests
// ============================================================================

TEST(Softmax, BatchSize2) {
    // Two samples, each should independently sum to 1
    Tensor input({2, 3}, {1, 2, 3, 10, 0, 0});
    Tensor output = softmax_forward(input);

    EXPECT_EQ(output.shape(), (std::vector<int>{2, 3}));

    // Sample 0 sums to 1
    float sum0 = output.at(0, 0) + output.at(0, 1) + output.at(0, 2);
    EXPECT_NEAR(sum0, 1.0f, 1e-5f);

    // Sample 1 sums to 1
    float sum1 = output.at(1, 0) + output.at(1, 1) + output.at(1, 2);
    EXPECT_NEAR(sum1, 1.0f, 1e-5f);

    // Sample 1: class 0 has highest logit
    EXPECT_GT(output.at(1, 0), 0.99f);
}

TEST(Softmax, BatchIndependence) {
    // Adding a second sample should not affect the first
    Tensor single({1, 3}, {1.0f, 2.0f, 3.0f});
    Tensor batch({2, 3}, {1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f});

    Tensor out_single = softmax_forward(single);
    Tensor out_batch = softmax_forward(batch);

    for (int j = 0; j < 3; ++j) {
        EXPECT_NEAR(out_single.at(0, j), out_batch.at(0, j), 1e-6f);
    }
}

// ============================================================================
// Softmax — 10-Class (MNIST) Test
// ============================================================================

TEST(Softmax, TenClasses) {
    // Simulate final layer output for MNIST
    Tensor input({1, 10}, {-1.5f, 0.2f, 3.1f, -0.5f, 0.8f,
                            1.2f, -2.0f, 0.5f, 4.0f, 0.1f});
    Tensor output = softmax_forward(input);

    EXPECT_EQ(output.shape(), (std::vector<int>{1, 10}));

    // Sum to 1
    float sum = 0.0f;
    for (int j = 0; j < 10; ++j) {
        EXPECT_GE(output.at(0, j), 0.0f);
        EXPECT_LE(output.at(0, j), 1.0f);
        sum += output.at(0, j);
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    // Class 8 (logit=4.0) should have highest probability
    int argmax = 0;
    for (int j = 1; j < 10; ++j) {
        if (output.at(0, j) > output.at(0, argmax)) argmax = j;
    }
    EXPECT_EQ(argmax, 8);
}

// ============================================================================
// Softmax — Does Not Modify Input
// ============================================================================

TEST(Softmax, DoesNotModifyInput) {
    Tensor input({1, 3}, {1.0f, 2.0f, 3.0f});
    Tensor input_copy = input;
    softmax_forward(input);
    EXPECT_TRUE(input == input_copy);
}

// ============================================================================
// Softmax — Validation Tests
// ============================================================================

TEST(SoftmaxValidation, Input1DThrows) {
    Tensor input({5});
    EXPECT_THROW(softmax_forward(input), std::invalid_argument);
}

TEST(SoftmaxValidation, Input3DThrows) {
    Tensor input({1, 3, 3});
    EXPECT_THROW(softmax_forward(input), std::invalid_argument);
}
