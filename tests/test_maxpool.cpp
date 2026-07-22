#include "layers/maxpool2d.h"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <vector>

using cnn::Tensor;
using cnn::maxpool2d_forward;

// ============================================================================
// MaxPool2D — Basic Correctness Tests
// ============================================================================

TEST(MaxPool2D, Simple2x2Pool) {
    // Input: (1, 1, 4, 4) — values 1..16
    //   1  2  3  4
    //   5  6  7  8
    //   9 10 11 12
    //  13 14 15 16
    //
    // Pool 2x2, stride 2 → (1, 1, 2, 2)
    // Window (0,0): max(1,2,5,6)  = 6
    // Window (0,1): max(3,4,7,8)  = 8
    // Window (1,0): max(9,10,13,14) = 14
    // Window (1,1): max(11,12,15,16) = 16
    Tensor input({1, 1, 4, 4},
                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    Tensor output = maxpool2d_forward(input, 2);

    EXPECT_EQ(output.shape(), (std::vector<int>{1, 1, 2, 2}));
    EXPECT_FLOAT_EQ(output.at(0, 0, 0, 0), 6.0f);
    EXPECT_FLOAT_EQ(output.at(0, 0, 0, 1), 8.0f);
    EXPECT_FLOAT_EQ(output.at(0, 0, 1, 0), 14.0f);
    EXPECT_FLOAT_EQ(output.at(0, 0, 1, 1), 16.0f);
}

TEST(MaxPool2D, Simple3x3Pool) {
    // Input: (1, 1, 3, 3) — values 1..9
    // Pool 3x3, stride 3 → (1, 1, 1, 1)
    // max(1..9) = 9
    Tensor input({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor output = maxpool2d_forward(input, 3);

    EXPECT_EQ(output.shape(), (std::vector<int>{1, 1, 1, 1}));
    EXPECT_FLOAT_EQ(output.at(0, 0, 0, 0), 9.0f);
}

TEST(MaxPool2D, PoolSize1Identity) {
    // Pool 1x1 with stride 1 should be identity
    Tensor input({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor output = maxpool2d_forward(input, 1, 1);

    EXPECT_EQ(output.shape(), input.shape());
    for (int i = 0; i < input.num_elements(); ++i) {
        EXPECT_FLOAT_EQ(output[i], input[i]);
    }
}

// ============================================================================
// MaxPool2D — Negative Values
// ============================================================================

TEST(MaxPool2D, AllNegativeValues) {
    // MaxPool should correctly find the max even when all values are negative
    Tensor input({1, 1, 2, 2}, {-4.0f, -3.0f, -2.0f, -1.0f});
    Tensor output = maxpool2d_forward(input, 2);

    EXPECT_EQ(output.shape(), (std::vector<int>{1, 1, 1, 1}));
    EXPECT_FLOAT_EQ(output.at(0, 0, 0, 0), -1.0f);
}

TEST(MaxPool2D, MixedPositiveNegative) {
    Tensor input({1, 1, 2, 2}, {-5.0f, 3.0f, 1.0f, -2.0f});
    Tensor output = maxpool2d_forward(input, 2);

    EXPECT_FLOAT_EQ(output.at(0, 0, 0, 0), 3.0f);
}

// ============================================================================
// MaxPool2D — Multi-Channel
// ============================================================================

TEST(MaxPool2D, MultiChannel) {
    // Input: (1, 2, 4, 4) — two channels, pool 2x2
    // Channel 0: 1..16, Channel 1: 17..32
    std::vector<float> data;
    for (int i = 1; i <= 32; ++i) {
        data.push_back(static_cast<float>(i));
    }
    Tensor input({1, 2, 4, 4}, data);
    Tensor output = maxpool2d_forward(input, 2);

    EXPECT_EQ(output.shape(), (std::vector<int>{1, 2, 2, 2}));

    // Channel 0: same as basic test
    EXPECT_FLOAT_EQ(output.at(0, 0, 0, 0), 6.0f);
    EXPECT_FLOAT_EQ(output.at(0, 0, 1, 1), 16.0f);

    // Channel 1: max of each 2x2 window starting from 17
    EXPECT_FLOAT_EQ(output.at(0, 1, 0, 0), 22.0f);  // max(17,18,21,22)
    EXPECT_FLOAT_EQ(output.at(0, 1, 1, 1), 32.0f);  // max(27,28,31,32)
}

TEST(MaxPool2D, BatchSize2) {
    // Two images in a batch
    Tensor input({2, 1, 2, 2}, {1, 2, 3, 4, 10, 20, 30, 40});
    Tensor output = maxpool2d_forward(input, 2);

    EXPECT_EQ(output.shape(), (std::vector<int>{2, 1, 1, 1}));
    EXPECT_FLOAT_EQ(output.at(0, 0, 0, 0), 4.0f);
    EXPECT_FLOAT_EQ(output.at(1, 0, 0, 0), 40.0f);
}

// ============================================================================
// MaxPool2D — Stride Tests
// ============================================================================

TEST(MaxPool2D, DefaultStrideEqualsPoolSize) {
    // Default stride should equal pool_size (non-overlapping)
    Tensor input({1, 1, 4, 4},
                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    Tensor out_default = maxpool2d_forward(input, 2);        // stride=0 → 2
    Tensor out_explicit = maxpool2d_forward(input, 2, 2);    // stride=2

    EXPECT_TRUE(out_default == out_explicit);
}

TEST(MaxPool2D, OverlappingPoolStride1) {
    // Pool 2x2 with stride 1 → overlapping windows
    // Input: (1, 1, 3, 3)
    //   1 2 3
    //   4 5 6
    //   7 8 9
    // Output: (1, 1, 2, 2)
    // (0,0): max(1,2,4,5)=5, (0,1): max(2,3,5,6)=6
    // (1,0): max(4,5,7,8)=8, (1,1): max(5,6,8,9)=9
    Tensor input({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor output = maxpool2d_forward(input, 2, 1);

    EXPECT_EQ(output.shape(), (std::vector<int>{1, 1, 2, 2}));
    EXPECT_FLOAT_EQ(output.at(0, 0, 0, 0), 5.0f);
    EXPECT_FLOAT_EQ(output.at(0, 0, 0, 1), 6.0f);
    EXPECT_FLOAT_EQ(output.at(0, 0, 1, 0), 8.0f);
    EXPECT_FLOAT_EQ(output.at(0, 0, 1, 1), 9.0f);
}

// ============================================================================
// MaxPool2D — Output Dimension Tests
// ============================================================================

TEST(MaxPool2D, OutputDims28x28) {
    // Standard architecture: 28x28 with pool 2x2, stride 2 → 14x14
    Tensor input({1, 8, 28, 28});
    Tensor output = maxpool2d_forward(input, 2);
    EXPECT_EQ(output.shape(), (std::vector<int>{1, 8, 14, 14}));
}

TEST(MaxPool2D, OutputDims14x14) {
    // Second pooling: 14x14 → 7x7 (if used)
    Tensor input({1, 16, 14, 14});
    Tensor output = maxpool2d_forward(input, 2);
    EXPECT_EQ(output.shape(), (std::vector<int>{1, 16, 7, 7}));
}

TEST(MaxPool2D, OutputDimsOddInput) {
    // 5x5 input with pool 2x2, stride 2 → 2x2 (floor division)
    Tensor input({1, 1, 5, 5});
    Tensor output = maxpool2d_forward(input, 2);
    EXPECT_EQ(output.shape(), (std::vector<int>{1, 1, 2, 2}));
}

// ============================================================================
// MaxPool2D — Does Not Modify Input
// ============================================================================

TEST(MaxPool2D, DoesNotModifyInput) {
    Tensor input({1, 1, 4, 4},
                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    Tensor input_copy = input;
    maxpool2d_forward(input, 2);
    EXPECT_TRUE(input == input_copy);
}

// ============================================================================
// MaxPool2D — Validation Tests
// ============================================================================

TEST(MaxPool2DValidation, Input3DThrows) {
    Tensor input({1, 4, 4});
    EXPECT_THROW(maxpool2d_forward(input, 2), std::invalid_argument);
}

TEST(MaxPool2DValidation, ZeroPoolSizeThrows) {
    Tensor input({1, 1, 4, 4});
    EXPECT_THROW(maxpool2d_forward(input, 0), std::invalid_argument);
}

TEST(MaxPool2DValidation, InputTooSmallThrows) {
    Tensor input({1, 1, 1, 1});
    EXPECT_THROW(maxpool2d_forward(input, 2), std::invalid_argument);
}

// ============================================================================
// MaxPool2D — Large Tensor
// ============================================================================

TEST(MaxPool2D, LargeTensorOutputRange) {
    // After ReLU, all values >= 0. MaxPool of non-negative values stays >= 0.
    auto input = Tensor::rand({1, 8, 28, 28}, 42);
    Tensor output = maxpool2d_forward(input, 2);

    EXPECT_EQ(output.shape(), (std::vector<int>{1, 8, 14, 14}));

    // Each output should be >= every element in its pooling window
    for (int c = 0; c < 8; ++c) {
        for (int oh = 0; oh < 14; ++oh) {
            for (int ow = 0; ow < 14; ++ow) {
                float max_val = output.at(0, c, oh, ow);
                for (int ph = 0; ph < 2; ++ph) {
                    for (int pw = 0; pw < 2; ++pw) {
                        EXPECT_GE(max_val, input.at(0, c, oh * 2 + ph, ow * 2 + pw));
                    }
                }
            }
        }
    }
}
