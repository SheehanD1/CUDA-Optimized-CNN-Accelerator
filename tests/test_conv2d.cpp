#include "layers/conv2d.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

using cnn::Tensor;
using cnn::compute_same_padding;
using cnn::conv2d_forward;
using cnn::pad_tensor;

// ============================================================================
// Conv2D — Basic Correctness Tests
// ============================================================================

TEST(Conv2DForward, SingleElementNoPadding) {
    // 1x1 spatial input, 1x1 kernel → output should be input * kernel + bias
    // Input: (1, 1, 1, 1) = [3.0]
    // Kernel: (1, 1, 1, 1) = [2.0]
    // Bias: (1) = [0.5]
    // Expected output: 3.0 * 2.0 + 0.5 = 6.5
    Tensor input({1, 1, 1, 1}, {3.0f});
    Tensor kernel({1, 1, 1, 1}, {2.0f});
    Tensor bias({1}, {0.5f});

    Tensor output = conv2d_forward(input, kernel, bias, 1, 0);
    EXPECT_EQ(output.shape(), (std::vector<int>{1, 1, 1, 1}));
    EXPECT_FLOAT_EQ(output.at(0, 0, 0, 0), 6.5f);
}

TEST(Conv2DForward, Simple3x3NoPadding) {
    // Input: (1, 1, 3, 3) — values 1..9
    // Kernel: (1, 1, 3, 3) — all ones
    // Bias: (1) = [0.0]
    // With no padding and 3x3 kernel on 3x3 input → 1x1 output
    // Output = sum(1..9) + 0 = 45
    Tensor input({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor kernel({1, 1, 3, 3}, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    Tensor bias({1}, {0.0f});

    Tensor output = conv2d_forward(input, kernel, bias, 1, 0);
    EXPECT_EQ(output.shape(), (std::vector<int>{1, 1, 1, 1}));
    EXPECT_FLOAT_EQ(output.at(0, 0, 0, 0), 45.0f);
}

TEST(Conv2DForward, Simple3x3WithBias) {
    // Same as above but with bias = 10.0
    // Output = 45 + 10 = 55
    Tensor input({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor kernel({1, 1, 3, 3}, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    Tensor bias({1}, {10.0f});

    Tensor output = conv2d_forward(input, kernel, bias, 1, 0);
    EXPECT_FLOAT_EQ(output.at(0, 0, 0, 0), 55.0f);
}

TEST(Conv2DForward, Padding1On3x3) {
    // Input: (1, 1, 3, 3) — values 1..9
    // Kernel: (1, 1, 3, 3) — identity kernel (1 at center, 0 elsewhere)
    // Bias: (1) = [0.0]
    // Padding: 1
    // With padding=1 and stride=1, output is same size as input: 3x3
    // Identity kernel: output[h][w] = input[h][w] (center of kernel picks original)
    Tensor input({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor kernel({1, 1, 3, 3}, {0, 0, 0, 0, 1, 0, 0, 0, 0});
    Tensor bias({1}, {0.0f});

    Tensor output = conv2d_forward(input, kernel, bias, 1, 1);
    EXPECT_EQ(output.shape(), (std::vector<int>{1, 1, 3, 3}));

    // Output should equal input (identity convolution)
    for (int h = 0; h < 3; ++h) {
        for (int w = 0; w < 3; ++w) {
            EXPECT_FLOAT_EQ(output.at(0, 0, h, w), input.at(0, 0, h, w))
                << "Mismatch at (" << h << ", " << w << ")";
        }
    }
}

TEST(Conv2DForward, Stride2) {
    // Input: (1, 1, 4, 4) — values 1..16
    // Kernel: (1, 1, 1, 1) — [1.0] (identity scalar)
    // Bias: (1) = [0.0]
    // Stride: 2, Padding: 0
    // Output: (1, 1, 2, 2) — picks every other element
    Tensor input({1, 1, 4, 4},
                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    Tensor kernel({1, 1, 1, 1}, {1.0f});
    Tensor bias({1}, {0.0f});

    Tensor output = conv2d_forward(input, kernel, bias, 2, 0);
    EXPECT_EQ(output.shape(), (std::vector<int>{1, 1, 2, 2}));
    EXPECT_FLOAT_EQ(output.at(0, 0, 0, 0), 1.0f);   // (0,0)
    EXPECT_FLOAT_EQ(output.at(0, 0, 0, 1), 3.0f);   // (0,2)
    EXPECT_FLOAT_EQ(output.at(0, 0, 1, 0), 9.0f);   // (2,0)
    EXPECT_FLOAT_EQ(output.at(0, 0, 1, 1), 11.0f);  // (2,2)
}

// ============================================================================
// Conv2D — Multi-channel Tests
// ============================================================================

TEST(Conv2DForward, MultiInputChannel) {
    // Input: (1, 2, 2, 2) — two channels
    // Channel 0: [1, 2, 3, 4], Channel 1: [5, 6, 7, 8]
    // Kernel: (1, 2, 1, 1) — [1, 1] — sum across channels
    // Bias: (1) = [0.0]
    // Output: elementwise sum of channels
    Tensor input({1, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    Tensor kernel({1, 2, 1, 1}, {1.0f, 1.0f});
    Tensor bias({1}, {0.0f});

    Tensor output = conv2d_forward(input, kernel, bias, 1, 0);
    EXPECT_EQ(output.shape(), (std::vector<int>{1, 1, 2, 2}));
    EXPECT_FLOAT_EQ(output.at(0, 0, 0, 0), 6.0f);   // 1 + 5
    EXPECT_FLOAT_EQ(output.at(0, 0, 0, 1), 8.0f);   // 2 + 6
    EXPECT_FLOAT_EQ(output.at(0, 0, 1, 0), 10.0f);  // 3 + 7
    EXPECT_FLOAT_EQ(output.at(0, 0, 1, 1), 12.0f);  // 4 + 8
}

TEST(Conv2DForward, MultiOutputChannel) {
    // Input: (1, 1, 2, 2) = [1, 2, 3, 4]
    // Kernel: (2, 1, 1, 1) — two output filters: [1.0] and [2.0]
    // Bias: (2) = [0, 0]
    // Output channel 0 = input * 1, channel 1 = input * 2
    Tensor input({1, 1, 2, 2}, {1, 2, 3, 4});
    Tensor kernel({2, 1, 1, 1}, {1.0f, 2.0f});
    Tensor bias({2}, {0.0f, 0.0f});

    Tensor output = conv2d_forward(input, kernel, bias, 1, 0);
    EXPECT_EQ(output.shape(), (std::vector<int>{1, 2, 2, 2}));

    // Channel 0: input * 1
    EXPECT_FLOAT_EQ(output.at(0, 0, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(output.at(0, 0, 1, 1), 4.0f);

    // Channel 1: input * 2
    EXPECT_FLOAT_EQ(output.at(0, 1, 0, 0), 2.0f);
    EXPECT_FLOAT_EQ(output.at(0, 1, 1, 1), 8.0f);
}

TEST(Conv2DForward, BatchSize2) {
    // Two identical images in a batch, same kernel
    Tensor input({2, 1, 1, 1}, {3.0f, 7.0f});
    Tensor kernel({1, 1, 1, 1}, {2.0f});
    Tensor bias({1}, {1.0f});

    Tensor output = conv2d_forward(input, kernel, bias, 1, 0);
    EXPECT_EQ(output.shape(), (std::vector<int>{2, 1, 1, 1}));
    EXPECT_FLOAT_EQ(output.at(0, 0, 0, 0), 7.0f);   // 3*2 + 1
    EXPECT_FLOAT_EQ(output.at(1, 0, 0, 0), 15.0f);  // 7*2 + 1
}

// ============================================================================
// Conv2D — Output Dimension Tests
// ============================================================================

TEST(Conv2DForward, OutputDimensions) {
    // Input: (1, 1, 28, 28), Kernel: 3x3, stride=1, padding=1
    // Expected output: (1, 8, 28, 28) — same spatial dims with "same" padding
    Tensor input({1, 1, 28, 28});
    Tensor kernel({8, 1, 3, 3});
    Tensor bias({8});

    Tensor output = conv2d_forward(input, kernel, bias, 1, 1);
    EXPECT_EQ(output.shape(), (std::vector<int>{1, 8, 28, 28}));
}

TEST(Conv2DForward, OutputDimsNoPadding) {
    // Input: (1, 1, 28, 28), Kernel: 3x3, stride=1, padding=0
    // Expected output: (1, 8, 26, 26)
    Tensor input({1, 1, 28, 28});
    Tensor kernel({8, 1, 3, 3});
    Tensor bias({8});

    Tensor output = conv2d_forward(input, kernel, bias, 1, 0);
    EXPECT_EQ(output.shape(), (std::vector<int>{1, 8, 26, 26}));
}

TEST(Conv2DForward, OutputDimsStride2) {
    // Input: (1, 8, 14, 14), Kernel: 3x3, stride=2, padding=1
    // H_out = (14 + 2 - 3) / 2 + 1 = 7
    Tensor input({1, 8, 14, 14});
    Tensor kernel({16, 8, 3, 3});
    Tensor bias({16});

    Tensor output = conv2d_forward(input, kernel, bias, 2, 1);
    EXPECT_EQ(output.shape(), (std::vector<int>{1, 16, 7, 7}));
}

// ============================================================================
// Conv2D — Hand-Computed Reference Test
// ============================================================================

TEST(Conv2DForward, HandComputedConvolution) {
    // Input: (1, 1, 4, 4)
    //   1  2  3  4
    //   5  6  7  8
    //   9 10 11 12
    //  13 14 15 16
    //
    // Kernel: (1, 1, 2, 2)
    //   1  0
    //   0  1
    //
    // Bias: [0]
    // Stride: 1, Padding: 0
    // Output: (1, 1, 3, 3)
    //
    // out[0,0] = 1*1 + 2*0 + 5*0 + 6*1 = 7
    // out[0,1] = 2*1 + 3*0 + 6*0 + 7*1 = 9
    // out[0,2] = 3*1 + 4*0 + 7*0 + 8*1 = 11
    // out[1,0] = 5*1 + 6*0 + 9*0 + 10*1 = 15
    // out[1,1] = 6*1 + 7*0 + 10*0 + 11*1 = 17
    // out[1,2] = 7*1 + 8*0 + 11*0 + 12*1 = 19
    // out[2,0] = 9*1 + 10*0 + 13*0 + 14*1 = 23
    // out[2,1] = 10*1 + 11*0 + 14*0 + 15*1 = 25
    // out[2,2] = 11*1 + 12*0 + 15*0 + 16*1 = 27

    Tensor input({1, 1, 4, 4},
                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    Tensor kernel({1, 1, 2, 2}, {1, 0, 0, 1});
    Tensor bias({1}, {0.0f});

    Tensor output = conv2d_forward(input, kernel, bias, 1, 0);
    EXPECT_EQ(output.shape(), (std::vector<int>{1, 1, 3, 3}));

    std::vector<float> expected = {7, 9, 11, 15, 17, 19, 23, 25, 27};
    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(output[i], expected[static_cast<size_t>(i)])
            << "Mismatch at flat index " << i;
    }
}

// ============================================================================
// Conv2D — String Padding Mode Tests
// ============================================================================

TEST(Conv2DForward, SamePaddingMode) {
    // "same" with 3x3 kernel → padding=1 → output same as input spatial dims
    Tensor input({1, 1, 5, 5});
    Tensor kernel({1, 1, 3, 3});
    Tensor bias({1});

    Tensor output = conv2d_forward(input, kernel, bias, 1, "same");
    EXPECT_EQ(output.shape(), (std::vector<int>{1, 1, 5, 5}));
}

TEST(Conv2DForward, ValidPaddingMode) {
    Tensor input({1, 1, 5, 5});
    Tensor kernel({1, 1, 3, 3});
    Tensor bias({1});

    Tensor output = conv2d_forward(input, kernel, bias, 1, "valid");
    EXPECT_EQ(output.shape(), (std::vector<int>{1, 1, 3, 3}));
}

TEST(Conv2DForward, InvalidPaddingModeThrows) {
    Tensor input({1, 1, 5, 5});
    Tensor kernel({1, 1, 3, 3});
    Tensor bias({1});

    EXPECT_THROW(conv2d_forward(input, kernel, bias, 1, "invalid"), std::invalid_argument);
}

// ============================================================================
// Conv2D — Input Validation Tests
// ============================================================================

TEST(Conv2DValidation, Input3DThrows) {
    Tensor input({1, 3, 3});
    Tensor kernel({1, 1, 3, 3});
    Tensor bias({1});
    EXPECT_THROW(conv2d_forward(input, kernel, bias), std::invalid_argument);
}

TEST(Conv2DValidation, ChannelMismatchThrows) {
    Tensor input({1, 3, 5, 5});   // 3 input channels
    Tensor kernel({1, 2, 3, 3});  // kernel expects 2 input channels
    Tensor bias({1});
    EXPECT_THROW(conv2d_forward(input, kernel, bias), std::invalid_argument);
}

TEST(Conv2DValidation, BiasSizeMismatchThrows) {
    Tensor input({1, 1, 5, 5});
    Tensor kernel({4, 1, 3, 3});  // 4 output channels
    Tensor bias({2});             // bias has only 2
    EXPECT_THROW(conv2d_forward(input, kernel, bias), std::invalid_argument);
}

// ============================================================================
// Padding Utility Tests
// ============================================================================

TEST(PadTensor, NoPadding) {
    Tensor input({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor padded = pad_tensor(input, 0);
    EXPECT_EQ(padded.shape(), input.shape());
    EXPECT_TRUE(padded == input);
}

TEST(PadTensor, Padding1) {
    Tensor input({1, 1, 2, 2}, {1, 2, 3, 4});
    Tensor padded = pad_tensor(input, 1);
    EXPECT_EQ(padded.shape(), (std::vector<int>{1, 1, 4, 4}));

    // Corners should be zero (padding)
    EXPECT_FLOAT_EQ(padded.at(0, 0, 0, 0), 0.0f);
    EXPECT_FLOAT_EQ(padded.at(0, 0, 3, 3), 0.0f);

    // Center should have original data
    EXPECT_FLOAT_EQ(padded.at(0, 0, 1, 1), 1.0f);
    EXPECT_FLOAT_EQ(padded.at(0, 0, 1, 2), 2.0f);
    EXPECT_FLOAT_EQ(padded.at(0, 0, 2, 1), 3.0f);
    EXPECT_FLOAT_EQ(padded.at(0, 0, 2, 2), 4.0f);
}

TEST(PadTensor, NegativePaddingThrows) {
    Tensor input({1, 1, 3, 3});
    EXPECT_THROW(pad_tensor(input, -1), std::invalid_argument);
}

TEST(PadTensor, Non4DThrows) {
    Tensor input({3, 3});
    EXPECT_THROW(pad_tensor(input, 1), std::invalid_argument);
}

// ============================================================================
// compute_same_padding Tests
// ============================================================================

TEST(ComputeSamePadding, Kernel3) {
    EXPECT_EQ(compute_same_padding(3), 1);
}

TEST(ComputeSamePadding, Kernel5) {
    EXPECT_EQ(compute_same_padding(5), 2);
}

TEST(ComputeSamePadding, Kernel1) {
    EXPECT_EQ(compute_same_padding(1), 0);
}

TEST(ComputeSamePadding, EvenKernelThrows) {
    EXPECT_THROW(compute_same_padding(4), std::invalid_argument);
}

TEST(ComputeSamePadding, ZeroKernelThrows) {
    EXPECT_THROW(compute_same_padding(0), std::invalid_argument);
}
