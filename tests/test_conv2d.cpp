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

// ============================================================================
// Conv2D — NumPy Reference Tests
// ============================================================================
// These values were generated by scripts/generate_conv2d_reference.py
// using a pure NumPy conv2d implementation with seeded random inputs.
// They serve as independent ground truth for validating our C++ code.

TEST(Conv2DReference, MultiChannelWithPadding) {
    // Input: (1, 2, 4, 4), Kernel: (2, 2, 3, 3), Bias: (2)
    // Stride: 1, Padding: 1
    // Generated with np.random.seed(42)

    // Shape: [1, 2, 4, 4], 32 elements
    std::vector<float> input_data = {
        -0.250920f, 0.901429f, 0.463988f, 0.197317f, -0.687963f, -0.688011f,
        -0.883833f, 0.732352f, 0.202230f, 0.416145f, -0.958831f, 0.939820f,
        0.664885f, -0.575322f, -0.636350f, -0.633191f, -0.391516f, 0.049513f,
        -0.136110f, -0.417542f, 0.223706f, -0.721012f, -0.415711f, -0.267276f,
        -0.087860f, 0.570352f, -0.600652f, 0.028469f, 0.184829f, -0.907099f,
        0.215090f, -0.658952f
    };

    // Shape: [2, 2, 3, 3], 36 elements
    std::vector<float> kernel_data = {
        -0.869897f, 0.897771f, 0.931264f, 0.616795f, -0.390772f, -0.804656f,
        0.368466f, -0.119695f, -0.755924f, -0.009646f, -0.931223f, 0.818641f,
        -0.482440f, 0.325045f, -0.376578f, 0.040136f, 0.093421f, -0.630291f,
        0.939169f, 0.550266f, 0.878998f, 0.789655f, 0.195800f, 0.843748f,
        -0.823015f, -0.608034f, -0.909545f, -0.349339f, -0.222645f, -0.457302f,
        0.657475f, -0.286493f, -0.438131f, 0.085392f, -0.718152f, 0.604394f
    };

    std::vector<float> bias_data = {0.100000f, -0.200000f};

    // Shape: [1, 2, 4, 4], 32 elements — NumPy reference output
    std::vector<float> expected = {
        0.404583f, 0.176473f, -0.195615f, -0.215951f, 1.579526f, 2.953706f,
        -1.147785f, -0.941223f, -1.669824f, 1.356251f, 0.898156f, 0.907997f,
        1.822616f, -0.935291f, 1.488546f, 1.294090f, 1.049574f, 2.031955f,
        1.536608f, 0.673718f, -0.038247f, -0.559212f, 0.826000f, -0.120469f,
        -1.299381f, -1.076335f, 2.052484f, 0.001775f, 0.025043f, -0.282882f,
        -1.071681f, -0.676135f
    };

    Tensor input({1, 2, 4, 4}, input_data);
    Tensor kernel({2, 2, 3, 3}, kernel_data);
    Tensor bias({2}, bias_data);

    Tensor output = conv2d_forward(input, kernel, bias, 1, 1);

    EXPECT_EQ(output.shape(), (std::vector<int>{1, 2, 4, 4}));

    // Verify all 32 output elements against NumPy reference
    Tensor expected_tensor({1, 2, 4, 4}, expected);
    EXPECT_TRUE(output.allclose(expected_tensor, 1e-4f))
        << "Max diff: " << output.max_diff(expected_tensor);
}

TEST(Conv2DReference, FourFiltersNoPadding) {
    // Input: (1, 1, 8, 8), Kernel: (4, 1, 3, 3), Bias: (4)
    // Stride: 1, Padding: 0
    // Generated with np.random.seed(123)

    // Shape: [1, 1, 8, 8], 64 elements
    std::vector<float> input_data = {
        0.696469f, 0.286139f, 0.226851f, 0.551315f, 0.719469f, 0.423106f,
        0.980764f, 0.684830f, 0.480932f, 0.392118f, 0.343178f, 0.729050f,
        0.438572f, 0.059678f, 0.398044f, 0.737995f, 0.182492f, 0.175452f,
        0.531551f, 0.531828f, 0.634401f, 0.849432f, 0.724455f, 0.611023f,
        0.722443f, 0.322959f, 0.361789f, 0.228263f, 0.293714f, 0.630976f,
        0.092105f, 0.433701f, 0.430863f, 0.493685f, 0.425830f, 0.312261f,
        0.426351f, 0.893389f, 0.944160f, 0.501837f, 0.623953f, 0.115618f,
        0.317285f, 0.414826f, 0.866309f, 0.250455f, 0.483034f, 0.985560f,
        0.519485f, 0.612895f, 0.120629f, 0.826341f, 0.603060f, 0.545068f,
        0.342764f, 0.304121f, 0.417022f, 0.681301f, 0.875457f, 0.510422f,
        0.669314f, 0.585937f, 0.624904f, 0.674689f
    };

    // Shape: [4, 1, 3, 3], 36 elements
    std::vector<float> kernel_data = {
        0.342342f, -0.416805f, 0.263683f, -0.256334f, -0.305777f, 0.072457f,
        -0.404287f, 0.385327f, 0.127249f, 0.223416f, -0.483871f, 0.094432f,
        0.056785f, -0.341040f, -0.346929f, 0.195530f, -0.181234f, 0.191970f,
        0.054383f, -0.111049f, 0.425132f, 0.341670f, -0.142602f, -0.456409f,
        -0.195232f, -0.101814f, 0.204959f, 0.495358f, -0.144085f, 0.262548f,
        0.093177f, 0.191702f, -0.348873f, -0.101124f, -0.259144f, -0.156544f
    };

    std::vector<float> bias_data = {0.000000f, 0.100000f, -0.100000f, 0.050000f};

    // Shape: [1, 4, 6, 6], 144 elements — NumPy reference output
    std::vector<float> expected = {
        0.022136f, 0.197715f, -0.170781f, -0.178705f, 0.389749f, -0.153410f,
        -0.091809f, 0.052347f, -0.344587f, -0.085059f, -0.003531f, -0.510284f,
        -0.057995f, -0.193899f, -0.011252f, 0.190746f, 0.084630f, 0.057959f,
        -0.189769f, -0.085874f, 0.074223f, 0.174394f, -0.644477f, 0.020915f,
        -0.076794f, -0.118892f, 0.329026f, -0.220662f, -0.231265f, -0.145439f,
        0.142575f, 0.009447f, -0.147839f, -0.383648f, 0.057833f, 0.011551f,
        0.019020f, -0.201411f, -0.300108f, -0.061799f, 0.124259f, -0.451666f,
        -0.031583f, -0.224027f, -0.420206f, -0.312014f, -0.337548f, -0.230454f,
        -0.012041f, -0.172718f, -0.029355f, -0.158755f, -0.228629f, -0.049203f,
        0.009623f, -0.162135f, 0.021053f, -0.396965f, -0.522295f, -0.057381f,
        -0.102826f, 0.044791f, -0.350104f, -0.154035f, -0.266741f, -0.499457f,
        0.118174f, -0.209001f, -0.225053f, -0.459713f, 0.092055f, -0.045320f,
        0.009771f, -0.102327f, -0.057793f, 0.194955f, 0.206906f, -0.382321f,
        -0.276540f, -0.118530f, -0.193291f, -0.325274f, -0.251285f, 0.037155f,
        0.104975f, -0.044556f, 0.069017f, 0.046453f, 0.135815f, -0.037675f,
        -0.128865f, -0.029991f, -0.001054f, -0.331670f, -0.623219f, 0.153952f,
        -0.037568f, -0.145333f, -0.260483f, 0.042319f, 0.162205f, -0.455023f,
        0.119668f, -0.255064f, -0.198083f, -0.167306f, 0.034284f, 0.237790f,
        0.266456f, -0.086964f, -0.000322f, 0.110550f, 0.118680f, -0.246812f,
        -0.026354f, 0.156762f, 0.018535f, 0.040471f, 0.124730f, 0.065258f,
        0.019518f, 0.010609f, 0.187516f, 0.020580f, 0.125830f, 0.038380f,
        0.299821f, 0.077622f, -0.050964f, -0.220092f, -0.217764f, 0.247533f,
        0.043466f, 0.019457f, -0.186006f, 0.169987f, 0.084683f, 0.068812f,
        0.193764f, -0.413256f, 0.008528f, -0.117923f, 0.293617f, 0.046810f
    };

    Tensor input({1, 1, 8, 8}, input_data);
    Tensor kernel({4, 1, 3, 3}, kernel_data);
    Tensor bias({4}, bias_data);

    Tensor output = conv2d_forward(input, kernel, bias, 1, 0);

    EXPECT_EQ(output.shape(), (std::vector<int>{1, 4, 6, 6}));

    // Verify all 144 output elements against NumPy reference
    Tensor expected_tensor({1, 4, 6, 6}, expected);
    EXPECT_TRUE(output.allclose(expected_tensor, 1e-4f))
        << "Max diff: " << output.max_diff(expected_tensor);
}
