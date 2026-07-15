#include "layers/conv2d.h"

#include <cassert>
#include <stdexcept>
#include <string>

namespace cnn {

Tensor conv2d_forward(const Tensor& input, const Tensor& kernel, const Tensor& bias,
                      int stride, int padding) {
    // ========================================================================
    // Validate inputs
    // ========================================================================

    if (input.ndim() != 4) {
        throw std::invalid_argument(
            "conv2d_forward: input must be 4D (N, C_in, H, W), got " +
            std::to_string(input.ndim()) + "D");
    }
    if (kernel.ndim() != 4) {
        throw std::invalid_argument(
            "conv2d_forward: kernel must be 4D (C_out, C_in, kH, kW), got " +
            std::to_string(kernel.ndim()) + "D");
    }
    if (bias.ndim() != 1) {
        throw std::invalid_argument(
            "conv2d_forward: bias must be 1D (C_out), got " +
            std::to_string(bias.ndim()) + "D");
    }

    // Extract dimensions
    int batch_size = input.dim(0);
    int in_channels = input.dim(1);
    int in_height = input.dim(2);
    int in_width = input.dim(3);

    int out_channels = kernel.dim(0);
    int kern_in_channels = kernel.dim(1);
    int kern_height = kernel.dim(2);
    int kern_width = kernel.dim(3);

    // Validate channel compatibility
    if (in_channels != kern_in_channels) {
        throw std::invalid_argument(
            "conv2d_forward: input channels (" + std::to_string(in_channels) +
            ") must match kernel input channels (" + std::to_string(kern_in_channels) + ")");
    }
    if (bias.dim(0) != out_channels) {
        throw std::invalid_argument(
            "conv2d_forward: bias size (" + std::to_string(bias.dim(0)) +
            ") must match output channels (" + std::to_string(out_channels) + ")");
    }

    // ========================================================================
    // Compute output dimensions
    // ========================================================================

    int out_height = (in_height + 2 * padding - kern_height) / stride + 1;
    int out_width = (in_width + 2 * padding - kern_width) / stride + 1;

    if (out_height <= 0 || out_width <= 0) {
        throw std::invalid_argument(
            "conv2d_forward: invalid output dimensions (" +
            std::to_string(out_height) + " x " + std::to_string(out_width) +
            "). Check input size, kernel size, stride, and padding.");
    }

    // ========================================================================
    // Allocate output tensor
    // ========================================================================

    Tensor output({batch_size, out_channels, out_height, out_width});

    // ========================================================================
    // Naive convolution: 7 nested loops
    // ========================================================================
    //
    // For each output element (n, oc, oh, ow):
    //   output[n][oc][oh][ow] = bias[oc] +
    //       sum over ic, kh, kw of:
    //           input[n][ic][oh*stride + kh - padding][ow*stride + kw - padding]
    //           * kernel[oc][ic][kh][kw]
    //
    // Out-of-bounds input accesses (from padding) are treated as zero.

    for (int n = 0; n < batch_size; ++n) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    // Start with bias
                    float sum = bias.data()[oc];

                    // Accumulate convolution sum
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kern_height; ++kh) {
                            for (int kw = 0; kw < kern_width; ++kw) {
                                // Compute input position (with padding offset)
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;

                                // Zero-padding: skip if out of bounds
                                if (ih >= 0 && ih < in_height &&
                                    iw >= 0 && iw < in_width) {
                                    sum += input.at(n, ic, ih, iw) *
                                           kernel.at(oc, ic, kh, kw);
                                }
                            }
                        }
                    }

                    output.at(n, oc, oh, ow) = sum;
                }
            }
        }
    }

    return output;
}

}  // namespace cnn
