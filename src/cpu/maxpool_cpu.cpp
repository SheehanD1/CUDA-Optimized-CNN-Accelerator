#include "layers/maxpool2d.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace cnn {

Tensor maxpool2d_forward(const Tensor& input, int pool_size, int stride) {
    // ========================================================================
    // Validate inputs
    // ========================================================================

    if (input.ndim() != 4) {
        throw std::invalid_argument(
            "maxpool2d_forward: input must be 4D (N, C, H, W), got " +
            std::to_string(input.ndim()) + "D");
    }

    if (pool_size <= 0) {
        throw std::invalid_argument(
            "maxpool2d_forward: pool_size must be positive, got " +
            std::to_string(pool_size));
    }

    // Default stride = pool_size (non-overlapping pooling)
    if (stride == 0) {
        stride = pool_size;
    }

    if (stride <= 0) {
        throw std::invalid_argument(
            "maxpool2d_forward: stride must be positive, got " +
            std::to_string(stride));
    }

    // Extract dimensions
    int batch_size = input.dim(0);
    int channels = input.dim(1);
    int in_height = input.dim(2);
    int in_width = input.dim(3);

    // ========================================================================
    // Compute output dimensions
    // ========================================================================

    int out_height = (in_height - pool_size) / stride + 1;
    int out_width = (in_width - pool_size) / stride + 1;

    if (out_height <= 0 || out_width <= 0) {
        throw std::invalid_argument(
            "maxpool2d_forward: invalid output dimensions (" +
            std::to_string(out_height) + " x " + std::to_string(out_width) +
            "). Input spatial dims (" + std::to_string(in_height) + " x " +
            std::to_string(in_width) + ") too small for pool_size=" +
            std::to_string(pool_size));
    }

    // ========================================================================
    // Allocate output tensor
    // ========================================================================

    Tensor output({batch_size, channels, out_height, out_width});

    // ========================================================================
    // Max pooling: for each output element, find max in the pooling window
    // ========================================================================
    //
    // For each (n, c, oh, ow):
    //   output[n][c][oh][ow] = max over (ph, pw) in [0, pool_size) of:
    //       input[n][c][oh*stride + ph][ow*stride + pw]

    for (int n = 0; n < batch_size; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();

                    // Scan the pooling window
                    for (int ph = 0; ph < pool_size; ++ph) {
                        for (int pw = 0; pw < pool_size; ++pw) {
                            int ih = oh * stride + ph;
                            int iw = ow * stride + pw;
                            float val = input.at(n, c, ih, iw);
                            max_val = std::max(max_val, val);
                        }
                    }

                    output.at(n, c, oh, ow) = max_val;
                }
            }
        }
    }

    return output;
}

}  // namespace cnn
