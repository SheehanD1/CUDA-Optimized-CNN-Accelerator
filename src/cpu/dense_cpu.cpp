#include "layers/dense.h"

#include <stdexcept>
#include <string>

namespace cnn {

Tensor dense_forward(const Tensor& input, const Tensor& weights, const Tensor& bias) {
    // ========================================================================
    // Validate inputs
    // ========================================================================

    if (input.ndim() != 2) {
        throw std::invalid_argument(
            "dense_forward: input must be 2D (N, in_features), got " +
            std::to_string(input.ndim()) + "D. "
            "Did you forget to flatten the tensor?");
    }
    if (weights.ndim() != 2) {
        throw std::invalid_argument(
            "dense_forward: weights must be 2D (out_features, in_features), got " +
            std::to_string(weights.ndim()) + "D");
    }
    if (bias.ndim() != 1) {
        throw std::invalid_argument(
            "dense_forward: bias must be 1D (out_features), got " +
            std::to_string(bias.ndim()) + "D");
    }

    int batch_size = input.dim(0);
    int in_features = input.dim(1);
    int out_features = weights.dim(0);
    int weights_in = weights.dim(1);

    if (in_features != weights_in) {
        throw std::invalid_argument(
            "dense_forward: input features (" + std::to_string(in_features) +
            ") must match weight columns (" + std::to_string(weights_in) + ")");
    }
    if (bias.dim(0) != out_features) {
        throw std::invalid_argument(
            "dense_forward: bias size (" + std::to_string(bias.dim(0)) +
            ") must match out_features (" + std::to_string(out_features) + ")");
    }

    // ========================================================================
    // Compute output = input * weights^T + bias
    // ========================================================================
    //
    // For each sample n and output feature j:
    //   output[n][j] = bias[j] + sum over k of: input[n][k] * weights[j][k]

    Tensor output({batch_size, out_features});

    for (int n = 0; n < batch_size; ++n) {
        for (int j = 0; j < out_features; ++j) {
            float sum = bias.data()[j];

            for (int k = 0; k < in_features; ++k) {
                sum += input.at(n, k) * weights.at(j, k);
            }

            output.at(n, j) = sum;
        }
    }

    return output;
}

}  // namespace cnn
