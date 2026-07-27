#include "layers/softmax.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace cnn {

Tensor softmax_forward(const Tensor& input) {
    // ========================================================================
    // Validate inputs
    // ========================================================================

    if (input.ndim() != 2) {
        throw std::invalid_argument(
            "softmax_forward: input must be 2D (N, num_classes), got " +
            std::to_string(input.ndim()) + "D");
    }

    int batch_size = input.dim(0);
    int num_classes = input.dim(1);

    Tensor output({batch_size, num_classes});

    // ========================================================================
    // Numerically stable softmax per sample
    // ========================================================================
    //
    // For each sample n:
    //   1. Find max_val = max over j of input[n][j]
    //   2. Compute exp_vals[j] = exp(input[n][j] - max_val)
    //   3. Compute sum_exp = sum of exp_vals
    //   4. output[n][j] = exp_vals[j] / sum_exp

    for (int n = 0; n < batch_size; ++n) {
        // Step 1: Find max for numerical stability
        float max_val = input.at(n, 0);
        for (int j = 1; j < num_classes; ++j) {
            max_val = std::max(max_val, input.at(n, j));
        }

        // Step 2 & 3: Compute exp(x - max) and accumulate sum
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            float exp_val = std::exp(input.at(n, j) - max_val);
            output.at(n, j) = exp_val;  // Store temporarily
            sum_exp += exp_val;
        }

        // Step 4: Normalize
        for (int j = 0; j < num_classes; ++j) {
            output.at(n, j) /= sum_exp;
        }
    }

    return output;
}

}  // namespace cnn
