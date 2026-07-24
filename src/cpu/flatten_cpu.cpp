#include "layers/flatten.h"

namespace cnn {

Tensor flatten(const Tensor& input) {
    // Delegates to Tensor::flatten() which reshapes (N, C, H, W) → (N, C*H*W)
    // For 1D input, returns (1, num_elements)
    return input.flatten();
}

}  // namespace cnn
