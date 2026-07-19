#include "layers/relu.h"

#include <algorithm>

namespace cnn {

Tensor relu_forward(const Tensor& input) {
    Tensor output(input.shape());

    for (int i = 0; i < input.num_elements(); ++i) {
        output[i] = std::max(0.0f, input[i]);
    }

    return output;
}

void relu_forward_inplace(Tensor& input) {
    for (int i = 0; i < input.num_elements(); ++i) {
        input[i] = std::max(0.0f, input[i]);
    }
}

}  // namespace cnn
