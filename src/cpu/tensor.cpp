#include "tensor.h"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <stdexcept>

namespace cnn {

// ============================================================================
// Construction
// ============================================================================

Tensor::Tensor(const std::vector<int>& shape) : shape_(shape) {
    // Validate: all dimensions must be positive
    for (int d : shape_) {
        if (d <= 0) {
            throw std::invalid_argument(
                "Tensor: all shape dimensions must be positive, got " + std::to_string(d));
        }
    }

    // Compute total number of elements
    int total = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());

    // Allocate zero-initialized storage
    data_.resize(static_cast<size_t>(total), 0.0f);

    // Precompute strides for index calculations
    compute_strides();
}

Tensor::Tensor(std::initializer_list<int> shape) : Tensor(std::vector<int>(shape)) {}

Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data) : shape_(shape) {
    // Validate dimensions
    for (int d : shape_) {
        if (d <= 0) {
            throw std::invalid_argument(
                "Tensor: all shape dimensions must be positive, got " + std::to_string(d));
        }
    }

    // Validate data size matches shape
    int total = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
    if (static_cast<int>(data.size()) != total) {
        throw std::invalid_argument(
            "Tensor: data size (" + std::to_string(data.size()) +
            ") does not match shape product (" + std::to_string(total) + ")");
    }

    data_ = data;
    compute_strides();
}

// ============================================================================
// Data access
// ============================================================================

int Tensor::dim(int d) const {
    // Support negative indexing: dim(-1) == last dimension
    if (d < 0) {
        d += ndim();
    }

    if (d < 0 || d >= ndim()) {
        throw std::out_of_range(
            "Tensor::dim: index " + std::to_string(d) + " out of range for tensor with " +
            std::to_string(ndim()) + " dimensions");
    }

    return shape_[static_cast<size_t>(d)];
}

// ============================================================================
// Internal helpers
// ============================================================================

void Tensor::compute_strides() {
    strides_.resize(shape_.size());

    if (shape_.empty()) {
        return;
    }

    // Strides are computed right-to-left:
    //   For shape {N, C, H, W}:
    //     stride[3] = 1
    //     stride[2] = W
    //     stride[1] = H * W
    //     stride[0] = C * H * W
    strides_.back() = 1;
    for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
        strides_[static_cast<size_t>(i)] =
            strides_[static_cast<size_t>(i + 1)] * shape_[static_cast<size_t>(i + 1)];
    }
}

}  // namespace cnn
