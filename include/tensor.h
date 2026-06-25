#pragma once

// ============================================================================
// Tensor — Core data structure for the CNN accelerator
// ============================================================================
// Stores multi-dimensional floating-point data in contiguous NCHW layout.
// Serves as the primary data exchange format between all CPU layer
// implementations and as the host-side representation for GPU transfers.
// ============================================================================

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <numeric>
#include <string>
#include <vector>

namespace cnn {

class Tensor {
public:
    // ========================================================================
    // Construction
    // ========================================================================

    /// Default constructor — creates an empty tensor with no shape or data.
    Tensor() = default;

    /// Construct a tensor with the given shape, zero-initialized.
    /// Example: Tensor({1, 3, 28, 28}) creates a batch-1, 3-channel, 28x28 tensor.
    explicit Tensor(const std::vector<int>& shape);

    /// Construct a tensor with the given shape from initializer list.
    /// Example: Tensor({1, 1, 28, 28})
    explicit Tensor(std::initializer_list<int> shape);

    /// Construct a tensor with the given shape and pre-existing data.
    /// Data is copied into internal storage. Size of data must match
    /// the product of shape dimensions.
    Tensor(const std::vector<int>& shape, const std::vector<float>& data);

    // ========================================================================
    // Data access
    // ========================================================================

    /// Returns a pointer to the underlying contiguous data array.
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    /// Returns the total number of elements in the tensor.
    int num_elements() const { return static_cast<int>(data_.size()); }

    /// Returns the number of dimensions (rank) of the tensor.
    int ndim() const { return static_cast<int>(shape_.size()); }

    /// Returns the size of the given dimension.
    /// Negative indices count from the end: dim(-1) == last dimension.
    int dim(int d) const;

    // ========================================================================
    // Shape
    // ========================================================================

    /// Returns the shape vector (e.g., {1, 8, 28, 28}).
    const std::vector<int>& shape() const { return shape_; }

    /// Returns the stride vector for index computation.
    /// For a shape {N, C, H, W}, strides are {C*H*W, H*W, W, 1}.
    const std::vector<int>& strides() const { return strides_; }

    /// Returns true if the tensor has no elements.
    bool empty() const { return data_.empty(); }

    // ========================================================================
    // Copy and move (defaulted — std::vector handles everything)
    // ========================================================================

    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

private:
    // ========================================================================
    // Internal helpers
    // ========================================================================

    /// Compute strides from shape. Called during construction.
    void compute_strides();

    // ========================================================================
    // Data members
    // ========================================================================

    std::vector<float> data_;    ///< Contiguous NCHW storage
    std::vector<int> shape_;     ///< Dimension sizes, e.g., {N, C, H, W}
    std::vector<int> strides_;   ///< Strides for each dimension
};

}  // namespace cnn
