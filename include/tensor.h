#pragma once

// ============================================================================
// Tensor — Core data structure for the CNN accelerator
// ============================================================================
// Stores multi-dimensional floating-point data in contiguous NCHW layout.
// Serves as the primary data exchange format between all CPU layer
// implementations and as the host-side representation for GPU transfers.
// ============================================================================

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <numeric>
#include <random>
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
    // Static factory methods
    // ========================================================================

    /// Create a tensor filled with zeros.
    /// Example: auto t = Tensor::zeros({1, 8, 28, 28});
    static Tensor zeros(const std::vector<int>& shape);

    /// Create a tensor filled with ones.
    /// Example: auto t = Tensor::ones({8});
    static Tensor ones(const std::vector<int>& shape);

    /// Create a tensor filled with a constant value.
    /// Example: auto t = Tensor::full({3, 3}, 0.5f);
    static Tensor full(const std::vector<int>& shape, float value);

    /// Create a tensor with uniform random values in [0, 1).
    /// Uses a deterministic seed if provided (default: random_device).
    /// Example: auto t = Tensor::rand({1, 1, 28, 28});
    /// Example: auto t = Tensor::rand({1, 1, 28, 28}, 42);  // reproducible
    static Tensor rand(const std::vector<int>& shape, unsigned int seed = 0);

    /// Create a tensor with values from a normal distribution N(mean, stddev).
    /// Useful for weight initialization.
    /// Example: auto t = Tensor::randn({8, 1, 3, 3}, 0.0f, 0.1f);
    static Tensor randn(const std::vector<int>& shape, float mean = 0.0f,
                        float stddev = 1.0f, unsigned int seed = 0);

    // ========================================================================
    // In-place operations
    // ========================================================================

    /// Fill all elements with the given value.
    void fill(float value) { std::fill(data_.begin(), data_.end(), value); }

    /// Set all elements to zero.
    void zero() { fill(0.0f); }

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
    // Element access (indexing)
    // ========================================================================

    /// Access element by 4D NCHW indices (batch, channel, height, width).
    /// This is the primary accessor for CNN tensor data.
    /// Bounds-checked in debug builds via assert.
    float& at(int n, int c, int h, int w) {
        assert(ndim() == 4 && "at(n,c,h,w) requires a 4D tensor");
        int idx = flat_index_4d(n, c, h, w);
        return data_[static_cast<size_t>(idx)];
    }

    const float& at(int n, int c, int h, int w) const {
        assert(ndim() == 4 && "at(n,c,h,w) requires a 4D tensor");
        int idx = flat_index_4d(n, c, h, w);
        return data_[static_cast<size_t>(idx)];
    }

    /// Access element by 2D indices (row, col) — for fully connected layers.
    float& at(int r, int c) {
        assert(ndim() == 2 && "at(r,c) requires a 2D tensor");
        int idx = r * strides_[0] + c * strides_[1];
        assert(idx >= 0 && idx < num_elements() && "Index out of bounds");
        return data_[static_cast<size_t>(idx)];
    }

    const float& at(int r, int c) const {
        assert(ndim() == 2 && "at(r,c) requires a 2D tensor");
        int idx = r * strides_[0] + c * strides_[1];
        assert(idx >= 0 && idx < num_elements() && "Index out of bounds");
        return data_[static_cast<size_t>(idx)];
    }

    /// Access element by generic index vector.
    /// Number of indices must match the tensor's number of dimensions.
    float& at(const std::vector<int>& indices) {
        int idx = flat_index(indices);
        return data_[static_cast<size_t>(idx)];
    }

    const float& at(const std::vector<int>& indices) const {
        int idx = flat_index(indices);
        return data_[static_cast<size_t>(idx)];
    }

    /// Flat (linear) element access — no dimension checking.
    /// Index must be in [0, num_elements()).
    float& operator[](int i) {
        assert(i >= 0 && i < num_elements() && "Flat index out of bounds");
        return data_[static_cast<size_t>(i)];
    }

    const float& operator[](int i) const {
        assert(i >= 0 && i < num_elements() && "Flat index out of bounds");
        return data_[static_cast<size_t>(i)];
    }

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
    // Shape manipulation
    // ========================================================================

    /// Returns a new tensor with the same data but a different shape.
    /// The total number of elements must remain unchanged.
    /// Supports -1 for one dimension to auto-infer its size.
    /// Example: Tensor({2, 3, 4}).reshape({6, 4})
    /// Example: Tensor({2, 3, 4}).reshape({-1, 4})  → {6, 4}
    Tensor reshape(const std::vector<int>& new_shape) const;

    /// Returns a new tensor with the same data but shape {new_shape} via initializer list.
    Tensor reshape(std::initializer_list<int> new_shape) const {
        return reshape(std::vector<int>(new_shape));
    }

    /// Flatten to 2D: {batch_size, all_other_dims}.
    /// For a tensor of shape {N, C, H, W}, returns shape {N, C*H*W}.
    /// For a 1D tensor, returns shape {1, num_elements}.
    Tensor flatten() const;

    /// Flatten completely to 1D: {num_elements}.
    Tensor flatten_all() const { return reshape({num_elements()}); }

    // ========================================================================
    // NCHW convenience accessors (for 4D tensors)
    // ========================================================================

    /// Returns batch size (N) — first dimension of a 4D NCHW tensor.
    int batch_size() const {
        assert(ndim() == 4 && "batch_size() requires a 4D tensor");
        return shape_[0];
    }

    /// Returns number of channels (C) — second dimension of a 4D NCHW tensor.
    int channels() const {
        assert(ndim() == 4 && "channels() requires a 4D tensor");
        return shape_[1];
    }

    /// Returns height (H) — third dimension of a 4D NCHW tensor.
    int height() const {
        assert(ndim() == 4 && "height() requires a 4D tensor");
        return shape_[2];
    }

    /// Returns width (W) — fourth dimension of a 4D NCHW tensor.
    int width() const {
        assert(ndim() == 4 && "width() requires a 4D tensor");
        return shape_[3];
    }

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

    /// Compute flat index from 4D NCHW indices.
    /// Optimized hot path — used by all CNN layer computations.
    int flat_index_4d(int n, int c, int h, int w) const {
        assert(n >= 0 && n < shape_[0] && "Batch index out of bounds");
        assert(c >= 0 && c < shape_[1] && "Channel index out of bounds");
        assert(h >= 0 && h < shape_[2] && "Height index out of bounds");
        assert(w >= 0 && w < shape_[3] && "Width index out of bounds");
        return n * strides_[0] + c * strides_[1] + h * strides_[2] + w * strides_[3];
    }

    /// Compute flat index from generic index vector.
    int flat_index(const std::vector<int>& indices) const {
        assert(static_cast<int>(indices.size()) == ndim() &&
               "Number of indices must match tensor dimensions");
        int idx = 0;
        for (int i = 0; i < ndim(); ++i) {
            assert(indices[static_cast<size_t>(i)] >= 0 &&
                   indices[static_cast<size_t>(i)] < shape_[static_cast<size_t>(i)] &&
                   "Index out of bounds");
            idx += indices[static_cast<size_t>(i)] * strides_[static_cast<size_t>(i)];
        }
        return idx;
    }

    // ========================================================================
    // Data members
    // ========================================================================

    std::vector<float> data_;    ///< Contiguous NCHW storage
    std::vector<int> shape_;     ///< Dimension sizes, e.g., {N, C, H, W}
    std::vector<int> strides_;   ///< Strides for each dimension
};

}  // namespace cnn
