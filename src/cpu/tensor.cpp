#include "tensor.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
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
// Static factory methods
// ============================================================================

Tensor Tensor::zeros(const std::vector<int>& shape) {
    // Constructor already zero-initializes
    return Tensor(shape);
}

Tensor Tensor::ones(const std::vector<int>& shape) {
    Tensor t(shape);
    t.fill(1.0f);
    return t;
}

Tensor Tensor::full(const std::vector<int>& shape, float value) {
    Tensor t(shape);
    t.fill(value);
    return t;
}

Tensor Tensor::rand(const std::vector<int>& shape, unsigned int seed) {
    Tensor t(shape);

    // Use random_device for non-deterministic seed if seed == 0
    std::mt19937 gen;
    if (seed != 0) {
        gen.seed(seed);
    } else {
        std::random_device rd;
        gen.seed(rd());
    }

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < t.num_elements(); ++i) {
        t[i] = dist(gen);
    }
    return t;
}

Tensor Tensor::randn(const std::vector<int>& shape, float mean, float stddev,
                     unsigned int seed) {
    Tensor t(shape);

    // Use random_device for non-deterministic seed if seed == 0
    std::mt19937 gen;
    if (seed != 0) {
        gen.seed(seed);
    } else {
        std::random_device rd;
        gen.seed(rd());
    }

    std::normal_distribution<float> dist(mean, stddev);
    for (int i = 0; i < t.num_elements(); ++i) {
        t[i] = dist(gen);
    }
    return t;
}

// ============================================================================
// Comparison
// ============================================================================

bool Tensor::allclose(const Tensor& other, float atol, float rtol) const {
    // Shapes must match
    if (shape_ != other.shape_) {
        return false;
    }

    // Element-wise check: |a - b| <= atol + rtol * |b|
    for (int i = 0; i < num_elements(); ++i) {
        float a = data_[static_cast<size_t>(i)];
        float b = other.data_[static_cast<size_t>(i)];
        float diff = std::fabs(a - b);
        float tol = atol + rtol * std::fabs(b);
        if (diff > tol) {
            return false;
        }
    }
    return true;
}

float Tensor::max_diff(const Tensor& other) const {
    if (shape_ != other.shape_) {
        return -1.0f;
    }

    float max_d = 0.0f;
    for (int i = 0; i < num_elements(); ++i) {
        float diff = std::fabs(data_[static_cast<size_t>(i)] -
                               other.data_[static_cast<size_t>(i)]);
        if (diff > max_d) {
            max_d = diff;
        }
    }
    return max_d;
}

bool Tensor::operator==(const Tensor& other) const {
    if (shape_ != other.shape_) {
        return false;
    }
    return data_ == other.data_;
}

// ============================================================================
// Reductions
// ============================================================================

int Tensor::argmax() const {
    if (num_elements() == 0) {
        throw std::runtime_error("argmax: tensor is empty");
    }

    int max_idx = 0;
    float max_val = data_[0];
    for (int i = 1; i < num_elements(); ++i) {
        if (data_[static_cast<size_t>(i)] > max_val) {
            max_val = data_[static_cast<size_t>(i)];
            max_idx = i;
        }
    }
    return max_idx;
}

float Tensor::max_value() const {
    if (num_elements() == 0) {
        throw std::runtime_error("max_value: tensor is empty");
    }

    float max_val = data_[0];
    for (int i = 1; i < num_elements(); ++i) {
        if (data_[static_cast<size_t>(i)] > max_val) {
            max_val = data_[static_cast<size_t>(i)];
        }
    }
    return max_val;
}

std::vector<int> Tensor::argmax_per_row() const {
    if (ndim() != 2) {
        throw std::invalid_argument(
            "argmax_per_row: tensor must be 2D (N, D), got " +
            std::to_string(ndim()) + "D");
    }

    int rows = dim(0);
    int cols = dim(1);
    std::vector<int> result(static_cast<size_t>(rows));

    for (int n = 0; n < rows; ++n) {
        int max_idx = 0;
        float max_val = at(n, 0);
        for (int j = 1; j < cols; ++j) {
            if (at(n, j) > max_val) {
                max_val = at(n, j);
                max_idx = j;
            }
        }
        result[static_cast<size_t>(n)] = max_idx;
    }

    return result;
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
// Shape manipulation
// ============================================================================

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    // Resolve -1 dimension (at most one allowed)
    std::vector<int> resolved = new_shape;
    int neg_one_idx = -1;
    int known_product = 1;

    for (int i = 0; i < static_cast<int>(resolved.size()); ++i) {
        if (resolved[static_cast<size_t>(i)] == -1) {
            if (neg_one_idx != -1) {
                throw std::invalid_argument(
                    "Tensor::reshape: at most one dimension can be -1");
            }
            neg_one_idx = i;
        } else if (resolved[static_cast<size_t>(i)] <= 0) {
            throw std::invalid_argument(
                "Tensor::reshape: dimensions must be positive or -1, got " +
                std::to_string(resolved[static_cast<size_t>(i)]));
        } else {
            known_product *= resolved[static_cast<size_t>(i)];
        }
    }

    // Infer the -1 dimension
    if (neg_one_idx != -1) {
        if (known_product == 0 || num_elements() % known_product != 0) {
            throw std::invalid_argument(
                "Tensor::reshape: cannot infer dimension -1 for shape with " +
                std::to_string(num_elements()) + " elements");
        }
        resolved[static_cast<size_t>(neg_one_idx)] = num_elements() / known_product;
    }

    // Validate total element count
    int total = std::accumulate(resolved.begin(), resolved.end(), 1, std::multiplies<int>());
    if (total != num_elements()) {
        throw std::invalid_argument(
            "Tensor::reshape: new shape has " + std::to_string(total) +
            " elements, but tensor has " + std::to_string(num_elements()));
    }

    // Create new tensor with same data but different shape
    // Data is copied (tensors don't share memory — simple ownership model)
    return Tensor(resolved, std::vector<float>(data_.begin(), data_.end()));
}

Tensor Tensor::flatten() const {
    if (ndim() <= 1) {
        // 0D or 1D tensor → {1, num_elements}
        return reshape({1, num_elements()});
    }

    // ND tensor → {dim(0), remaining_elements}
    // For NCHW: {N, C*H*W}
    int batch = shape_[0];
    int features = num_elements() / batch;
    return reshape({batch, features});
}

// ============================================================================
// Debug and printing
// ============================================================================

std::string Tensor::shape_string() const {
    std::ostringstream ss;
    ss << "(";
    for (int i = 0; i < ndim(); ++i) {
        if (i > 0) ss << ", ";
        ss << shape_[static_cast<size_t>(i)];
    }
    ss << ")";
    return ss.str();
}

std::string Tensor::to_string(int max_elements) const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);

    // Header: shape and element count
    ss << "Tensor" << shape_string();
    ss << " [" << num_elements() << " elements, FP32]\n";

    if (empty()) {
        ss << "  (empty)";
        return ss.str();
    }

    int n = num_elements();
    int show = max_elements;

    if (n <= show) {
        // Show all elements
        ss << "  [";
        for (int i = 0; i < n; ++i) {
            if (i > 0) ss << ", ";
            ss << data_[static_cast<size_t>(i)];
        }
        ss << "]";
    } else {
        // Truncated: show first and last few elements
        int half = show / 2;
        ss << "  [";
        for (int i = 0; i < half; ++i) {
            if (i > 0) ss << ", ";
            ss << data_[static_cast<size_t>(i)];
        }
        ss << ", ... , ";
        for (int i = n - half; i < n; ++i) {
            if (i > n - half) ss << ", ";
            ss << data_[static_cast<size_t>(i)];
        }
        ss << "]";
    }

    return ss.str();
}

void Tensor::print(const std::string& name, int max_elements) const {
    if (!name.empty()) {
        std::cout << name << ": ";
    }
    std::cout << to_string(max_elements) << std::endl;
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << t.to_string();
    return os;
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
