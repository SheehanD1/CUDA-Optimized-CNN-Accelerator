#include "gpu_tensor.h"
#include "gpu_memory.h"

#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

// ============================================================================
// Construction / Destruction
// ============================================================================

GpuTensor::GpuTensor(const std::vector<int>& shape)
    : d_data_(nullptr), shape_(shape) {
    int n = num_elements();
    if (n > 0) {
        d_data_ = gpu_alloc<float>(static_cast<size_t>(n));
        gpu_memset_zero(d_data_, static_cast<size_t>(n));
    }
}

GpuTensor::GpuTensor(const Tensor& cpu_tensor)
    : d_data_(nullptr), shape_(cpu_tensor.shape()) {
    int n = num_elements();
    if (n > 0) {
        d_data_ = gpu_alloc<float>(static_cast<size_t>(n));
        host_to_device(d_data_, cpu_tensor.data(), static_cast<size_t>(n));
    }
}

GpuTensor::~GpuTensor() {
    if (d_data_ != nullptr) {
        gpu_free(d_data_);
        d_data_ = nullptr;
    }
}

// ============================================================================
// Move Semantics
// ============================================================================

GpuTensor::GpuTensor(GpuTensor&& other) noexcept
    : d_data_(other.d_data_), shape_(std::move(other.shape_)) {
    other.d_data_ = nullptr;
}

GpuTensor& GpuTensor::operator=(GpuTensor&& other) noexcept {
    if (this != &other) {
        // Free existing allocation
        if (d_data_ != nullptr) {
            gpu_free(d_data_);
        }
        // Take ownership
        d_data_ = other.d_data_;
        shape_ = std::move(other.shape_);
        other.d_data_ = nullptr;
    }
    return *this;
}

// ============================================================================
// Data Transfer
// ============================================================================

void GpuTensor::upload(const Tensor& cpu_tensor) {
    int n = num_elements();
    if (cpu_tensor.num_elements() != n) {
        throw std::invalid_argument(
            "GpuTensor::upload: element count mismatch. GPU tensor has " +
            std::to_string(n) + " elements, CPU tensor has " +
            std::to_string(cpu_tensor.num_elements()));
    }
    if (n > 0) {
        host_to_device(d_data_, cpu_tensor.data(), static_cast<size_t>(n));
    }
}

Tensor GpuTensor::download() const {
    Tensor cpu_tensor(shape_);
    int n = num_elements();
    if (n > 0) {
        device_to_host(cpu_tensor.data(), d_data_, static_cast<size_t>(n));
    }
    return cpu_tensor;
}

// ============================================================================
// Accessors
// ============================================================================

int GpuTensor::num_elements() const {
    if (shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
}

void GpuTensor::reshape(const std::vector<int>& new_shape) {
    // Compute new total elements
    int new_total = 1;
    for (int d : new_shape) {
        new_total *= d;
    }

    // Validate: must have same total elements
    int old_total = num_elements();
    if (new_total != old_total) {
        throw std::invalid_argument(
            "GpuTensor::reshape: new shape has " + std::to_string(new_total) +
            " elements, but tensor has " + std::to_string(old_total));
    }

    // Just update metadata — no data movement
    shape_ = new_shape;
}
