#pragma once

// ============================================================================
// GpuTensor — RAII Wrapper for Device Memory
// ============================================================================
//
// Owns a float* device pointer with shape metadata. Provides RAII semantics:
//   - Constructor allocates GPU memory via cudaMalloc
//   - Destructor frees via cudaFree
//   - Move-only (no copies) to prevent double-free
//   - upload(Tensor&) to transfer CPU data to GPU
//   - download() to transfer GPU data back to CPU Tensor
//
// This is the GPU equivalent of the CPU Tensor class, used to keep data
// on the device between kernel launches and minimize H↔D transfers.
//
// Usage:
//   Tensor cpu_data({1, 1, 28, 28}, input_data);
//   GpuTensor gpu_data(cpu_data);              // Upload to GPU
//   // ... run kernels on gpu_data.data() ...
//   Tensor result = gpu_data.download();       // Download back to CPU
//
// ============================================================================

#include "tensor.h"

#include <vector>

class GpuTensor {
public:
    // ========================================================================
    // Construction / Destruction
    // ========================================================================

    /// Allocate device memory for a tensor of the given shape.
    /// Memory is zero-initialized.
    explicit GpuTensor(const std::vector<int>& shape);

    /// Allocate device memory and upload CPU tensor data.
    /// Shape is copied from the CPU tensor.
    explicit GpuTensor(const Tensor& cpu_tensor);

    /// Destructor: frees device memory.
    ~GpuTensor();

    // ========================================================================
    // Move Semantics (no copies)
    // ========================================================================

    GpuTensor(GpuTensor&& other) noexcept;
    GpuTensor& operator=(GpuTensor&& other) noexcept;

    // Delete copy operations to prevent double-free
    GpuTensor(const GpuTensor&) = delete;
    GpuTensor& operator=(const GpuTensor&) = delete;

    // ========================================================================
    // Data Transfer
    // ========================================================================

    /// Upload CPU tensor data to this GPU tensor.
    /// The CPU tensor must have the same number of elements.
    void upload(const Tensor& cpu_tensor);

    /// Download GPU tensor data to a new CPU tensor.
    Tensor download() const;

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Returns a device pointer to the underlying GPU memory.
    float* data() { return d_data_; }
    const float* data() const { return d_data_; }

    /// Returns the tensor shape.
    const std::vector<int>& shape() const { return shape_; }

    /// Returns the number of dimensions.
    int ndim() const { return static_cast<int>(shape_.size()); }

    /// Returns the size of dimension d.
    int dim(int d) const { return shape_[static_cast<size_t>(d)]; }

    /// Returns the total number of elements.
    int num_elements() const;

    /// Returns the size in bytes of the GPU allocation.
    size_t size_bytes() const {
        return static_cast<size_t>(num_elements()) * sizeof(float);
    }

    /// Reshape the tensor in-place (metadata only — no data movement).
    /// The new shape must have the same total number of elements.
    /// This is used for flatten: (N, C, H, W) → (N, C*H*W) with zero cost.
    void reshape(const std::vector<int>& new_shape);

    /// Returns true if this tensor owns valid device memory.
    bool is_valid() const { return d_data_ != nullptr; }

private:
    float* d_data_ = nullptr;        ///< Device pointer (owned)
    std::vector<int> shape_;         ///< Tensor dimensions (e.g., {N, C, H, W})
};
