#include "kernels.h"
#include "cuda_utils.h"
#include "gpu_tensor.h"

#include <cuda_runtime.h>

// ============================================================================
// Naive Dense (Fully Connected) CUDA Kernel — One Thread Per Output Element
// ============================================================================
//
// Computes: output[n][j] = bias[j] + sum_k(input[n][k] * weights[j][k])
//
// Each thread computes one output element via a dot product over in_features.
// This is equivalent to: output = input * weights^T + bias
//
// Memory layout: row-major 2D
//   input:   (N, in_features)
//   weights: (out_features, in_features)
//   bias:    (out_features,)
//   output:  (N, out_features)
//
// ============================================================================

__global__ void dense_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_features;

    if (idx >= total) return;

    // Decompose into (n, j) — sample index and output feature
    int j = idx % out_features;
    int n = idx / out_features;

    // Dot product: input[n] · weights[j] + bias[j]
    float sum = bias[j];
    for (int k = 0; k < in_features; ++k) {
        sum += input[n * in_features + k] * weights[j * in_features + k];
    }

    output[idx] = sum;
}

// ============================================================================
// Dense GPU Launch Wrapper
// ============================================================================

GpuTensor dense_gpu(const GpuTensor& input, const GpuTensor& weights,
                     const GpuTensor& bias) {
    int batch_size = input.dim(0);
    int in_features = input.dim(1);
    int out_features = weights.dim(0);

    GpuTensor output(std::vector<int>{batch_size, out_features});

    int total = batch_size * out_features;
    int block_size = 256;
    int grid_size = cuda_grid_size(total, block_size);

    dense_kernel<<<grid_size, block_size>>>(
        input.data(), weights.data(), bias.data(), output.data(),
        batch_size, in_features, out_features
    );
    KERNEL_CHECK();

    return output;
}
