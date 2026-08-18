#include "kernels.h"
#include "cuda_utils.h"
#include "gpu_tensor.h"

#include <cuda_runtime.h>

// ============================================================================
// ReLU CUDA Kernel — One Thread Per Element
// ============================================================================
//
// Element-wise: output[i] = max(0, input[i])
//
// This is the simplest possible CUDA kernel — pure memory-bound with
// no shared memory or reduction. One thread per element.
//
// ============================================================================

__global__ void relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

// ============================================================================
// ReLU GPU Launch Wrapper
// ============================================================================

GpuTensor relu_gpu(const GpuTensor& input) {
    int total = input.num_elements();
    GpuTensor output(input.shape());

    int block_size = 256;
    int grid_size = cuda_grid_size(total, block_size);

    relu_kernel<<<grid_size, block_size>>>(
        input.data(), output.data(), total
    );
    KERNEL_CHECK();

    return output;
}
