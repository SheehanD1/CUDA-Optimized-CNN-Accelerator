#include "kernels.h"
#include "cuda_utils.h"
#include "gpu_tensor.h"

#include <cuda_runtime.h>

#include <cfloat>

// ============================================================================
// Softmax CUDA Kernel — One Block Per Row, Shared Memory Reductions
// ============================================================================
//
// Numerically stable softmax per row (per sample):
//   1. Find max_val = max over j of input[n][j]
//   2. Compute exp_val[j] = exp(input[n][j] - max_val)
//   3. Compute sum_exp = sum of exp_val[j]
//   4. output[n][j] = exp_val[j] / sum_exp
//
// Each block handles one row (one sample). Threads within the block
// cooperate on the reductions using shared memory.
//
// For our CNN, num_classes = 10, so this kernel is efficient even
// with a small number of threads per block.
//
// ============================================================================

__global__ void softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_classes
) {
    // Each block handles one row
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* row_input = input + row * num_classes;
    float* row_output = output + row * num_classes;

    // Shared memory for reductions
    extern __shared__ float sdata[];

    // ========================================================================
    // Step 1: Find max value in this row (parallel reduction)
    // ========================================================================

    float local_max = -FLT_MAX;
    for (int j = tid; j < num_classes; j += blockDim.x) {
        local_max = fmaxf(local_max, row_input[j]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    // Tree reduction for max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    // ========================================================================
    // Step 2 & 3: Compute exp(x - max) and sum
    // ========================================================================

    float local_sum = 0.0f;
    for (int j = tid; j < num_classes; j += blockDim.x) {
        float exp_val = expf(row_input[j] - max_val);
        row_output[j] = exp_val;  // Store temporarily
        local_sum += exp_val;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Tree reduction for sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float sum_exp = sdata[0];
    __syncthreads();

    // ========================================================================
    // Step 4: Normalize
    // ========================================================================

    for (int j = tid; j < num_classes; j += blockDim.x) {
        row_output[j] /= sum_exp;
    }
}

// ============================================================================
// Softmax GPU Launch Wrapper
// ============================================================================

GpuTensor softmax_gpu(const GpuTensor& input) {
    int batch_size = input.dim(0);
    int num_classes = input.dim(1);

    GpuTensor output(input.shape());

    // One block per row; threads per block = next power of 2 >= num_classes,
    // capped at 256
    int threads = 1;
    while (threads < num_classes) threads *= 2;
    if (threads > 256) threads = 256;

    // Shared memory: one float per thread for reductions
    size_t shared_mem = static_cast<size_t>(threads) * sizeof(float);

    softmax_kernel<<<batch_size, threads, shared_mem>>>(
        input.data(), output.data(), num_classes
    );
    KERNEL_CHECK();

    return output;
}
