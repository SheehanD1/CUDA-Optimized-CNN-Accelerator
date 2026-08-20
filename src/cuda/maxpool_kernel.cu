#include "kernels.h"
#include "cuda_utils.h"
#include "gpu_tensor.h"

#include <cuda_runtime.h>

#include <cfloat>

// ============================================================================
// Naive MaxPool2D CUDA Kernel — One Thread Per Output Element
// ============================================================================
//
// Each thread computes one output element by scanning a pool_size x pool_size
// window and taking the maximum value. Direct global memory reads.
//
// Memory layout: NCHW (contiguous, row-major)
//
// ============================================================================

__global__ void maxpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels, int in_height, int in_width,
    int out_height, int out_width,
    int pool_size, int stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_height * out_width;

    if (idx >= total) return;

    // Decompose flat index into (n, c, oh, ow)
    int ow = idx % out_width;
    int tmp = idx / out_width;
    int oh = tmp % out_height;
    tmp = tmp / out_height;
    int c = tmp % channels;
    int n = tmp / channels;

    // Find maximum in the pooling window
    float max_val = -FLT_MAX;

    for (int ph = 0; ph < pool_size; ++ph) {
        for (int pw = 0; pw < pool_size; ++pw) {
            int ih = oh * stride + ph;
            int iw = ow * stride + pw;
            int input_idx = n * (channels * in_height * in_width) +
                            c * (in_height * in_width) +
                            ih * in_width + iw;
            float val = input[input_idx];
            max_val = fmaxf(max_val, val);
        }
    }

    output[idx] = max_val;
}

// ============================================================================
// MaxPool2D GPU Launch Wrapper
// ============================================================================

GpuTensor maxpool2d_gpu(const GpuTensor& input, int pool_size, int stride) {
    // Default stride = pool_size (non-overlapping windows)
    if (stride <= 0) {
        stride = pool_size;
    }

    int batch_size = input.dim(0);
    int channels = input.dim(1);
    int in_height = input.dim(2);
    int in_width = input.dim(3);

    int out_height = (in_height - pool_size) / stride + 1;
    int out_width = (in_width - pool_size) / stride + 1;

    GpuTensor output(std::vector<int>{batch_size, channels, out_height, out_width});

    int total = batch_size * channels * out_height * out_width;
    int block_size = 256;
    int grid_size = cuda_grid_size(total, block_size);

    maxpool2d_kernel<<<grid_size, block_size>>>(
        input.data(), output.data(),
        batch_size,
        channels, in_height, in_width,
        out_height, out_width,
        pool_size, stride
    );
    KERNEL_CHECK();

    return output;
}
