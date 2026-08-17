#include "kernels.h"
#include "cuda_utils.h"
#include "gpu_tensor.h"

#include <cuda_runtime.h>

// ============================================================================
// Naive Conv2D CUDA Kernel — One Thread Per Output Element
// ============================================================================
//
// Each thread computes one element of the output tensor by iterating over
// all input channels and kernel spatial dimensions. Direct global memory
// reads with no shared memory optimization.
//
// This is the correctness baseline. Shared-memory tiling comes in Phase 7.
//
// Memory layout: NCHW (contiguous, row-major)
//   Flat index for input[n][c][h][w] = n*(C*H*W) + c*(H*W) + h*W + w
//
// ============================================================================

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width,
    int kern_height, int kern_width,
    int stride, int padding
) {
    // Each thread computes one output element
    // Total output elements: batch_size * out_channels * out_height * out_width
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * out_height * out_width;

    if (idx >= total_outputs) return;

    // Decompose flat index into (n, oc, oh, ow)
    int ow = idx % out_width;
    int tmp = idx / out_width;
    int oh = tmp % out_height;
    tmp = tmp / out_height;
    int oc = tmp % out_channels;
    int n = tmp / out_channels;

    // Start with bias
    float sum = bias[oc];

    // Accumulate convolution sum over input channels and kernel spatial dims
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kern_height; ++kh) {
            for (int kw = 0; kw < kern_width; ++kw) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;

                // Zero-padding: skip out-of-bounds
                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                    int input_idx = n * (in_channels * in_height * in_width) +
                                    ic * (in_height * in_width) +
                                    ih * in_width + iw;
                    int kernel_idx = oc * (in_channels * kern_height * kern_width) +
                                     ic * (kern_height * kern_width) +
                                     kh * kern_width + kw;
                    sum += input[input_idx] * kernel[kernel_idx];
                }
            }
        }
    }

    output[idx] = sum;
}

// ============================================================================
// Conv2D GPU Launch Wrapper
// ============================================================================

GpuTensor conv2d_gpu(
    const GpuTensor& input,
    const GpuTensor& weights,
    const GpuTensor& bias,
    int stride,
    int padding
) {
    // Extract dimensions from input: (N, C_in, H, W)
    int batch_size = input.dim(0);
    int in_channels = input.dim(1);
    int in_height = input.dim(2);
    int in_width = input.dim(3);

    // Extract dimensions from weights: (C_out, C_in, KH, KW)
    int out_channels = weights.dim(0);
    int kern_height = weights.dim(2);
    int kern_width = weights.dim(3);

    // Compute output spatial dimensions
    int out_height = (in_height + 2 * padding - kern_height) / stride + 1;
    int out_width = (in_width + 2 * padding - kern_width) / stride + 1;

    // Allocate output: (N, C_out, OH, OW)
    GpuTensor output(std::vector<int>{batch_size, out_channels, out_height, out_width});

    // Launch kernel: one thread per output element
    int total = batch_size * out_channels * out_height * out_width;
    int block_size = 256;
    int grid_size = cuda_grid_size(total, block_size);

    conv2d_kernel<<<grid_size, block_size>>>(
        input.data(), weights.data(), bias.data(), output.data(),
        batch_size,
        in_channels, in_height, in_width,
        out_channels, out_height, out_width,
        kern_height, kern_width,
        stride, padding
    );
    KERNEL_CHECK();

    return output;
}
