#include "kernels.h"
#include "cuda_utils.h"
#include "gpu_tensor.h"

#include <cuda_runtime.h>

// ============================================================================
// Tiled Conv2D CUDA Kernel — Shared Memory Optimization
// ============================================================================
//
// Reduces global memory accesses by loading input tiles into shared memory.
//
// Strategy:
//   - 2D thread blocks map to output spatial tiles (TILE_W × TILE_H)
//   - Each block computes one tile of one output channel for one batch sample
//   - For each input channel, the block cooperatively loads the required
//     input region (tile + kernel halo) into shared memory, then each thread
//     computes its convolution sum from shared data
//   - The kernel weights are small enough (3×3 = 9 floats per ic/oc pair)
//     to be read from global memory (L1/L2 cached)
//
// Memory savings:
//   Each input pixel is read once from global → shared, then reused by
//   multiple threads in the tile. For a 3×3 kernel, each pixel participates
//   in up to 9 output computations → ~9× reduction in global reads.
//
// ============================================================================

// Tile dimensions for output spatial region
#define TILE_W 16
#define TILE_H 16

__global__ void conv2d_tiled_kernel(
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
    // Block (bx, by) computes output tile at spatial offset (bx*TILE_W, by*TILE_H)
    // blockIdx.z encodes: batch_sample * out_channels + out_channel
    int ow_start = blockIdx.x * TILE_W;
    int oh_start = blockIdx.y * TILE_H;
    int oc = blockIdx.z % out_channels;
    int n = blockIdx.z / out_channels;

    int tx = threadIdx.x;  // within tile [0, TILE_W)
    int ty = threadIdx.y;  // within tile [0, TILE_H)

    int ow = ow_start + tx;
    int oh = oh_start + ty;

    // Shared memory for input tile + halo
    // Tile covers output [oh_start .. oh_start+TILE_H-1] × [ow_start .. ow_start+TILE_W-1]
    // Input region needed: [oh_start*stride - padding .. (oh_start+TILE_H-1)*stride - padding + kern_height-1]
    //                      [ow_start*stride - padding .. (ow_start+TILE_W-1)*stride - padding + kern_width-1]
    // Shared memory size: (TILE_H * stride + kern_height - 1) × (TILE_W * stride + kern_width - 1)

    int shared_h = TILE_H * stride + kern_height - 1;
    int shared_w = TILE_W * stride + kern_width - 1;

    extern __shared__ float s_input[];

    // Starting input coordinates for this tile
    int ih_start = oh_start * stride - padding;
    int iw_start = ow_start * stride - padding;

    float sum = 0.0f;

    // Accumulate over input channels
    for (int ic = 0; ic < in_channels; ++ic) {
        // Cooperatively load input tile + halo into shared memory
        int total_shared = shared_h * shared_w;
        int threads_per_block = TILE_W * TILE_H;
        int tid = ty * TILE_W + tx;

        for (int idx = tid; idx < total_shared; idx += threads_per_block) {
            int sy = idx / shared_w;
            int sx = idx % shared_w;
            int ih = ih_start + sy;
            int iw = iw_start + sx;

            float val = 0.0f;
            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                val = input[n * (in_channels * in_height * in_width) +
                            ic * (in_height * in_width) +
                            ih * in_width + iw];
            }
            s_input[idx] = val;
        }
        __syncthreads();

        // Compute convolution from shared memory
        if (oh < out_height && ow < out_width) {
            for (int kh = 0; kh < kern_height; ++kh) {
                for (int kw = 0; kw < kern_width; ++kw) {
                    // Local shared memory coordinates
                    int sy = ty * stride + kh;
                    int sx = tx * stride + kw;
                    float input_val = s_input[sy * shared_w + sx];

                    int kernel_idx = oc * (in_channels * kern_height * kern_width) +
                                     ic * (kern_height * kern_width) +
                                     kh * kern_width + kw;
                    sum += input_val * kernel[kernel_idx];
                }
            }
        }
        __syncthreads();  // Ensure all threads done before next ic overwrites shared
    }

    // Write output with bias
    if (oh < out_height && ow < out_width) {
        int out_idx = n * (out_channels * out_height * out_width) +
                      oc * (out_height * out_width) +
                      oh * out_width + ow;
        output[out_idx] = sum + bias[oc];
    }
}

// ============================================================================
// Tiled Conv2D GPU Launch Wrapper
// ============================================================================

GpuTensor conv2d_tiled_gpu(const GpuTensor& input, const GpuTensor& weights,
                            const GpuTensor& bias, int stride, int padding) {
    int batch_size = input.dim(0);
    int in_channels = input.dim(1);
    int in_height = input.dim(2);
    int in_width = input.dim(3);

    int out_channels = weights.dim(0);
    int kern_height = weights.dim(2);
    int kern_width = weights.dim(3);

    int out_height = (in_height + 2 * padding - kern_height) / stride + 1;
    int out_width = (in_width + 2 * padding - kern_width) / stride + 1;

    GpuTensor output(std::vector<int>{batch_size, out_channels, out_height, out_width});

    // Grid: tiles across spatial dims × (batch * out_channels)
    dim3 block(TILE_W, TILE_H);
    dim3 grid(
        (out_width + TILE_W - 1) / TILE_W,
        (out_height + TILE_H - 1) / TILE_H,
        batch_size * out_channels
    );

    // Shared memory: input tile + halo for one input channel
    int shared_h = TILE_H * stride + kern_height - 1;
    int shared_w = TILE_W * stride + kern_width - 1;
    size_t shared_mem = static_cast<size_t>(shared_h * shared_w) * sizeof(float);

    conv2d_tiled_kernel<<<grid, block, shared_mem>>>(
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
