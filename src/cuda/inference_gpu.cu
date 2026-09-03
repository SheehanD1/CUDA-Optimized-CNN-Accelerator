#include "inference.h"

#include "gpu_tensor.h"
#include "kernels.h"

// ============================================================================
// GPU Inference Pipeline — Full Forward Pass on Device
// ============================================================================
//
// Architecture (identical to CPU pipeline):
//   Input (N, 1, 28, 28)
//     → Conv1(1→8, 3×3, pad=1) → ReLU → MaxPool(2×2)    → (N, 8, 14, 14)
//     → Conv2(8→16, 3×3, pad=1) → ReLU → MaxPool(2×2)   → (N, 16, 7, 7)
//     → Flatten                                           → (N, 784)
//     → Dense1(784→120) → ReLU                            → (N, 120)
//     → Dense2(120→10)                                    → (N, 10)
//     → Softmax                                           → (N, 10)
//
// Data flow: CPU input → GPU → all kernels → GPU output → CPU
// Only 2 transfer points: upload at start, download at end.
// Model weights are uploaded once per inference call.
//
// ============================================================================

Tensor gpu_inference(const Model& model, const Tensor& input) {
    // ========================================================================
    // Upload input and model weights to GPU
    // ========================================================================

    GpuTensor g_input(input);

    GpuTensor g_conv1_w(model.conv1_weights);
    GpuTensor g_conv1_b(model.conv1_bias);
    GpuTensor g_conv2_w(model.conv2_weights);
    GpuTensor g_conv2_b(model.conv2_bias);
    GpuTensor g_dense1_w(model.dense1_weights);
    GpuTensor g_dense1_b(model.dense1_bias);
    GpuTensor g_dense2_w(model.dense2_weights);
    GpuTensor g_dense2_b(model.dense2_bias);

    // ========================================================================
    // Block 1: Conv1 → ReLU → MaxPool
    // ========================================================================
    // Input:  (N, 1, 28, 28)
    // Conv1:  (N, 8, 28, 28)
    // ReLU:   (N, 8, 28, 28)
    // Pool:   (N, 8, 14, 14)

    GpuTensor x = conv2d_tiled_gpu(g_input, g_conv1_w, g_conv1_b, 1, 1);
    x = relu_gpu(x);
    x = maxpool2d_gpu(x, 2);

    // ========================================================================
    // Block 2: Conv2 → ReLU → MaxPool
    // ========================================================================
    // Input:  (N, 8, 14, 14)
    // Conv2:  (N, 16, 14, 14)
    // ReLU:   (N, 16, 14, 14)
    // Pool:   (N, 16, 7, 7)

    x = conv2d_tiled_gpu(x, g_conv2_w, g_conv2_b, 1, 1);
    x = relu_gpu(x);
    x = maxpool2d_gpu(x, 2);

    // ========================================================================
    // Flatten: (N, 16, 7, 7) → (N, 784)
    // ========================================================================
    // On GPU, flatten is just a reshape — no data movement needed.
    // The underlying memory is already contiguous in row-major order.

    int batch_size = x.dim(0);
    int flat_size = x.num_elements() / batch_size;
    // Create a new GpuTensor with reshaped metadata pointing to same data
    // Since GpuTensor is move-only and owns memory, we download and re-upload
    // with the new shape. This is a small overhead for correctness.
    {
        Tensor flat_cpu = x.download();
        flat_cpu = flat_cpu.reshape({batch_size, flat_size});
        x = GpuTensor(flat_cpu);
    }

    // ========================================================================
    // Dense Block: Dense1 → ReLU → Dense2
    // ========================================================================
    // Dense1: (N, 784) → (N, 120)
    // ReLU:   (N, 120)
    // Dense2: (N, 120) → (N, 10)

    x = dense_gpu(x, g_dense1_w, g_dense1_b);
    x = relu_gpu(x);
    x = dense_gpu(x, g_dense2_w, g_dense2_b);

    // ========================================================================
    // Softmax: (N, 10) → (N, 10) probabilities
    // ========================================================================

    x = softmax_gpu(x);

    // ========================================================================
    // Download result back to CPU
    // ========================================================================

    return x.download();
}

std::vector<int> gpu_predict(const Model& model, const Tensor& input) {
    Tensor probabilities = gpu_inference(model, input);
    return probabilities.argmax_per_row();
}
