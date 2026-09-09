# Architecture Overview

> **CUDA-Optimized CNN Accelerator** — A high-performance CNN inference engine built from scratch in C++17 and CUDA, featuring custom convolution kernels, shared-memory tiling, and occupancy-aware thread scheduling.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Network Architecture](#network-architecture)
3. [Data Flow & Tensor Shapes](#data-flow--tensor-shapes)
4. [Module Structure](#module-structure)
5. [Memory Management](#memory-management)
6. [Execution Pipelines](#execution-pipelines)
7. [Design Principles](#design-principles)
8. [Key Abstractions](#key-abstractions)

---

## System Overview

The system implements a complete CNN inference pipeline with two execution backends:

```
                    ┌─────────────────────────────────────────────┐
                    │              CNN Accelerator                │
                    │                                             │
  MNIST Image ───► │  ┌──────────┐         ┌──────────────────┐  │ ───► Prediction
  (28×28 px)       │  │ CPU      │         │ GPU              │  │      (digit 0-9)
                    │  │ Backend  │         │ Backend          │  │
                    │  │ (C++17)  │         │ (CUDA Kernels)   │  │
                    │  └──────────┘         └──────────────────┘  │
                    │        │                      │             │
                    │        ▼                      ▼             │
                    │  ┌─────────────────────────────────────┐   │
                    │  │         Benchmark Suite              │   │
                    │  │   CPU vs Naive GPU vs Optimized GPU  │   │
                    │  └─────────────────────────────────────┘   │
                    └─────────────────────────────────────────────┘
```

Both backends share the same `Tensor` data structure and produce identical outputs (within floating-point tolerance). This enables correctness validation and performance comparison.

---

## Network Architecture

A deeper LeNet-style CNN with two convolutional stages, chosen to exercise the GPU pipeline more thoroughly and produce meaningful benchmark data.

```
Input (1×28×28)
       │
       ▼
┌──────────────┐
│  Conv2D      │  1 → 8 channels, 3×3 kernel, stride=1, padding=1
│  (72 params) │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  ReLU        │  Element-wise max(0, x)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  MaxPool2D   │  2×2 window, stride=2
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Conv2D      │  8 → 16 channels, 3×3 kernel, stride=1, padding=1
│  (1168 params)│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  ReLU        │  Element-wise max(0, x)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  MaxPool2D   │  2×2 window, stride=2
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Flatten     │  16×7×7 → 784
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Dense       │  784 → 120 neurons
│  (94200 params)│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  ReLU        │  Element-wise max(0, x)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Dense       │  120 → 10 neurons (output classes)
│  (1210 params)│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Softmax     │  Numerically stable (subtract max before exp)
└──────┬───────┘
       │
       ▼
  Prediction
  (digit 0-9)
```

**Total Parameters:** ~96,650

---

## Data Flow & Tensor Shapes

All tensors use **NCHW layout** (Batch, Channels, Height, Width). This is the standard for GPU inference and enables coalesced memory access patterns in CUDA kernels.

| Layer | Operation | Output Shape | Output Size | Notes |
|-------|-----------|-------------|-------------|-------|
| Input | — | `(1, 1, 28, 28)` | 784 | Normalized to [0, 1] |
| Conv2D #1 | 3×3, pad=1, s=1 | `(1, 8, 28, 28)` | 6,272 | Same spatial dims (padding=1) |
| ReLU #1 | max(0, x) | `(1, 8, 28, 28)` | 6,272 | In-place capable |
| MaxPool #1 | 2×2, s=2 | `(1, 8, 14, 14)` | 1,568 | Spatial dims halved |
| Conv2D #2 | 3×3, pad=1, s=1 | `(1, 16, 14, 14)` | 3,136 | Channel doubling |
| ReLU #2 | max(0, x) | `(1, 16, 14, 14)` | 3,136 | In-place capable |
| MaxPool #2 | 2×2, s=2 | `(1, 16, 7, 7)` | 784 | Spatial dims halved |
| Flatten | reshape | `(1, 784)` | 784 | No data copy needed |
| Dense #1 | matmul + bias | `(1, 120)` | 120 | Weights: 784×120 |
| ReLU #3 | max(0, x) | `(1, 120)` | 120 | In-place capable |
| Dense #2 | matmul + bias | `(1, 10)` | 10 | Weights: 120×10 |
| Softmax | exp/normalize | `(1, 10)` | 10 | Probabilities sum to 1.0 |

---

## Module Structure

```
cnn-accelerator/
│
├── include/                    # Public headers (shared between CPU & GPU)
│   ├── tensor.h                # Core Tensor class (NCHW layout)
│   ├── model.h                 # Model struct (weights + architecture)
│   ├── inference.h             # CPU + GPU inference function declarations
│   ├── data_loader.h           # MNIST IDX format parser declarations
│   ├── gpu_tensor.h            # RAII GPU memory wrapper + async transfers
│   ├── gpu_model.h             # Cached GPU model (upload weights once)
│   ├── gpu_memory.h            # Host/device transfer function declarations
│   ├── kernels.h               # All GPU kernel wrapper declarations
│   ├── cuda_utils.h            # CUDA_CHECK / KERNEL_CHECK macros
│   ├── cuda_timer.h            # CudaTimer (cudaEvent-based profiling)
│   ├── cuda_stream.h           # CudaStream RAII wrapper
│   ├── device_info.h           # GPU device query declarations
│   └── layers/                 # CPU layer function declarations
│       ├── conv2d.h
│       ├── relu.h
│       ├── maxpool2d.h
│       ├── flatten.h
│       ├── dense.h
│       └── softmax.h
│
├── src/
│   ├── cpu/                    # CPU baseline (pure C++17)
│   │   ├── tensor.cpp          # Tensor implementation
│   │   ├── model.cpp           # Model load/save/Xavier init
│   │   ├── data_loader.cpp     # MNIST IDX format parser
│   │   ├── conv2d_cpu.cpp      # Naive 6-loop convolution
│   │   ├── relu_cpu.cpp        # Element-wise ReLU
│   │   ├── maxpool_cpu.cpp     # Sliding window max pooling
│   │   ├── flatten_cpu.cpp     # Reshape (N,C,H,W) → (N, C*H*W)
│   │   ├── dense_cpu.cpp       # Matrix-vector multiply + bias
│   │   ├── softmax_cpu.cpp     # Numerically stable softmax
│   │   ├── inference_cpu.cpp   # Full CPU inference chain
│   │   └── main_cpu.cpp        # CPU executable entry point
│   │
│   └── cuda/                   # GPU implementation (CUDA)
│       ├── memory.cu           # cudaMalloc/cudaMemcpy/cudaFree wrappers
│       ├── gpu_tensor.cu       # GpuTensor RAII + sync/async transfers
│       ├── gpu_model.cu        # GpuModel cached inference
│       ├── device_info.cu      # GPU capability query + printing
│       ├── conv2d_kernel.cu    # Naive conv2d (1 thread/output)
│       ├── conv2d_tiled_kernel.cu  # Shared-memory tiled conv2d (16×16 tiles)
│       ├── relu_kernel.cu      # ReLU kernel (1 thread/element)
│       ├── maxpool_kernel.cu   # MaxPool kernel (1 thread/output)
│       ├── dense_kernel.cu     # Dense/FC kernel (1 thread/output)
│       ├── softmax_kernel.cu   # Softmax (1 block/row, shared mem reduction)
│       ├── inference_gpu.cu    # Full GPU inference chain
│       └── main_gpu.cu         # GPU executable entry point
│
├── tests/                      # Google Test suites (100+ tests)
│   ├── test_tensor.cpp         # Tensor construction, indexing, reshape
│   ├── test_model.cpp          # Model save/load/Xavier init
│   ├── test_conv2d.cpp         # Conv2D CPU correctness
│   ├── test_relu.cpp           # ReLU CPU correctness
│   ├── test_maxpool.cpp        # MaxPool CPU correctness
│   ├── test_dense.cpp          # Dense CPU correctness
│   ├── test_softmax.cpp        # Softmax CPU correctness
│   ├── test_pipeline.cpp       # CPU end-to-end pipeline
│   ├── test_gpu_tensor.cpp     # GPU memory round-trip, RAII
│   ├── test_gpu_raii.cpp       # RAII stress tests (move, exception safety)
│   ├── test_conv2d_gpu.cpp     # Conv2D GPU vs CPU validation
│   ├── test_relu_gpu.cpp       # ReLU GPU vs CPU validation
│   ├── test_maxpool_gpu.cpp    # MaxPool GPU vs CPU validation
│   ├── test_dense_gpu.cpp      # Dense GPU vs CPU validation
│   ├── test_softmax_gpu.cpp    # Softmax GPU vs CPU validation
│   ├── test_gpu_pipeline.cpp   # GPU pipeline vs CPU pipeline
│   ├── test_conv2d_tiled.cpp   # Tiled vs naive vs CPU triple comparison
│   ├── test_optimized_pipeline.cpp  # GpuModel + streams + all optimizations
│   ├── test_timing.cpp         # CPU vs GPU timing comparison
│   ├── test_conv2d_perf.cpp    # Naive vs tiled Conv2D benchmark
│   └── test_final_benchmark.cpp  # Full 3-pipeline benchmark suite
│
├── scripts/
│   ├── build.bat               # Windows build script
│   ├── build.sh                # Linux/macOS build script
│   └── generate_conv2d_reference.py  # NumPy reference value generator
│
├── profiling/                  # Nsight profiling configs + results
│
└── docs/
    └── architecture.md         # This document
```

---

## Memory Management

### CPU Side

- **`Tensor`** owns data via `std::vector<float>`, providing automatic memory management.
- Tensors are passed by `const&` to layer functions; each layer allocates and returns a new output tensor.
- No dynamic allocation inside hot loops.

### GPU Side

Two-tier memory model:

```
┌──────────────────┐           ┌──────────────────┐
│   Host (CPU)     │           │   Device (GPU)   │
│                  │  upload   │                  │
│  Tensor          │ ────────► │  GpuTensor       │
│  (std::vector)   │           │  (float* d_ptr)  │
│                  │ ◄──────── │                  │
│                  │ download  │                  │
└──────────────────┘           └──────────────────┘
```

- **`GpuTensor`** is a RAII wrapper around `cudaMalloc`/`cudaFree`.
  - Move-only semantics (no copies) — prevents accidental double-free.
  - `upload(const Tensor&)` — `cudaMemcpy` host → device.
  - `download() → Tensor` — `cudaMemcpy` device → host.
- **Minimize transfers:** During GPU inference, data stays on-device between layers. Only the input upload and output download cross the PCIe bus.
- **All CUDA API calls** are wrapped with `CUDA_CHECK()` for error detection.

### Memory Access Patterns

| Pattern | Description | Used In |
|---------|-------------|---------|
| **Coalesced** | Consecutive threads access consecutive addresses | All optimized kernels |
| **Shared Memory** | Block-local scratchpad for data reuse | Tiled Conv2D, Softmax reduction |
| **Global** | Direct DRAM access (high latency) | Naive kernel fallback |

---

## Execution Pipelines

### CPU Pipeline (`./cnn_cpu`)

```
Load MNIST Image → Normalize → Conv2D → ReLU → MaxPool
    → Conv2D → ReLU → MaxPool → Flatten → Dense → ReLU
    → Dense → Softmax → argmax → Print Prediction
```

All operations execute sequentially on the CPU. This serves as the **correctness reference** and **performance baseline**.

### GPU Pipeline (`./cnn_gpu`)

```
Load MNIST Image → Normalize
    → cudaMemcpy (H→D)                    ← One-time upload
    → conv2d_kernel → relu_kernel          ← All on GPU
    → maxpool_kernel → conv2d_kernel       ← No host transfers
    → relu_kernel → maxpool_kernel         ← between layers
    → dense_kernel → relu_kernel
    → dense_kernel → softmax_kernel
    → cudaMemcpy (D→H)                    ← One-time download
    → argmax → Print Prediction
```

Weights are uploaded once at initialization. Only the input tensor and final output cross the PCIe bus per inference call.

### Optimized GPU Pipeline (`GpuModel`)

```
GpuModel gpu_model(model);             ← Weights uploaded once (~380 KB)

for each input:
    upload_async(input, stream)         ← Only input crosses PCIe
    → conv2d_tiled_kernel (shared mem)  ← ~9× fewer global reads
    → relu_kernel → maxpool_kernel
    → conv2d_tiled_kernel
    → relu_kernel → maxpool_kernel
    → reshape (metadata only)           ← Zero-cost flatten
    → dense_kernel → relu_kernel
    → dense_kernel → softmax_kernel
    download_async(output, stream)      ← Only output crosses PCIe
```

Optimizations applied:
- **Shared memory tiling** — input tile + halo loaded cooperatively, reused ~9× per pixel.
- **Weight caching** — model weights uploaded to GPU once in `GpuModel` constructor.
- **In-place reshape** — `GpuTensor::reshape()` changes metadata only, no data movement.
- **CUDA streams** — `CudaStream` RAII wrapper enables async transfers and pipelining.
- **16×16 thread blocks** — 2D spatial tiles for output, `blockIdx.z` encodes batch × channels.

---

## Design Principles

### 1. No External ML Libraries
Every kernel is hand-written — no cuDNN, cuBLAS, or ONNX Runtime. The purpose is demonstrating low-level GPU programming, not wrapping existing libraries.

### 2. CPU-First Development
Every layer is implemented and tested in pure C++ before any CUDA code is written. The CPU implementation serves as the ground truth for validating GPU kernels.

### 3. Correctness Before Performance
Naive (correct) CUDA kernels are written first. Optimizations are applied incrementally, with GPU outputs validated against CPU outputs at every step using `Tensor::allclose(atol=1e-5)`.

### 4. NCHW Data Layout
All tensors use batch-channels-height-width ordering. This matches cuDNN conventions and provides natural coalesced access for convolution kernels where threads iterate over the width dimension.

### 5. Explicit Memory Management
No hidden allocations. `GpuTensor` uses RAII for device memory. All `cudaMalloc`/`cudaFree` calls are traceable. Transfer times are measured and reported.

### 6. FP32 Only (MVP)
Single-precision floating point throughout. FP16/Tensor Core support is a stretch goal.

---

## Key Abstractions

### `Tensor` (CPU)
```cpp
class Tensor {
    std::vector<float> data_;    // Contiguous NCHW storage
    std::vector<int> shape_;     // e.g., {1, 8, 28, 28}

public:
    float& at(int n, int c, int h, int w);           // 4D NCHW indexing
    float& at(int n, int j);                          // 2D indexing
    static Tensor zeros(std::vector<int> shape);
    static Tensor rand(std::vector<int> shape, unsigned seed);
    bool allclose(const Tensor& other, float atol = 1e-5f);
    float max_diff(const Tensor& other) const;
    Tensor reshape(std::vector<int> new_shape);
    std::vector<int> argmax_per_row() const;
    int num_elements() const;
};
```

### `GpuTensor` (Device)
```cpp
class GpuTensor {
    float* d_data_ = nullptr;    // Device pointer (cudaMalloc)
    std::vector<int> shape_;

public:
    GpuTensor(std::vector<int> shape);       // Allocate device memory
    GpuTensor(const Tensor& cpu_tensor);     // Upload from CPU
    ~GpuTensor();                             // cudaFree
    GpuTensor(GpuTensor&&) noexcept;         // Move only
    void upload(const Tensor& t);             // Sync H→D
    Tensor download() const;                  // Sync D→H
    void upload_async(const Tensor& t, cudaStream_t s);   // Async
    Tensor download_async(cudaStream_t s) const;           // Async
    void reshape(const std::vector<int>& new_shape);       // Metadata only
    float* data();
};
```

### `GpuModel` (Cached Weights)
```cpp
class GpuModel {
public:
    GpuModel(const Model& model);             // Upload weights once
    Tensor inference(const Tensor& input) const;  // Uses cached weights
    std::vector<int> predict(const Tensor& input) const;

    // 8 GpuTensor weight members (cached on device)
    GpuTensor conv1_weights, conv1_bias;
    GpuTensor conv2_weights, conv2_bias;
    GpuTensor dense1_weights, dense1_bias;
    GpuTensor dense2_weights, dense2_bias;
};
```

### `Model` (CPU Weights)
```cpp
struct Model {
    Tensor conv1_weights;   // {8, 1, 3, 3}
    Tensor conv1_bias;      // {8}
    Tensor conv2_weights;   // {16, 8, 3, 3}
    Tensor conv2_bias;      // {16}
    Tensor dense1_weights;  // {120, 784}
    Tensor dense1_bias;     // {120}
    Tensor dense2_weights;  // {10, 120}
    Tensor dense2_bias;     // {10}

    void load(const std::string& path);
    void save(const std::string& path) const;
    void initialize_xavier(unsigned seed);
};
```

---

## CUDA Kernel Strategies

| Layer | Naive | Optimized | Key Technique |
|-------|-------|-----------|---------------|
| **Conv2D** | 1 thread/output, global reads | 16×16 tiles, shared memory | Cooperative loading, ~9× read reduction |
| **ReLU** | 1 thread/element | — (already optimal) | `fmaxf(x, 0)` |
| **MaxPool2D** | 1 thread/output | — | Window scan |
| **Dense** | 1 thread/output | — | Dot product over in_features |
| **Softmax** | — | 1 block/row, shared memory | Three-pass: max → exp+sum → normalize |
