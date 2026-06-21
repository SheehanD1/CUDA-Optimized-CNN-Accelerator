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
│   ├── gpu_tensor.h            # RAII GPU memory wrapper
│   ├── cuda_utils.h            # CUDA_CHECK macro, error handling
│   ├── model.h                 # Model struct (weights + architecture)
│   └── layers/                 # Layer function declarations
│       ├── conv2d.h
│       ├── relu.h
│       ├── maxpool.h
│       ├── dense.h
│       └── softmax.h
│
├── src/
│   ├── cpu/                    # CPU baseline (pure C++17)
│   │   ├── tensor.cpp          # Tensor implementation
│   │   ├── conv2d_cpu.cpp      # Naive 6-loop convolution
│   │   ├── relu_cpu.cpp        # Element-wise ReLU
│   │   ├── maxpool_cpu.cpp     # Sliding window max
│   │   ├── dense_cpu.cpp       # Matrix-vector multiply
│   │   ├── softmax_cpu.cpp     # Stable softmax
│   │   ├── inference_cpu.cpp   # Full CPU inference chain
│   │   ├── data_loader.cpp     # MNIST IDX format parser
│   │   └── main_cpu.cpp        # CPU executable entry point
│   │
│   ├── cuda/                   # GPU implementation (CUDA)
│   │   ├── memory.cu           # cudaMalloc/cudaMemcpy/cudaFree wrappers
│   │   ├── device_info.cu      # GPU capability query
│   │   ├── conv2d_kernel.cu    # Naive conv2d kernel
│   │   ├── conv2d_tiled_kernel.cu  # Shared-memory tiled conv2d
│   │   ├── relu_kernel.cu      # ReLU kernel
│   │   ├── maxpool_kernel.cu   # MaxPool kernel
│   │   ├── dense_kernel.cu     # Dense kernel
│   │   ├── softmax_kernel.cu   # Softmax kernel (parallel reduction)
│   │   ├── inference_gpu.cu    # Full GPU inference chain
│   │   └── main_gpu.cu         # GPU executable entry point
│   │
│   ├── layers/                 # Shared layer interfaces (header-only)
│   │
│   └── benchmarks/             # Performance measurement
│       ├── benchmark.cpp       # Timer class + benchmark harness
│       ├── bench_cpu.cpp       # CPU inference timing
│       ├── bench_gpu_naive.cpp # Naive GPU timing
│       └── bench_gpu_opt.cpp   # Optimized GPU timing
│
├── tests/                      # Google Test unit tests
│   ├── test_tensor.cpp         # Tensor construction, indexing, reshape
│   ├── test_conv2d.cpp         # Conv2D correctness (CPU + GPU vs CPU)
│   ├── test_relu.cpp           # ReLU correctness
│   ├── test_maxpool.cpp        # MaxPool correctness
│   ├── test_dense.cpp          # Dense correctness
│   ├── test_softmax.cpp        # Softmax correctness + numerical stability
│   └── test_gpu_memory.cpp     # GPU memory round-trip, RAII
│
├── scripts/
│   └── export_weights.py       # PyTorch: train model, export binary weights
│
├── weights/                    # Exported binary weight files
│
├── profiling/                  # Nsight profiling configs + results
│
└── docs/
    ├── architecture.md         # This document
    ├── benchmarks.md           # Performance results and analysis
    ├── optimizations.md        # Optimization techniques documentation
    └── profiling.md            # Nsight Compute/Systems guide
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

### Optimized GPU Pipeline

Same structure as GPU pipeline, but kernels use:
- **Shared memory tiling** — reduces global memory reads by loading tiles into `__shared__` memory.
- **Coalesced access patterns** — consecutive threads access consecutive memory addresses.
- **Tuned thread blocks** — block dimensions chosen via occupancy analysis (8×8 / 16×16 / 32×32 experiments + `cudaOccupancyMaxPotentialBlockSize()`).

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
    float& at(int n, int c, int h, int w);    // NCHW indexing
    static Tensor zeros(std::vector<int> shape);
    static Tensor rand(std::vector<int> shape);
    bool allclose(const Tensor& other, float atol = 1e-5f);
    Tensor reshape(std::vector<int> new_shape);
    int num_elements() const;
};
```

### `GpuTensor` (Device)
```cpp
class GpuTensor {
    float* d_ptr_ = nullptr;     // Device pointer (cudaMalloc)
    std::vector<int> shape_;

public:
    GpuTensor(std::vector<int> shape);   // Allocates device memory
    ~GpuTensor();                         // cudaFree
    GpuTensor(GpuTensor&&);              // Move only
    void upload(const Tensor& host_tensor);
    Tensor download() const;
    float* data();                        // Raw pointer for kernels
};
```

### `Model`
```cpp
struct Model {
    // Conv2D #1: 1 → 8 channels, 3×3
    Tensor conv1_weights;   // shape: {8, 1, 3, 3}
    Tensor conv1_bias;      // shape: {8}

    // Conv2D #2: 8 → 16 channels, 3×3
    Tensor conv2_weights;   // shape: {16, 8, 3, 3}
    Tensor conv2_bias;      // shape: {16}

    // Dense #1: 784 → 120
    Tensor fc1_weights;     // shape: {120, 784}
    Tensor fc1_bias;        // shape: {120}

    // Dense #2: 120 → 10
    Tensor fc2_weights;     // shape: {10, 120}
    Tensor fc2_bias;        // shape: {10}

    void load_weights(const std::string& dir);
    void initialize_random(unsigned seed);
};
```
