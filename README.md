# CUDA-Optimized CNN Accelerator

A high-performance CNN inference engine built from scratch in **C++17** and **CUDA**, featuring hand-written CUDA kernels, shared-memory tiled convolution, GPU weight caching, and CUDA stream support for GPU-accelerated neural network inference.

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue?style=flat-square&logo=cplusplus" alt="C++17">
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900?style=flat-square&logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/CMake-3.18+-064F8C?style=flat-square&logo=cmake" alt="CMake">
  <img src="https://img.shields.io/badge/Test-Google%20Test-4285F4?style=flat-square&logo=google" alt="Google Test">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="MIT License">
</p>

---

## Overview

This project implements a complete CNN inference pipeline with **three execution backends** — a pure C++ CPU baseline, a naive CUDA GPU implementation, and an optimized GPU pipeline — with comprehensive benchmarking across all three.

Every CUDA kernel is **hand-written from scratch** — no cuDNN, cuBLAS, or ONNX Runtime. The goal is to demonstrate deep understanding of GPU architecture and parallel programming, not to wrap existing libraries.

### Network Architecture

A LeNet-style CNN for MNIST classification (~96,650 parameters):

```
Input (1×28×28)
  → Conv2D (1→8, 3×3, pad=1) → ReLU → MaxPool (2×2)
  → Conv2D (8→16, 3×3, pad=1) → ReLU → MaxPool (2×2)
  → Flatten (784) → Dense (784→120) → ReLU
  → Dense (120→10) → Softmax → Prediction (digit 0-9)
```

### Key Features

| Feature | Description |
|---|---|
| **5 Custom CUDA Kernels** | Hand-written Conv2D, ReLU, MaxPool2D, Dense, Softmax |
| **Shared-Memory Tiling** | 16×16 tiled convolution with cooperative input loading (~9× fewer global reads) |
| **GPU Weight Caching** | `GpuModel` class uploads weights once, reuses across inference calls |
| **In-Place Reshape** | Zero-cost flatten via `GpuTensor::reshape()` (metadata only, no data movement) |
| **CUDA Streams** | `CudaStream` RAII wrapper + async H↔D transfers |
| **CPU Baseline** | Complete C++ reference implementation for correctness validation |
| **100+ Tests** | Google Test suites covering every layer, kernel, pipeline, and optimization |
| **Benchmark Suite** | 3-way comparison: CPU vs Naive GPU vs Optimized GPU across batch sizes |

### GPU Pipeline

Data stays on the GPU throughout the entire inference pipeline. Only the input image and final probabilities cross the PCIe bus:

```
                    PCIe Bus
CPU ──── Input ──────────────► GPU
                                │
                     ┌──────────▼──────────┐
                     │  Conv2D (tiled)     │
                     │  ReLU               │
                     │  MaxPool            │  All on GPU
                     │  Conv2D (tiled)     │  (no host
                     │  ReLU               │   transfers)
                     │  MaxPool            │
                     │  Flatten (metadata) │
                     │  Dense → ReLU       │
                     │  Dense → Softmax    │
                     └──────────┬──────────┘
                                │
CPU ◄── Probabilities ──────────┘
                    PCIe Bus
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| **NVIDIA GPU** | Compute Capability ≥ 7.5 | Turing or newer recommended |
| **CUDA Toolkit** | 12.0+ | Includes `nvcc` and `cudart` |
| **CMake** | 3.18+ | Required for native CUDA language support |
| **C++ Compiler** | C++17 support | GCC ≥ 9, Clang ≥ 10, or MSVC ≥ 2019 |
| **Google Test** | v1.14 | Auto-fetched by CMake (no manual install) |

### Verify CUDA Installation

```bash
nvcc --version
nvidia-smi
```

---

## Building

### Quick Start

```bash
# Clone the repository
git clone https://github.com/SheehanD1/CUDA-Optimized-CNN-Accelerator.git
cd CUDA-Optimized-CNN-Accelerator

# Configure and build (Release mode)
cmake --preset release
cmake --build build/release

# Run tests
ctest --test-dir build/release --output-on-failure
```

### Windows

```powershell
scripts\build.bat
```

### Linux / macOS

```bash
chmod +x scripts/build.sh
./scripts/build.sh
```

### Manual CMake Configuration

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build . --parallel $(nproc)
```

---

## Usage

### Running Inference

```bash
# CPU inference
./build/release/cnn_cpu

# GPU inference (prints device info + comparison with CPU)
./build/release/cnn_gpu
```

### Running Tests

```bash
# All tests
ctest --test-dir build/release --output-on-failure

# Specific test suites
./build/release/test_tensor          # Tensor operations
./build/release/test_conv2d_tiled    # Tiled Conv2D correctness
./build/release/test_gpu_pipeline    # GPU vs CPU pipeline
./build/release/test_optimized_pipeline  # All optimizations
```

### Running Benchmarks

```bash
# Full 3-way benchmark (CPU vs Naive GPU vs Optimized GPU)
./build/release/test_final_benchmark

# Conv2D kernel comparison (naive vs tiled)
./build/release/test_conv2d_perf

# CPU vs GPU timing
./build/release/test_timing
```

Example benchmark output:

```
  ┌───────┬────────────┬────────────┬────────────┬──────────┬──────────┐
  │ Batch │  CPU (ms)  │ GPU Naive  │ GPU Optim  │ Naive ×  │ Optim ×  │
  ├───────┼────────────┼────────────┼────────────┼──────────┼──────────┤
  │     1 │     5.23   │     1.82   │     0.95   │   2.87x  │   5.51x  │
  │    16 │    83.45   │     8.12   │     3.41   │  10.28x  │  24.47x  │
  │    32 │   167.21   │    14.83   │     5.92   │  11.28x  │  28.25x  │
  └───────┴────────────┴────────────┴────────────┴──────────┴──────────┘
```

---

## Project Structure

```
CUDA-Optimized-CNN-Accelerator/
├── include/               # Headers
│   ├── tensor.h           # CPU Tensor (NCHW layout)
│   ├── model.h            # Model weights struct
│   ├── inference.h        # CPU + GPU inference APIs
│   ├── gpu_tensor.h       # RAII GPU memory + async transfers
│   ├── gpu_model.h        # Cached GPU weights
│   ├── kernels.h          # GPU kernel declarations
│   ├── cuda_utils.h       # CUDA_CHECK / KERNEL_CHECK macros
│   ├── cuda_timer.h       # cudaEvent-based profiling
│   ├── cuda_stream.h      # CudaStream RAII wrapper
│   └── layers/            # CPU layer declarations
│
├── src/cpu/               # CPU baseline (pure C++17)
│   ├── tensor.cpp         # Tensor implementation
│   ├── model.cpp          # Model load/save/Xavier init
│   ├── inference_cpu.cpp  # Full CPU inference pipeline
│   └── *_cpu.cpp          # Layer implementations
│
├── src/cuda/              # GPU implementation (CUDA)
│   ├── conv2d_kernel.cu          # Naive Conv2D (1 thread/output)
│   ├── conv2d_tiled_kernel.cu    # Tiled Conv2D (16×16, shared memory)
│   ├── relu_kernel.cu            # ReLU kernel
│   ├── maxpool_kernel.cu         # MaxPool kernel
│   ├── dense_kernel.cu           # Dense/FC kernel
│   ├── softmax_kernel.cu         # Softmax (shared memory reduction)
│   ├── gpu_tensor.cu             # GpuTensor RAII + transfers
│   ├── gpu_model.cu              # GpuModel cached inference
│   └── inference_gpu.cu          # Full GPU inference pipeline
│
├── tests/                 # Google Test suites (100+ tests)
│   ├── test_tensor.cpp             # Tensor operations
│   ├── test_conv2d.cpp             # Conv2D CPU correctness
│   ├── test_conv2d_gpu.cpp         # Conv2D GPU vs CPU
│   ├── test_conv2d_tiled.cpp       # Tiled vs naive vs CPU
│   ├── test_gpu_pipeline.cpp       # GPU pipeline vs CPU pipeline
│   ├── test_optimized_pipeline.cpp # GpuModel + streams + all opts
│   ├── test_conv2d_perf.cpp        # Naive vs tiled benchmark
│   ├── test_final_benchmark.cpp    # 3-way pipeline benchmark
│   └── ...                         # ReLU, MaxPool, Dense, Softmax, etc.
│
├── scripts/
│   ├── build.bat          # Windows build script
│   └── build.sh           # Linux/macOS build script
│
└── docs/
    └── architecture.md    # System design and data flow
```

For detailed architecture information, see [docs/architecture.md](docs/architecture.md).

---

## Optimization Techniques

| # | Technique | Location | Impact |
|---|---|---|---|
| 1 | **Shared Memory Tiling** | `conv2d_tiled_kernel.cu` | 16×16 tiles with cooperative input loading; ~9× fewer global memory reads for 3×3 kernels |
| 2 | **GPU Weight Caching** | `gpu_model.h/cu` | Upload ~380 KB weights once in constructor; eliminates per-call transfer overhead |
| 3 | **In-Place Reshape** | `GpuTensor::reshape()` | Zero-cost flatten: metadata swap only, no `cudaMemcpy` or allocation |
| 4 | **CUDA Streams** | `cuda_stream.h` | RAII stream wrapper; async `upload_async` / `download_async` for pipelined transfers |
| 5 | **Shared Memory Softmax** | `softmax_kernel.cu` | One block per row; three-pass tree reduction (max → exp+sum → normalize) |
| 6 | **Minimal H↔D Transfers** | `inference_gpu.cu` | Data stays on GPU through entire 11-kernel pipeline; only input/output cross PCIe |

---

## Milestones

- [x] **Phase 0** — Project scaffolding, build system, documentation
- [x] **Phase 1** — Tensor foundation (NCHW layout, indexing, factory methods)
- [x] **Phase 2** — CPU baseline (Conv2D, ReLU, MaxPool, Dense, Softmax, pipeline)
- [x] **Phase 3** — GPU memory management (GpuTensor RAII, transfer wrappers)
- [x] **Phase 4** — Model architecture (weights, serialization, Xavier init)
- [x] **Phase 5** — CUDA kernels (all 5 layer kernels + validation)
- [x] **Phase 6** — GPU pipeline (inference, validation, timing benchmarks)
- [x] **Phase 7** — Optimization (tiled conv, weight caching, streams, benchmarks)
- [ ] **Phase 8** — Documentation and final polish

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.