# CUDA-Optimized CNN Accelerator

A high-performance CNN inference engine built from scratch in **C++17** and **CUDA**, featuring custom convolution kernels, shared-memory tiling, and occupancy-aware thread scheduling for GPU-accelerated neural network inference.

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue?style=flat-square&logo=cplusplus" alt="C++17">
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900?style=flat-square&logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/CMake-3.18+-064F8C?style=flat-square&logo=cmake" alt="CMake">
  <img src="https://img.shields.io/badge/Test-Google%20Test-4285F4?style=flat-square&logo=google" alt="Google Test">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="MIT License">
</p>

---

## Overview

This project implements a complete CNN inference pipeline with **two execution backends** — a pure C++ CPU baseline and a CUDA GPU-accelerated version — then benchmarks and optimizes GPU kernel performance through shared-memory tiling, memory coalescing, and occupancy tuning.

Every CUDA kernel is **hand-written from scratch** — no cuDNN, cuBLAS, or ONNX Runtime. The goal is to demonstrate deep understanding of GPU architecture and parallel programming, not to wrap existing libraries.

### Network Architecture

A deeper LeNet-style CNN trained on MNIST (~96,650 parameters):

```
Input (1×28×28)
  → Conv2D (1→8, 3×3) → ReLU → MaxPool (2×2)
  → Conv2D (8→16, 3×3) → ReLU → MaxPool (2×2)
  → Flatten (784) → Dense (784→120) → ReLU
  → Dense (120→10) → Softmax → Prediction
```

### Key Features

- **Custom CUDA Kernels** — Hand-written kernels for Conv2D, ReLU, MaxPool, Dense, and Softmax
- **Shared-Memory Tiling** — Tiled convolution kernel reducing global memory latency
- **Memory Coalescing** — Optimized access patterns for maximum GPU memory bandwidth
- **Occupancy Tuning** — Systematic block size experimentation (8×8, 16×16, 32×32) with `cudaOccupancyMaxPotentialBlockSize()`
- **CPU Baseline** — Complete C++ reference implementation for correctness validation
- **Benchmark Suite** — Automated CPU vs. Naive GPU vs. Optimized GPU comparison
- **NVIDIA Nsight Integration** — NVTX annotations for profiling with Nsight Compute and Nsight Systems

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| **NVIDIA GPU** | Compute Capability ≥ 7.5 | Turing or newer recommended |
| **CUDA Toolkit** | 12.0+ | Includes `nvcc`, `cudart`, and profiling tools |
| **CMake** | 3.18+ | Required for native CUDA language support |
| **C++ Compiler** | C++17 support | GCC ≥ 9, Clang ≥ 10, or MSVC ≥ 2019 |
| **Python** | 3.8+ | For weight export script (requires PyTorch) |
| **Google Test** | v1.14 | Auto-fetched by CMake (no manual install) |

### Verify CUDA Installation

```bash
# Check CUDA compiler
nvcc --version

# Check available GPU
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

### Build Presets

The project includes CMake presets for common configurations:

```bash
# Debug build (with CUDA device-side debugging)
cmake --preset debug
cmake --build build/debug

# Release build (optimized, for benchmarking)
cmake --preset release
cmake --build build/release

# Release with debug info (for Nsight profiling)
cmake --preset relwithdebinfo
cmake --build build/relwithdebinfo
```

### Manual CMake Configuration

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build . --parallel $(nproc)
```

---

## Usage

### Export Model Weights (One-Time Setup)

Train the CNN in PyTorch and export weights for the C++ inference engine:

```bash
# Install PyTorch (if not already installed)
pip install torch torchvision

# Train model and export weights
python scripts/export_weights.py --output weights/
```

This produces binary weight files in the `weights/` directory that the C++ loaders can read.

### CPU Inference

```bash
./build/release/cnn_cpu --image data/mnist/test_image.idx --weights weights/
```

### GPU Inference

```bash
./build/release/cnn_gpu --image data/mnist/test_image.idx --weights weights/
```

### Benchmarks

```bash
# Run full benchmark suite (CPU vs. Naive GPU vs. Optimized GPU)
./build/release/benchmark
```

Example output:

```
╔══════════════════╦═══════════════╦═══════════════╦══════════╗
║ Version          ║ Inference (ms)║ Kernel (ms)   ║ Speedup  ║
╠══════════════════╬═══════════════╬═══════════════╬══════════╣
║ CPU Baseline     ║ X.XXX         ║ —             ║ 1.0x     ║
║ Naive CUDA       ║ X.XXX         ║ X.XXX         ║ X.Xx     ║
║ Optimized CUDA   ║ X.XXX         ║ X.XXX         ║ X.Xx     ║
╚══════════════════╩═══════════════╩═══════════════╩══════════╝
```

---

## Profiling

### Nsight Compute (Kernel Analysis)

```bash
ncu --set full ./build/relwithdebinfo/cnn_gpu
```

### Nsight Systems (Timeline)

```bash
nsys profile --trace=cuda,nvtx ./build/relwithdebinfo/cnn_gpu
nsys-ui report.nsys-rep
```

NVTX annotations provide clear visual separation of each layer in the Nsight timeline view.

---

## Project Structure

```
cnn-accelerator/
├── src/
│   ├── cpu/           # Pure C++ baseline implementation
│   ├── cuda/          # CUDA kernels and GPU inference pipeline
│   ├── layers/        # Shared layer interfaces (header-only)
│   └── benchmarks/    # Performance measurement harness
├── include/           # Public headers (Tensor, GpuTensor, Model)
├── tests/             # Google Test unit tests
├── scripts/           # Weight export and build scripts
├── weights/           # Exported model weights (binary)
├── profiling/         # Nsight profiling configs and results
└── docs/
    ├── architecture.md    # System design and data flow
    ├── benchmarks.md      # Performance results and analysis
    ├── optimizations.md   # Optimization techniques documentation
    └── profiling.md       # Nsight profiling guide
```

For detailed architecture information, see [docs/architecture.md](docs/architecture.md).

---

## Milestones

- [x] **Milestone 0** — Project scaffolding, build system, documentation
- [ ] **Milestone 1** — CPU baseline (Tensor, Conv2D, ReLU, MaxPool, Dense, Softmax)
- [ ] **Milestone 2** — CUDA convolution kernel (naive, correct)
- [ ] **Milestone 3** — GPU memory management (GpuTensor, transfer timing)
- [ ] **Milestone 4** — Performance optimization (tiling, coalescing, occupancy)
- [ ] **Milestone 5** — Benchmark suite and performance analysis

### Stretch Goals

- [ ] Kernel fusion (Conv2D + ReLU in single kernel)
- [ ] FP16 half-precision support
- [ ] CUDA streams for pipelined inference
- [ ] Tensor Core WMMA support

---

## Optimization Techniques

| Technique | Description | Expected Impact |
|---|---|---|
| **Shared Memory Tiling** | Load input/kernel tiles into `__shared__` before computation | Reduced global memory latency |
| **Memory Coalescing** | Consecutive threads access consecutive addresses | Higher effective memory bandwidth |
| **Occupancy Tuning** | Systematic block size experimentation | Better SM utilization |
| **Kernel Fusion** | Combine Conv2D + ReLU into single kernel | Eliminate intermediate global memory write |

For detailed documentation, see [docs/optimizations.md](docs/optimizations.md).

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.