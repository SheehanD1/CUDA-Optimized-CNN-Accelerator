# ============================================================================
# CUDA Kernel Implementations
# ============================================================================
# Custom CUDA kernels for GPU-accelerated CNN inference, including naive
# implementations, shared-memory tiled variants, and optimized kernels.
# ============================================================================

# Static library: CUDA kernels and GPU pipeline
add_library(cnn_cuda_kernels STATIC)

target_sources(cnn_cuda_kernels
    PRIVATE
        # CUDA infrastructure
        memory.cu
        gpu_tensor.cu
        cuda_timer.cu
        device_info.cu
        #
        # Naive kernels
        conv2d_kernel.cu
        relu_kernel.cu
        maxpool_kernel.cu
        dense_kernel.cu
        softmax_kernel.cu
        #
        # Optimized kernels
        conv2d_tiled_kernel.cu
        #
        # GPU pipeline
        inference_gpu.cu
        gpu_model.cu
)

target_include_directories(cnn_cuda_kernels
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(cnn_cuda_kernels PUBLIC cxx_std_17)

# Link CUDA runtime
target_link_libraries(cnn_cuda_kernels
    PUBLIC
        CUDA::cudart
)

# CUDA-specific compiler flags
target_compile_options(cnn_cuda_kernels
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
            -Xcompiler=${CNN_CXX_WARNING_FLAGS}
        >
)

# GPU inference executable
add_executable(cnn_gpu main_gpu.cu)
target_link_libraries(cnn_gpu PRIVATE cnn_cuda_kernels cnn_cpu_layers)
