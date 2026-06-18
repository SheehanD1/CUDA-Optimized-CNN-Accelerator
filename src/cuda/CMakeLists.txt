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
        # CUDA infrastructure (Commits 35-41)
        # memory.cu            (Commit 36)
        # device_info.cu       (Commit 40)
        #
        # Naive kernels (Commits 42-52)
        # conv2d_kernel.cu     (Commit 42)
        # relu_kernel.cu       (Commit 45)
        # maxpool_kernel.cu    (Commit 47)
        # dense_kernel.cu      (Commit 49)
        # softmax_kernel.cu    (Commit 51)
        #
        # Optimized kernels (Commits 57-63)
        # conv2d_tiled_kernel.cu  (Commit 57)
        #
        # GPU pipeline (Commit 53)
        # inference_gpu.cu     (Commit 53)
        ${CMAKE_CURRENT_SOURCE_DIR}/.gitkeep   # Placeholder until sources are added
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

# GPU inference executable (Commit 54)
# add_executable(cnn_gpu main_gpu.cu)
# target_link_libraries(cnn_gpu PRIVATE cnn_cuda_kernels cnn_cpu_layers)
