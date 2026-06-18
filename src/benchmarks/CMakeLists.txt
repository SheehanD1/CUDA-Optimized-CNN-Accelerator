# ============================================================================
# Benchmark Suite
# ============================================================================
# Benchmark harness for measuring CPU vs naive GPU vs optimized GPU
# inference performance.
# ============================================================================

# Benchmark executable (Commits 66-70)
# add_executable(benchmark
#     benchmark.cpp
#     bench_cpu.cpp
#     bench_gpu_naive.cpp
#     bench_gpu_optimized.cpp
# )
# target_link_libraries(benchmark PRIVATE cnn_cpu_layers cnn_cuda_kernels)
