# ============================================================================
# Unit Tests (Google Test)
# ============================================================================
# All unit tests for the CNN accelerator project. Each layer has
# corresponding test files that validate CPU implementation correctness
# and GPU vs CPU output equivalence.
# ============================================================================

# --- Tensor tests ---
add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE
    GTest::gtest_main
    cnn_cpu_layers
)
gtest_discover_tests(test_tensor)

# --- Conv2D tests ---
add_executable(test_conv2d test_conv2d.cpp)
target_link_libraries(test_conv2d PRIVATE GTest::gtest_main cnn_cpu_layers)
gtest_discover_tests(test_conv2d)

# --- ReLU tests ---
add_executable(test_relu test_relu.cpp)
target_link_libraries(test_relu PRIVATE GTest::gtest_main cnn_cpu_layers)
gtest_discover_tests(test_relu)

# --- MaxPool2D tests ---
add_executable(test_maxpool test_maxpool.cpp)
target_link_libraries(test_maxpool PRIVATE GTest::gtest_main cnn_cpu_layers)
gtest_discover_tests(test_maxpool)

# --- Dense tests ---
add_executable(test_dense test_dense.cpp)
target_link_libraries(test_dense PRIVATE GTest::gtest_main cnn_cpu_layers)
gtest_discover_tests(test_dense)

# --- Softmax tests ---
add_executable(test_softmax test_softmax.cpp)
target_link_libraries(test_softmax PRIVATE GTest::gtest_main cnn_cpu_layers)
gtest_discover_tests(test_softmax)

# --- Model tests (construction, Xavier init, save/load, pipeline smoke) ---
add_executable(test_model test_model.cpp)
target_link_libraries(test_model PRIVATE GTest::gtest_main cnn_cpu_layers)
gtest_discover_tests(test_model)

# --- End-to-end pipeline integration tests ---
add_executable(test_pipeline test_pipeline.cpp)
target_link_libraries(test_pipeline PRIVATE GTest::gtest_main cnn_cpu_layers)
gtest_discover_tests(test_pipeline
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

# --- GPU tensor round-trip and RAII tests ---
add_executable(test_gpu_tensor test_gpu_tensor.cpp)
target_link_libraries(test_gpu_tensor PRIVATE GTest::gtest_main cnn_cpu_layers cnn_cuda_kernels)
gtest_discover_tests(test_gpu_tensor)

# --- GPU RAII and move semantics stress tests ---
add_executable(test_gpu_raii test_gpu_raii.cpp)
target_link_libraries(test_gpu_raii PRIVATE GTest::gtest_main cnn_cpu_layers cnn_cuda_kernels)
gtest_discover_tests(test_gpu_raii)

# --- Conv2D GPU vs CPU validation tests ---
add_executable(test_conv2d_gpu test_conv2d_gpu.cpp)
target_link_libraries(test_conv2d_gpu PRIVATE GTest::gtest_main cnn_cpu_layers cnn_cuda_kernels)
gtest_discover_tests(test_conv2d_gpu)

# --- ReLU GPU vs CPU validation tests ---
add_executable(test_relu_gpu test_relu_gpu.cpp)
target_link_libraries(test_relu_gpu PRIVATE GTest::gtest_main cnn_cpu_layers cnn_cuda_kernels)
gtest_discover_tests(test_relu_gpu)

# --- MaxPool2D GPU vs CPU validation tests ---
add_executable(test_maxpool_gpu test_maxpool_gpu.cpp)
target_link_libraries(test_maxpool_gpu PRIVATE GTest::gtest_main cnn_cpu_layers cnn_cuda_kernels)
gtest_discover_tests(test_maxpool_gpu)

# --- Dense GPU vs CPU validation tests ---
add_executable(test_dense_gpu test_dense_gpu.cpp)
target_link_libraries(test_dense_gpu PRIVATE GTest::gtest_main cnn_cpu_layers cnn_cuda_kernels)
gtest_discover_tests(test_dense_gpu)

# --- Softmax GPU vs CPU validation tests ---
add_executable(test_softmax_gpu test_softmax_gpu.cpp)
target_link_libraries(test_softmax_gpu PRIVATE GTest::gtest_main cnn_cpu_layers cnn_cuda_kernels)
gtest_discover_tests(test_softmax_gpu)

# --- GPU vs CPU end-to-end pipeline validation tests ---
add_executable(test_gpu_pipeline test_gpu_pipeline.cpp)
target_link_libraries(test_gpu_pipeline PRIVATE GTest::gtest_main cnn_cpu_layers cnn_cuda_kernels)
gtest_discover_tests(test_gpu_pipeline)

# --- CPU vs GPU timing benchmark ---
add_executable(test_timing test_timing.cpp)
target_link_libraries(test_timing PRIVATE GTest::gtest_main cnn_cpu_layers cnn_cuda_kernels)
gtest_discover_tests(test_timing)
