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

# add_executable(test_relu test_relu.cpp)
# target_link_libraries(test_relu PRIVATE GTest::gtest_main cnn_cpu_layers)
# gtest_discover_tests(test_relu)
#
# add_executable(test_maxpool test_maxpool.cpp)
# target_link_libraries(test_maxpool PRIVATE GTest::gtest_main cnn_cpu_layers)
# gtest_discover_tests(test_maxpool)
#
# add_executable(test_dense test_dense.cpp)
# target_link_libraries(test_dense PRIVATE GTest::gtest_main cnn_cpu_layers)
# gtest_discover_tests(test_dense)
#
# add_executable(test_softmax test_softmax.cpp)
# target_link_libraries(test_softmax PRIVATE GTest::gtest_main cnn_cpu_layers)
# gtest_discover_tests(test_softmax)
