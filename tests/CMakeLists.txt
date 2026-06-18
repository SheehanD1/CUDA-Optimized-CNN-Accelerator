# ============================================================================
# Unit Tests (Google Test)
# ============================================================================
# All unit tests for the CNN accelerator project. Each layer has
# corresponding test files that validate CPU implementation correctness
# and GPU vs CPU output equivalence.
# ============================================================================

# Test targets will be added as layer implementations are completed.
# Each test links against GTest::gtest_main for automatic test discovery.

# Example (uncommented when test files are added):
#
# add_executable(test_tensor test_tensor.cpp)
# target_link_libraries(test_tensor PRIVATE
#     GTest::gtest_main
#     cnn_cpu_layers
# )
# gtest_discover_tests(test_tensor)
#
# add_executable(test_conv2d test_conv2d.cpp)
# target_link_libraries(test_conv2d PRIVATE
#     GTest::gtest_main
#     cnn_cpu_layers
# )
# gtest_discover_tests(test_conv2d)
