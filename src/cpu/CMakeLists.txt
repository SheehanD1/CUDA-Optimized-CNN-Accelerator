# ============================================================================
# CPU Baseline Implementation
# ============================================================================
# Pure C++ implementations of all CNN layers for correctness verification
# and performance baseline measurement.
# ============================================================================

# Static library: CPU layer implementations
# Sources will be added as layers are implemented (Commits 15-28)
add_library(cnn_cpu_layers STATIC)

target_sources(cnn_cpu_layers
    PRIVATE
        # Tensor implementation (Commit 7)
        # conv2d_cpu.cpp     (Commit 15)
        # relu_cpu.cpp       (Commit 19)
        # maxpool_cpu.cpp    (Commit 21)
        # dense_cpu.cpp      (Commit 23)
        # softmax_cpu.cpp    (Commit 26)
        # inference_cpu.cpp  (Commit 30)
        # data_loader.cpp    (Commit 32)
        ${CMAKE_CURRENT_SOURCE_DIR}/.gitkeep   # Placeholder until sources are added
)

target_include_directories(cnn_cpu_layers
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(cnn_cpu_layers PUBLIC cxx_std_17)

# Apply warning flags based on compiler
if(MSVC)
    target_compile_options(cnn_cpu_layers PRIVATE ${CNN_MSVC_WARNING_FLAGS})
else()
    target_compile_options(cnn_cpu_layers PRIVATE ${CNN_CXX_WARNING_FLAGS})
endif()

# CPU inference executable (Commit 33)
# add_executable(cnn_cpu main_cpu.cpp)
# target_link_libraries(cnn_cpu PRIVATE cnn_cpu_layers)
