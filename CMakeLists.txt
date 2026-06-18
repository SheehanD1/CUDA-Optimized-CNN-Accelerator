cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# ============================================================================
# CUDA-Optimized CNN Accelerator
# ============================================================================
# A high-performance CNN inference engine built with C++17 and CUDA.
# Implements custom convolution kernels, shared-memory tiling, and
# occupancy-aware thread scheduling for GPU-accelerated neural network
# inference.
# ============================================================================

project(cnn-accelerator
    VERSION 0.1.0
    LANGUAGES CXX CUDA
    DESCRIPTION "GPU-accelerated CNN inference engine with custom CUDA kernels"
)

# ============================================================================
# Global Settings
# ============================================================================

# C++17 standard required for structured bindings, std::optional, etc.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA standard matching C++ standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Export compile commands for IDE/LSP support (clangd, etc.)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release build if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# ============================================================================
# CUDA Configuration
# ============================================================================

# Find CUDA Toolkit (provides cudart, nvtx, etc.)
find_package(CUDAToolkit REQUIRED)

# Print CUDA info for build diagnostics
message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Toolkit Root:    ${CUDAToolkit_TARGET_DIR}")
message(STATUS "CUDA Architectures:   ${CMAKE_CUDA_ARCHITECTURES}")

# If no CUDA architecture specified, auto-detect or use common defaults
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Support Turing (75), Ampere (80, 86), Ada Lovelace (89), Hopper (90)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
    message(STATUS "No CUDA architecture specified, using defaults: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# ============================================================================
# Compiler Flags
# ============================================================================

# Common warning flags for CXX
set(CNN_CXX_WARNING_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    -Wnon-virtual-dtor
)

# MSVC-compatible warning flags
set(CNN_MSVC_WARNING_FLAGS
    /W4
    /permissive-
)

# ============================================================================
# Include Directories
# ============================================================================

# Global include path for all targets
include_directories(${PROJECT_SOURCE_DIR}/include)

# ============================================================================
# Google Test (fetched at configure time)
# ============================================================================

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE
)

# Prevent GoogleTest from overriding compiler/linker options on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Don't install GTest when installing our project
option(INSTALL_GTEST "Enable installation of googletest." OFF)

FetchContent_MakeAvailable(googletest)

# Enable CTest integration
enable_testing()
include(GoogleTest)

# ============================================================================
# Subdirectories
# ============================================================================

# Source code targets
add_subdirectory(src)

# Test targets
add_subdirectory(tests)
