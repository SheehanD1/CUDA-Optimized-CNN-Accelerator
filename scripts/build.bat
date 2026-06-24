@echo off
REM ============================================================================
REM CUDA-Optimized CNN Accelerator — Windows Build Script
REM ============================================================================
REM Usage:
REM   scripts\build.bat              Build Release (default)
REM   scripts\build.bat debug        Build Debug
REM   scripts\build.bat release      Build Release
REM   scripts\build.bat clean        Remove build directories
REM   scripts\build.bat test         Build Debug + run tests
REM ============================================================================

setlocal enabledelayedexpansion

REM Navigate to project root (one level up from scripts/)
pushd "%~dp0\.."
set "PROJECT_ROOT=%CD%"

REM Parse command line argument
set "BUILD_TYPE=%~1"
if "%BUILD_TYPE%"=="" set "BUILD_TYPE=release"

REM ============================================================================
REM Clean
REM ============================================================================
if /I "%BUILD_TYPE%"=="clean" (
    echo [CNN] Cleaning build directories...
    if exist "build" rmdir /s /q "build"
    echo [CNN] Clean complete.
    goto :end
)

REM ============================================================================
REM Validate prerequisites
REM ============================================================================
echo [CNN] Checking prerequisites...

where cmake >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [CNN] ERROR: CMake not found. Install CMake 3.18+ and add to PATH.
    exit /b 1
)

where nvcc >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [CNN] ERROR: CUDA toolkit not found. Install CUDA 12.0+ and add nvcc to PATH.
    exit /b 1
)

echo [CNN] CMake found:
cmake --version | findstr /I "version"

echo [CNN] CUDA found:
nvcc --version | findstr /I "release"

REM ============================================================================
REM Configure
REM ============================================================================
set "BUILD_DIR=build\%BUILD_TYPE%"

echo.
echo [CNN] Configuring %BUILD_TYPE% build...
echo [CNN] Build directory: %BUILD_DIR%

cmake --preset %BUILD_TYPE%
if %ERRORLEVEL% neq 0 (
    echo [CNN] ERROR: CMake configuration failed.
    exit /b 1
)

REM ============================================================================
REM Build
REM ============================================================================
echo.
echo [CNN] Building...

cmake --build "%BUILD_DIR%" --parallel
if %ERRORLEVEL% neq 0 (
    echo [CNN] ERROR: Build failed.
    exit /b 1
)

echo.
echo [CNN] ============================================
echo [CNN]   Build successful! (%BUILD_TYPE%)
echo [CNN] ============================================

REM ============================================================================
REM Test (if requested or if build type is "test")
REM ============================================================================
if /I "%~1"=="test" (
    echo.
    echo [CNN] Running tests...
    ctest --test-dir "%BUILD_DIR%" --output-on-failure
    if %ERRORLEVEL% neq 0 (
        echo [CNN] ERROR: Some tests failed.
        exit /b 1
    )
    echo.
    echo [CNN] ============================================
    echo [CNN]   All tests passed!
    echo [CNN] ============================================
)

:end
popd
endlocal
