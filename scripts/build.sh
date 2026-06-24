#!/usr/bin/env bash
# ============================================================================
# CUDA-Optimized CNN Accelerator — Linux/macOS Build Script
# ============================================================================
# Usage:
#   ./scripts/build.sh              Build Release (default)
#   ./scripts/build.sh debug        Build Debug
#   ./scripts/build.sh release      Build Release
#   ./scripts/build.sh clean        Remove build directories
#   ./scripts/build.sh test         Build Debug + run tests
# ============================================================================

set -euo pipefail

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[CNN]${NC} $*"; }
ok()    { echo -e "${GREEN}[CNN]${NC} $*"; }
warn()  { echo -e "${YELLOW}[CNN]${NC} $*"; }
err()   { echo -e "${RED}[CNN] ERROR:${NC} $*" >&2; }

# Parse argument
BUILD_TYPE="${1:-release}"

# ============================================================================
# Clean
# ============================================================================
if [[ "$BUILD_TYPE" == "clean" ]]; then
    info "Cleaning build directories..."
    rm -rf build/
    ok "Clean complete."
    exit 0
fi

# ============================================================================
# Validate prerequisites
# ============================================================================
info "Checking prerequisites..."

if ! command -v cmake &>/dev/null; then
    err "CMake not found. Install CMake 3.18+ and add to PATH."
    exit 1
fi

if ! command -v nvcc &>/dev/null; then
    err "CUDA toolkit not found. Install CUDA 12.0+ and add nvcc to PATH."
    exit 1
fi

info "CMake: $(cmake --version | head -1)"
info "CUDA:  $(nvcc --version | grep release)"
info "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# ============================================================================
# Configure
# ============================================================================
BUILD_DIR="build/${BUILD_TYPE}"

echo ""
info "Configuring ${BUILD_TYPE} build..."
info "Build directory: ${BUILD_DIR}"

cmake --preset "${BUILD_TYPE}"

# ============================================================================
# Build
# ============================================================================
echo ""
info "Building with $(nproc) parallel jobs..."

cmake --build "${BUILD_DIR}" --parallel "$(nproc)"

echo ""
ok "============================================"
ok "  Build successful! (${BUILD_TYPE})"
ok "============================================"

# ============================================================================
# Test (if requested)
# ============================================================================
if [[ "$BUILD_TYPE" == "test" ]]; then
    echo ""
    info "Running tests..."
    ctest --test-dir "${BUILD_DIR}" --output-on-failure

    echo ""
    ok "============================================"
    ok "  All tests passed!"
    ok "============================================"
fi
