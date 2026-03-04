#!/usr/bin/env bash
# setup_linux_env.sh — bootstrap vauban development on Linux (x86_64).
#
# MLX ships a Linux wheel whose Python extension links against libmlx.so,
# but the wheel does *not* bundle that shared library.  This script builds
# libmlx.so from source and installs it system-wide so that `import mlx`
# succeeds on CPU-only Linux boxes (CI, cloud VMs, etc.).
#
# Prerequisites: cmake, a C++17 compiler, libblas-dev, liblapacke-dev
#
# Usage:
#   chmod +x scripts/setup_linux_env.sh
#   ./scripts/setup_linux_env.sh

set -euo pipefail

# ── 1. System dependencies ────────────────────────────────────────────
echo "==> Installing system build dependencies …"
apt-get update -qq
apt-get install -y -qq libblas-dev liblapack-dev liblapacke-dev >/dev/null 2>&1

# ── 2. Python venv + project deps via uv ──────────────────────────────
echo "==> Running uv sync (Python 3.13) …"
uv sync --python 3.13

# ── 3. Determine installed MLX version ────────────────────────────────
MLX_VERSION=$(uv pip show mlx 2>/dev/null | grep '^Version:' | awk '{print $2}')
echo "==> Installed MLX wheel version: ${MLX_VERSION}"

# ── 4. Build libmlx.so from matching source tag ──────────────────────
BUILD_DIR=$(mktemp -d)
echo "==> Cloning mlx v${MLX_VERSION} into ${BUILD_DIR} …"
git clone --depth 1 --branch "v${MLX_VERSION}" \
    https://github.com/ml-explore/mlx.git "${BUILD_DIR}/mlx_src" 2>/dev/null

echo "==> Building libmlx.so …"
cmake -S "${BUILD_DIR}/mlx_src" -B "${BUILD_DIR}/build" \
    -DMLX_BUILD_PYTHON_BINDINGS=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DMLX_BUILD_TESTS=OFF \
    -DMLX_BUILD_BENCHMARKS=OFF \
    -DMLX_BUILD_EXAMPLES=OFF \
    -DLAPACK_INCLUDE_DIRS=/usr/include \
    -DCMAKE_CXX_FLAGS="-I/usr/include/x86_64-linux-gnu" \
    >/dev/null 2>&1

make -C "${BUILD_DIR}/build" -j"$(nproc)" mlx >/dev/null 2>&1

# ── 5. Install libmlx.so system-wide ─────────────────────────────────
echo "==> Installing libmlx.so to /usr/lib …"
cp "${BUILD_DIR}/build/libmlx.so" /usr/lib/
ldconfig

# ── 6. Verify ────────────────────────────────────────────────────────
echo "==> Verifying MLX import …"
uv run python -c "import mlx.core as mx; print(f'  mlx {mx.__version__} OK — {mx.array([1,2,3])}')"

# ── 7. Clean up ──────────────────────────────────────────────────────
rm -rf "${BUILD_DIR}"

echo "==> Done.  Run tests with:  uv run pytest tests/"
