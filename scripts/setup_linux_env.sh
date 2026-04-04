#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Bootstrap Vauban development on Linux x86_64 for CPU-only MLX workflows.
# This builds libmlx.so from the matching MLX source tag and installs it
# system-wide so `import mlx.core` works on machines without Apple Silicon.

echo "==> Installing system build dependencies"
apt-get update -qq
apt-get install -y -qq libblas-dev liblapack-dev liblapacke-dev >/dev/null 2>&1

echo "==> Running uv sync (Python 3.13)"
uv sync --python 3.13

mlx_version="$(uv pip show mlx 2>/dev/null | awk '/^Version:/ {print $2}')"
if [[ -z "${mlx_version}" ]]; then
    echo "Unable to determine installed mlx version" >&2
    exit 1
fi
echo "==> Installed MLX wheel version: ${mlx_version}"

build_dir="$(mktemp -d)"
trap 'rm -rf "${build_dir}"' EXIT

echo "==> Cloning mlx v${mlx_version} into ${build_dir}"
git clone --depth 1 --branch "v${mlx_version}" \
    https://github.com/ml-explore/mlx.git "${build_dir}/mlx_src" >/dev/null 2>&1

echo "==> Building libmlx.so"
cmake -S "${build_dir}/mlx_src" -B "${build_dir}/build" \
    -DMLX_BUILD_PYTHON_BINDINGS=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DMLX_BUILD_TESTS=OFF \
    -DMLX_BUILD_BENCHMARKS=OFF \
    -DMLX_BUILD_EXAMPLES=OFF \
    -DLAPACK_INCLUDE_DIRS=/usr/include \
    -DCMAKE_CXX_FLAGS="-I/usr/include/x86_64-linux-gnu" >/dev/null 2>&1

make -C "${build_dir}/build" -j"$(nproc)" mlx >/dev/null 2>&1

echo "==> Installing libmlx.so to /usr/lib"
cp "${build_dir}/build/libmlx.so" /usr/lib/
ldconfig

echo "==> Verifying MLX import"
uv run python -c "import mlx.core as mx; print(f'mlx {mx.__version__} OK - {mx.array([1, 2, 3])}')"

echo "==> Done. Run tests with: uv run pytest tests/"
