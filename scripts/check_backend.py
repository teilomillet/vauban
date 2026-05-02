# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Report Vauban runtime backend and accelerator visibility."""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import subprocess
import sys
from typing import Protocol, cast


class _ScalarTensor(Protocol):
    def sum(self) -> _ScalarTensor:
        """Return the tensor sum."""

    def item(self) -> float:
        """Return the scalar tensor value."""


class _TorchCudaProperties(Protocol):
    total_memory: int


class _TorchCuda(Protocol):
    def is_available(self) -> bool:
        """Return whether CUDA is visible to PyTorch."""

    def device_count(self) -> int:
        """Return the number of visible CUDA devices."""

    def get_device_properties(self, index: int) -> _TorchCudaProperties:
        """Return CUDA device properties."""

    def get_device_name(self, index: int) -> str:
        """Return a CUDA device name."""

    def synchronize(self) -> None:
        """Synchronize the current CUDA device."""


class _TorchMps(Protocol):
    def is_available(self) -> bool:
        """Return whether MPS is visible to PyTorch."""

    def synchronize(self) -> None:
        """Synchronize the current MPS device."""


class _TorchVersion(Protocol):
    cuda: str | None


class _TorchModule(Protocol):
    __version__: str
    version: _TorchVersion
    cuda: _TorchCuda
    mps: _TorchMps

    def ones(self, shape: tuple[int, int], *, device: str) -> _ScalarTensor:
        """Create a tensor of ones."""

    def eye(self, rows: int, *, device: str) -> _ScalarTensor:
        """Create an identity tensor."""

    def matmul(self, lhs: _ScalarTensor, rhs: _ScalarTensor) -> _ScalarTensor:
        """Multiply two tensors."""


class _MlxArray(Protocol):
    def item(self) -> int:
        """Return the scalar array value."""


class _MlxModule(Protocol):
    __version__: str

    def array(self, values: list[int]) -> _MlxArray:
        """Create an MLX array."""

    def sum(self, values: _MlxArray) -> _MlxArray:
        """Sum an MLX array."""

    def eval(self, values: _MlxArray) -> None:
        """Evaluate an MLX array."""

    def default_device(self) -> object:
        """Return the default MLX device."""


def _find_package(name: str) -> bool:
    """Return whether a Python package can be imported."""
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def _run_command(cmd: list[str]) -> tuple[int, str]:
    """Run a command and return status plus trimmed combined output."""
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return 127, f"{cmd[0]} not found"
    output = (result.stdout + result.stderr).strip()
    return result.returncode, output


def _print_host() -> None:
    """Print stable host facts that affect backend resolution."""
    print(f"python: {sys.version.split()[0]} ({sys.executable})")
    print(f"platform: {platform.platform()}")
    print(f"VAUBAN_BACKEND: {os.environ.get('VAUBAN_BACKEND', '<unset>')}")


def _print_nvidia() -> None:
    """Print NVIDIA driver visibility from nvidia-smi when available."""
    status, output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
    )
    if status == 0:
        print(f"nvidia-smi: {output}")
        return
    print(f"nvidia-smi: unavailable ({output})")


def _check_torch(require_cuda: bool) -> int:
    """Check PyTorch import and accelerator visibility."""
    if not _find_package("torch"):
        print("torch: not installed")
        return 2

    try:
        torch = cast("_TorchModule", importlib.import_module("torch"))
    except ImportError as exc:
        print(f"torch: import failed ({exc})")
        return 2

    print(f"torch: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")

    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(torch.mps.is_available())
    print(f"torch.cuda.is_available: {cuda_available}")
    print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
    print(f"torch.mps.is_available: {mps_available}")
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            total_gib = props.total_memory / (1024**3)
            print(
                "torch.cuda.device"
                f"[{index}]: {torch.cuda.get_device_name(index)}"
                f" ({total_gib:.1f} GiB)",
            )
        lhs = torch.ones((2, 2), device="cuda")
        rhs = torch.eye(2, device="cuda")
        value = torch.matmul(lhs, rhs).sum().item()
        torch.cuda.synchronize()
        print(f"torch.cuda.smoke_sum: {value:.1f}")
    if mps_available:
        lhs = torch.ones((2, 2), device="mps")
        rhs = torch.eye(2, device="mps")
        value = torch.matmul(lhs, rhs).sum().item()
        torch.mps.synchronize()
        print(f"torch.mps.smoke_sum: {value:.1f}")

    if require_cuda and not cuda_available:
        print("error: CUDA was required but PyTorch cannot see a CUDA device")
        return 3
    return 0


def _check_mlx() -> int:
    """Check MLX import and default device."""
    if not _find_package("mlx.core"):
        print("mlx: not installed")
        return 2

    try:
        mx = cast("_MlxModule", importlib.import_module("mlx.core"))
    except ImportError as exc:
        print(f"mlx: import failed ({exc})")
        return 2

    print(f"mlx: {mx.__version__}")
    print(f"mlx.default_device: {mx.default_device()}")
    value = mx.sum(mx.array([1, 2, 3]))
    mx.eval(value)
    print(f"mlx.smoke_sum: {value.item()}")
    return 0


def main() -> int:
    """Run backend diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("auto", "mlx", "torch"),
        default="auto",
        help="Backend to check. 'auto' uses VAUBAN_BACKEND or torch.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail if the selected torch backend cannot see CUDA.",
    )
    args = parser.parse_args()

    backend = args.backend
    if backend == "auto":
        backend = os.environ.get("VAUBAN_BACKEND", "torch")

    _print_host()
    _print_nvidia()

    if backend == "torch":
        return _check_torch(require_cuda=args.require_cuda)
    if backend == "mlx":
        return _check_mlx()

    print(f"error: unsupported backend {backend!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
