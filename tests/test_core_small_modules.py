# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Coverage tests for small core modules."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import cast

import pytest

import vauban._backend as backend_module
import vauban._compose as compose_module
from vauban import _ops as ops


class FakeMlxArray:
    """Minimal fake MLX array type."""


class FakeTorchTensor:
    """Minimal fake torch tensor type."""


class FakeMlxCoreModule(ModuleType):
    """Typed fake ``mlx.core`` module."""

    array = FakeMlxArray


class FakeTorchModule(ModuleType):
    """Typed fake ``torch`` module."""

    Tensor = FakeTorchTensor


def _reload_array_module(
    monkeypatch: pytest.MonkeyPatch,
    backend_name: str,
) -> ModuleType:
    """Reload ``vauban._array`` under a specific fake backend."""
    monkeypatch.setattr(backend_module, "get_backend", lambda: backend_name)
    monkeypatch.setitem(sys.modules, "mlx", ModuleType("mlx"))
    monkeypatch.setitem(sys.modules, "mlx.core", FakeMlxCoreModule("mlx.core"))
    monkeypatch.setitem(sys.modules, "torch", FakeTorchModule("torch"))
    sys.modules.pop("vauban._array", None)
    return importlib.import_module("vauban._array")


class TestBackendModule:
    """Tests for backend selection."""

    def test_reload_backend_respects_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VAUBAN_BACKEND", "torch")
        reloaded = importlib.reload(backend_module)

        assert reloaded.get_backend() == "torch"

        monkeypatch.delenv("VAUBAN_BACKEND", raising=False)
        importlib.reload(backend_module)

    def test_set_backend_validates_and_respects_lock(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(backend_module, "_BACKEND", "mlx")
        monkeypatch.setattr(backend_module, "_LOCKED", False)

        backend_module.set_backend("torch")
        assert backend_module.get_backend() == "torch"

        with pytest.raises(ValueError, match="Unknown backend"):
            backend_module.set_backend(cast("backend_module.BackendName", "invalid"))

        monkeypatch.setattr(backend_module, "_LOCKED", True)
        with pytest.raises(RuntimeError, match="Cannot change backend"):
            backend_module.set_backend("mlx")

        monkeypatch.setattr(backend_module, "_LOCKED", False)
        monkeypatch.setattr(backend_module, "_BACKEND", "mlx")
        backend_module._lock()
        assert backend_module._LOCKED is True


class TestArrayModule:
    """Tests for import-time array alias resolution."""

    def test_array_module_uses_mlx_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        array_module = _reload_array_module(monkeypatch, "mlx")

        assert array_module.Array is FakeMlxArray

    def test_array_module_uses_torch_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        array_module = _reload_array_module(monkeypatch, "torch")

        assert array_module.Array is FakeTorchTensor

    def test_array_module_rejects_unknown_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(backend_module, "get_backend", lambda: "unknown")
        monkeypatch.setitem(sys.modules, "mlx", ModuleType("mlx"))
        monkeypatch.setitem(sys.modules, "mlx.core", FakeMlxCoreModule("mlx.core"))
        monkeypatch.setitem(sys.modules, "torch", FakeTorchModule("torch"))
        sys.modules.pop("vauban._array", None)

        with pytest.raises(ValueError, match="Unknown backend"):
            importlib.import_module("vauban._array")


class TestComposeModule:
    """Tests for subspace-bank composition helpers."""

    def test_load_bank_validates_structure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        valid_array = ops.array([[3.0, 4.0]])
        monkeypatch.setattr(
            compose_module.ops,
            "load",
            lambda path: {"safe": valid_array},
        )

        loaded = compose_module.load_bank("bank.safetensors")
        assert loaded == {"safe": valid_array}

        monkeypatch.setattr(compose_module.ops, "load", lambda path: ["not-a-dict"])
        with pytest.raises(TypeError, match="Expected dict"):
            compose_module.load_bank("bank.safetensors")

        monkeypatch.setattr(
            compose_module.ops,
            "load",
            lambda path: {"bad": [1.0, 2.0]},
        )
        with pytest.raises(TypeError, match="is not an Array"):
            compose_module.load_bank("bank.safetensors")

    def test_compose_direction_handles_success_and_errors(self) -> None:
        bank = {
            "alpha": ops.array([[3.0, 4.0], [1.0, 0.0]]),
            "beta": ops.array([1.0, 0.0]),
        }

        direction = compose_module.compose_direction(bank, {"alpha": 1.0, "beta": -1.0})

        assert pytest.approx(float(ops.linalg.norm(direction).item())) == 1.0

        with pytest.raises(KeyError, match="unknown bank entry"):
            compose_module.compose_direction(bank, {"missing": 1.0})
        with pytest.raises(ValueError, match="Composition is empty"):
            compose_module.compose_direction(bank, {})
        with pytest.raises(ValueError, match="near-zero norm"):
            compose_module.compose_direction(bank, {"beta": 0.0})

    def test_compose_subspaces_uses_algebra_helpers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bank = {
            "alpha": ops.array([[1.0, 0.0]]),
            "beta": ops.array([[0.0, 1.0]]),
        }
        expected = object()
        from_array_calls: list[tuple[str, object]] = []
        compose_calls: list[tuple[list[object], list[float], int | None]] = []

        def _fake_from_array(array: object, label: str) -> object:
            result = {"label": label, "array": array}
            from_array_calls.append((label, array))
            return result

        def _fake_compose(
            spaces: list[object],
            weights: list[float],
            *,
            max_rank: int | None = None,
        ) -> object:
            compose_calls.append((spaces, weights, max_rank))
            return expected

        monkeypatch.setattr("vauban.algebra.from_array", _fake_from_array)
        monkeypatch.setattr("vauban.algebra.compose", _fake_compose)

        result = compose_module.compose_subspaces(
            bank,
            {"alpha": 0.5, "beta": -1.0},
            max_rank=3,
        )

        assert result is expected
        assert from_array_calls == [
            ("alpha", bank["alpha"]),
            ("beta", bank["beta"]),
        ]
        assert compose_calls == [
            (
                [
                    {"label": "alpha", "array": bank["alpha"]},
                    {"label": "beta", "array": bank["beta"]},
                ],
                [0.5, -1.0],
                3,
            ),
        ]

        with pytest.raises(ValueError, match="Composition is empty"):
            compose_module.compose_subspaces(bank, {})
        with pytest.raises(KeyError, match="unknown bank entry"):
            compose_module.compose_subspaces(bank, {"missing": 1.0})
