"""Tests for vauban._compose: subspace bank composition."""

from pathlib import Path

import mlx.core as mx
import pytest

from vauban._compose import compose_direction, load_bank


class TestLoadBank:
    def test_load_safetensors(self, tmp_path: Path) -> None:
        basis_a = mx.random.normal((3, 16))
        basis_b = mx.random.normal((3, 16))
        mx.eval(basis_a, basis_b)
        path = tmp_path / "bank.safetensors"
        mx.save_safetensors(str(path), {"safety": basis_a, "format": basis_b})

        bank = load_bank(path)
        assert set(bank.keys()) == {"safety", "format"}
        assert bank["safety"].shape == (3, 16)


class TestComposeDirection:
    def test_weighted_sum_normalized(self) -> None:
        bank = {
            "a": mx.array([[1.0, 0.0, 0.0, 0.0]]),
            "b": mx.array([[0.0, 1.0, 0.0, 0.0]]),
        }
        result = compose_direction(bank, {"a": 1.0, "b": 1.0})
        mx.eval(result)
        norm = float(mx.linalg.norm(result).item())
        assert abs(norm - 1.0) < 1e-5

    def test_single_entry(self) -> None:
        bank = {"x": mx.array([[3.0, 4.0]])}
        result = compose_direction(bank, {"x": 1.0})
        mx.eval(result)
        norm = float(mx.linalg.norm(result).item())
        assert abs(norm - 1.0) < 1e-5

    def test_missing_key_raises(self) -> None:
        bank = {"a": mx.array([[1.0, 0.0]])}
        with pytest.raises(KeyError, match="missing"):
            compose_direction(bank, {"missing": 1.0})

    def test_empty_composition_raises(self) -> None:
        bank = {"a": mx.array([[1.0, 0.0]])}
        with pytest.raises(ValueError, match="empty"):
            compose_direction(bank, {})

    def test_2d_basis_takes_first_row(self) -> None:
        bank = {
            "multi": mx.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]),
        }
        result = compose_direction(bank, {"multi": 1.0})
        mx.eval(result)
        # Should be [1, 0, 0] normalized
        assert abs(float(result[0].item()) - 1.0) < 1e-5
        assert abs(float(result[1].item())) < 1e-5
