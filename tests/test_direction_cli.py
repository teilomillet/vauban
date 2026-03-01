"""Tests for direction measurement helpers and CLI backend switch."""

import os
from pathlib import Path

import pytest

from tests.conftest import D_MODEL, MockTokenizer
from vauban import _ops as ops
from vauban.measure._direction import (
    _best_direction,
    _cosine_separation,
    _match_suffix,
    find_instruction_boundary,
)

# ── _match_suffix ────────────────────────────────────────────────────


class TestMatchSuffix:
    def test_no_match(self) -> None:
        assert _match_suffix([1, 2, 3], [4, 5, 6]) == 0

    def test_full_match(self) -> None:
        assert _match_suffix([1, 2, 3], [1, 2, 3]) == 3

    def test_partial_match(self) -> None:
        assert _match_suffix([1, 2, 3, 4], [9, 3, 4]) == 2

    def test_empty_lists(self) -> None:
        assert _match_suffix([], []) == 0

    def test_one_empty(self) -> None:
        assert _match_suffix([1, 2], []) == 0
        assert _match_suffix([], [1, 2]) == 0

    def test_single_match(self) -> None:
        assert _match_suffix([1, 2, 3], [9, 3]) == 1


# ── find_instruction_boundary ────────────────────────────────────────


class TestFindInstructionBoundary:
    def test_returns_valid_index(self) -> None:
        tokenizer = MockTokenizer(32)
        idx = find_instruction_boundary(tokenizer, "Hello world")
        # Should return an index within the full tokenized sequence
        full = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello world"}], tokenize=True,
        )
        assert isinstance(full, list)
        assert 0 <= idx < len(full)

    def test_different_prompts_different_boundaries(self) -> None:
        tokenizer = MockTokenizer(32)
        idx_short = find_instruction_boundary(tokenizer, "Hi")
        idx_long = find_instruction_boundary(tokenizer, "Hello world foo bar")
        # Longer prompt should have a later boundary
        assert idx_long > idx_short


# ── _cosine_separation ───────────────────────────────────────────────


class TestCosineSeparation:
    def test_orthogonal_vectors(self) -> None:
        harmful = ops.array([1.0, 0.0, 0.0, 0.0])
        harmless = ops.array([0.0, 1.0, 0.0, 0.0])
        direction = ops.array([1.0, 0.0, 0.0, 0.0])
        sep = _cosine_separation(harmful, harmless, direction)
        ops.eval(sep)
        assert float(sep.item()) == pytest.approx(1.0)

    def test_same_vectors_zero_separation(self) -> None:
        v = ops.array([1.0, 2.0, 3.0, 4.0])
        direction = ops.array([1.0, 0.0, 0.0, 0.0])
        sep = _cosine_separation(v, v, direction)
        ops.eval(sep)
        assert float(sep.item()) == pytest.approx(0.0)

    def test_opposite_direction(self) -> None:
        harmful = ops.array([1.0, 0.0])
        harmless = ops.array([0.0, 0.0])
        direction = ops.array([-1.0, 0.0])
        sep = _cosine_separation(harmful, harmless, direction)
        ops.eval(sep)
        # harmful proj = -1, harmless proj = 0 → sep = -1
        assert float(sep.item()) == pytest.approx(-1.0)


# ── _best_direction ──────────────────────────────────────────────────


class TestBestDirection:
    def test_selects_best_layer(self) -> None:
        # Layer 0: small difference, Layer 1: large difference
        harmful_acts = [
            ops.array([1.0, 0.0, 0.0, 0.0] * (D_MODEL // 4)),
            ops.array([5.0, 0.0, 0.0, 0.0] * (D_MODEL // 4)),
        ]
        harmless_acts = [
            ops.array([0.9, 0.0, 0.0, 0.0] * (D_MODEL // 4)),
            ops.array([0.0, 0.0, 0.0, 0.0] * (D_MODEL // 4)),
        ]
        for a in harmful_acts + harmless_acts:
            ops.eval(a)

        direction, best_layer, scores = _best_direction(
            harmful_acts, harmless_acts,
        )
        ops.eval(direction)

        assert len(scores) == 2
        assert best_layer == 1  # Layer 1 has larger separation
        assert direction.shape == (D_MODEL,)

    def test_unit_direction(self) -> None:
        harmful_acts = [ops.array([3.0] * D_MODEL)]
        harmless_acts = [ops.array([0.0] * D_MODEL)]
        ops.eval(harmful_acts[0], harmless_acts[0])

        direction, _, _ = _best_direction(harmful_acts, harmless_acts)
        ops.eval(direction)

        norm = float(ops.linalg.norm(direction).item())
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_returns_per_layer_scores(self) -> None:
        n_layers = 4
        harmful = [ops.random.normal((D_MODEL,)) for _ in range(n_layers)]
        harmless = [ops.random.normal((D_MODEL,)) for _ in range(n_layers)]
        for a in harmful + harmless:
            ops.eval(a)

        _, _, scores = _best_direction(harmful, harmless)
        assert len(scores) == n_layers
        for s in scores:
            assert isinstance(s, float)


# ── _set_backend_from_config ─────────────────────────────────────────


class TestSetBackendFromConfig:
    def test_sets_env_var(self, tmp_path: Path) -> None:
        from vauban.__main__ import _set_backend_from_config

        config = tmp_path / "run.toml"
        config.write_text('backend = "torch"\n[model]\npath = "test"\n')

        # Clear any existing value
        os.environ.pop("VAUBAN_BACKEND", None)
        _set_backend_from_config(str(config))
        assert os.environ.get("VAUBAN_BACKEND") == "torch"
        # Cleanup
        os.environ.pop("VAUBAN_BACKEND", None)

    def test_defaults_to_mlx(self, tmp_path: Path) -> None:
        from vauban.__main__ import _set_backend_from_config

        config = tmp_path / "run.toml"
        config.write_text('[model]\npath = "test"\n')

        os.environ.pop("VAUBAN_BACKEND", None)
        _set_backend_from_config(str(config))
        assert os.environ.get("VAUBAN_BACKEND") == "mlx"
        os.environ.pop("VAUBAN_BACKEND", None)

    def test_invalid_file_silently_fails(self) -> None:
        from vauban.__main__ import _set_backend_from_config

        os.environ.pop("VAUBAN_BACKEND", None)
        # Should not raise
        _set_backend_from_config("/nonexistent/file.toml")


# ── _command_suggestion ──────────────────────────────────────────────


class TestCommandSuggestion:
    def test_close_match(self) -> None:
        from vauban.__main__ import _command_suggestion

        assert _command_suggestion("mna") == "man"
        assert _command_suggestion("inti") == "init"
        assert _command_suggestion("dif") == "diff"

    def test_no_match(self) -> None:
        from vauban.__main__ import _command_suggestion

        assert _command_suggestion("zzzzzzz") is None

    def test_validate_alias(self) -> None:
        from vauban.__main__ import _command_suggestion

        assert _command_suggestion("validat") == "--validate"
