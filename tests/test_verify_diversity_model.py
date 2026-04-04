# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``vauban.data.verify_diversity_model``."""

from __future__ import annotations

import json
import runpy
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import vauban._forward as forward_module
import vauban._model_io as model_io_module
import vauban._ops as ops_module
import vauban.data.verify_diversity_model as verify_diversity_model
import vauban.measure as measure_module

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(slots=True)
class FakeDirectionResult:
    """Minimal stand-in for ``measure()`` results."""

    direction: np.ndarray
    layer_index: int
    d_model: int
    cosine_scores: list[float]


class FakeTokenizer:
    """Tokenizer stub with deterministic chat-template behavior."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
    ) -> str:
        """Return the user content as plain text."""
        del tokenize
        return messages[0]["content"]

    def encode(self, text: str) -> list[int]:
        """Encode text into a tiny deterministic token sequence."""
        base = len(text)
        return [base, base + 1]


class FakeNonStringTokenizer:
    """Tokenizer variant that violates the expected return contract."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
    ) -> list[int]:
        """Return a non-string object to trigger the type guard."""
        del messages, tokenize
        return [1, 2, 3]

    def encode(self, text: str) -> list[int]:
        """Unused fallback kept for interface completeness."""
        del text
        return [1, 2]


class FakeLayer:
    """Simple residual layer that adds a fixed delta."""

    def __init__(self, delta: np.ndarray) -> None:
        self.delta = delta.reshape(1, 1, -1)

    def __call__(
        self,
        h: np.ndarray,
        mask: np.ndarray | None,
        *,
        cache: object | None = None,
    ) -> np.ndarray:
        """Apply the fixed residual update."""
        del mask, cache
        return h + self.delta


class FakeTransformer:
    """Transformer stub exposing a ``layers`` list."""

    def __init__(self, layers: list[FakeLayer]) -> None:
        self.layers = layers


class FakeSentenceTransformer:
    """Sentence-transformer stub for CLI tests."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode(
        self,
        prompts: list[str],
        *,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        """Encode prompts into simple 2D embeddings."""
        del show_progress_bar, convert_to_numpy
        vectors: list[list[float]] = []
        for index, prompt in enumerate(prompts):
            if "harm" in prompt:
                vectors.append([1.0, float(index)])
            elif "safe" in prompt:
                vectors.append([0.0, float(index) + 1.0])
            else:
                vectors.append([0.5, float(index) + 0.5])
        return np.array(vectors, dtype=float)


class FakeSentenceTransformersModule(ModuleType):
    """Typed fake ``sentence_transformers`` module."""

    SentenceTransformer: type[FakeSentenceTransformer]


class FakeScipyStatsModule(ModuleType):
    """Typed fake ``scipy.stats`` module."""

    spearmanr: object


class FakeScipyModule(ModuleType):
    """Typed fake top-level ``scipy`` module."""

    stats: FakeScipyStatsModule


def _write_jsonl(path: Path, entries: list[dict[str, str]]) -> None:
    """Write JSONL entries to disk."""
    path.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries),
    )


def _write_measure_datasets(tmp_path: Path) -> None:
    """Write the bundled harmful/harmless datasets needed by extraction."""
    _write_jsonl(
        tmp_path / "harmful.jsonl",
        [{"prompt": "harm-measure-a"}, {"prompt": "harm-measure-b"}],
    )
    _write_jsonl(
        tmp_path / "harmless.jsonl",
        [{"prompt": "safe-measure-a"}, {"prompt": "safe-measure-b"}],
    )


def _install_fake_model_stack(
    monkeypatch: pytest.MonkeyPatch,
    tokenizer: FakeTokenizer | FakeNonStringTokenizer,
    direction_result: FakeDirectionResult,
    layer_deltas: list[np.ndarray],
) -> None:
    """Patch the model stack used by ``extract_activations``."""
    transformer = FakeTransformer(
        [FakeLayer(delta) for delta in layer_deltas],
    )

    def _load_model(
        model_path: str,
    ) -> tuple[object, FakeTokenizer | FakeNonStringTokenizer]:
        del model_path
        return object(), tokenizer

    def _measure(
        model: object,
        tokenizer: FakeTokenizer | FakeNonStringTokenizer,
        harmful_prompts: list[str],
        harmless_prompts: list[str],
    ) -> FakeDirectionResult:
        del model, tokenizer, harmful_prompts, harmless_prompts
        return direction_result

    def _get_transformer(model: object) -> FakeTransformer:
        del model
        return transformer

    def _embed_and_mask(
        model_transformer: FakeTransformer,
        token_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        del model_transformer
        seq_len = int(token_ids.shape[1])
        hidden = np.zeros(
            (1, seq_len, direction_result.d_model),
            dtype=float,
        )
        mask = np.ones((1, seq_len), dtype=float)
        return hidden, mask

    def _make_ssm_mask(
        model_transformer: FakeTransformer,
        hidden: np.ndarray,
    ) -> None:
        del model_transformer, hidden
        return None

    def _select_mask(
        layer: FakeLayer,
        mask: np.ndarray | None,
        ssm_mask: np.ndarray | None,
    ) -> np.ndarray | None:
        del layer, ssm_mask
        return mask

    def _force_eval(*args: object) -> None:
        del args

    def _ops_array(values: list[int]) -> np.ndarray:
        return np.array(values, dtype=int)

    monkeypatch.setattr(model_io_module, "load_model", _load_model)
    monkeypatch.setattr(measure_module, "measure", _measure)
    monkeypatch.setattr(forward_module, "get_transformer", _get_transformer)
    monkeypatch.setattr(forward_module, "embed_and_mask", _embed_and_mask)
    monkeypatch.setattr(forward_module, "make_ssm_mask", _make_ssm_mask)
    monkeypatch.setattr(forward_module, "select_mask", _select_mask)
    monkeypatch.setattr(forward_module, "force_eval", _force_eval)
    monkeypatch.setattr(ops_module, "array", _ops_array, raising=False)
    monkeypatch.setattr(ops_module, "sum", np.sum, raising=False)
    monkeypatch.setattr(ops_module, "float32", np.float32, raising=False)


def _install_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Install a fake ``sentence_transformers`` module."""
    module = FakeSentenceTransformersModule("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def _install_fake_scipy(
    monkeypatch: pytest.MonkeyPatch,
    rho: float,
    p_value: float,
) -> None:
    """Install a fake ``scipy.stats`` module."""
    scipy_module = FakeScipyModule("scipy")
    stats_module = FakeScipyStatsModule("scipy.stats")

    def _spearmanr(
        values_a: np.ndarray,
        values_b: np.ndarray,
    ) -> tuple[float, float]:
        del values_a, values_b
        return rho, p_value

    stats_module.spearmanr = _spearmanr
    scipy_module.stats = stats_module
    monkeypatch.setitem(sys.modules, "scipy", scipy_module)
    monkeypatch.setitem(sys.modules, "scipy.stats", stats_module)


class TestVerifyDiversityModelHelpers:
    """Tests for helper functions."""

    def test_load_jsonl_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "sample.jsonl"
        path.write_text('{"prompt":"a"}\n\n{"prompt":"b"}\n')

        entries = verify_diversity_model.load_jsonl(path)

        assert entries == [{"prompt": "a"}, {"prompt": "b"}]

    def test_cosine_similarity_matrix(self) -> None:
        embeddings = np.array([[1.0, 0.0], [0.0, 3.0], [1.0, 1.0]])

        sim = verify_diversity_model.cosine_similarity_matrix(embeddings)

        assert sim.shape == (3, 3)
        assert sim[0, 0] == pytest.approx(1.0)
        assert sim[0, 1] == pytest.approx(0.0)
        assert sim[0, 2] == pytest.approx(1 / np.sqrt(2))

    def test_extract_activations_uses_measured_best_layer(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _write_measure_datasets(tmp_path)
        monkeypatch.setattr(
            verify_diversity_model,
            "__file__",
            str(tmp_path / "verify_diversity_model.py"),
        )
        direction_result = FakeDirectionResult(
            direction=np.array([2.0, 0.0], dtype=float),
            layer_index=1,
            d_model=2,
            cosine_scores=[0.1, 0.8, 0.2],
        )
        _install_fake_model_stack(
            monkeypatch,
            FakeTokenizer(),
            direction_result,
            [
                np.array([1.0, 0.0], dtype=float),
                np.array([0.5, -0.5], dtype=float),
                np.array([9.0, 9.0], dtype=float),
            ],
        )

        prompts = [f"prompt-{index}" for index in range(25)]
        activations, projections, best_layer, scores = (
            verify_diversity_model.extract_activations(
                "fake-model",
                prompts,
            )
        )

        out = capsys.readouterr().out
        assert activations.shape == (25, 2)
        assert np.allclose(activations[0], np.array([1.5, -0.5]))
        assert np.allclose(projections, np.full(25, 3.0))
        assert best_layer == 1
        assert scores == [0.1, 0.8, 0.2]
        assert "25/25 done" in out

    def test_extract_activations_respects_explicit_layer_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _write_measure_datasets(tmp_path)
        monkeypatch.setattr(
            verify_diversity_model,
            "__file__",
            str(tmp_path / "verify_diversity_model.py"),
        )
        direction_result = FakeDirectionResult(
            direction=np.array([1.0, 1.0], dtype=float),
            layer_index=2,
            d_model=2,
            cosine_scores=[0.2, 0.4, 0.9],
        )
        _install_fake_model_stack(
            monkeypatch,
            FakeTokenizer(),
            direction_result,
            [
                np.array([2.0, 1.0], dtype=float),
                np.array([5.0, 5.0], dtype=float),
                np.array([8.0, 8.0], dtype=float),
            ],
        )

        activations, projections, best_layer, _scores = (
            verify_diversity_model.extract_activations(
                "fake-model",
                ["prompt-0"],
                layer_index=0,
            )
        )

        assert best_layer == 0
        assert np.allclose(activations, np.array([[2.0, 1.0]]))
        assert np.allclose(projections, np.array([3.0]))

    def test_extract_activations_requires_string_chat_template(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _write_measure_datasets(tmp_path)
        monkeypatch.setattr(
            verify_diversity_model,
            "__file__",
            str(tmp_path / "verify_diversity_model.py"),
        )
        direction_result = FakeDirectionResult(
            direction=np.array([1.0, 0.0], dtype=float),
            layer_index=0,
            d_model=2,
            cosine_scores=[0.7],
        )
        _install_fake_model_stack(
            monkeypatch,
            FakeNonStringTokenizer(),
            direction_result,
            [np.array([1.0, 0.0], dtype=float)],
        )

        with pytest.raises(
            TypeError,
            match="apply_chat_template must return str",
        ):
            verify_diversity_model.extract_activations(
                "fake-model",
                ["prompt-0"],
            )

    def test_analyze_set_reports_category_spread_and_duplicates(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        entries = [
            {"prompt": "alpha one", "category": "alpha"},
            {"prompt": "alpha two", "category": "alpha"},
            {"prompt": "beta one", "category": "beta"},
        ]
        activations = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.01],
                [0.0, 1.0],
            ],
            dtype=float,
        )
        projections = np.array([1.0, 2.0, -1.0], dtype=float)

        stats = verify_diversity_model.analyze_set(
            "sample",
            entries,
            activations,
            projections,
            0.95,
        )

        out = capsys.readouterr().out
        assert stats["n_prompts"] == 3
        assert stats["act_near_duplicates"] == 1
        assert stats["proj_range"] == pytest.approx(3.0)
        assert stats["category_proj_spread"] == pytest.approx(1.25)
        assert "Per-category refusal projection" in out

    @pytest.mark.parametrize(
        ("correlation", "expected_label"),
        [
            (0.2, "WEAK correlation"),
            (0.4, "MODERATE correlation"),
            (0.9, "STRONG correlation"),
        ],
    )
    def test_compare_spaces_prints_interpretation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        correlation: float,
        expected_label: str,
    ) -> None:
        _install_fake_scipy(monkeypatch, rho=0.33, p_value=0.01)

        def _corrcoef(
            values_a: np.ndarray,
            values_b: np.ndarray,
        ) -> np.ndarray:
            del values_a, values_b
            return np.array(
                [[1.0, correlation], [correlation, 1.0]],
                dtype=float,
            )

        monkeypatch.setattr(
            verify_diversity_model.np,
            "corrcoef",
            _corrcoef,
        )

        result = verify_diversity_model.compare_spaces(
            "sample",
            np.eye(3, dtype=float),
            np.eye(3, dtype=float),
            3,
        )

        out = capsys.readouterr().out
        assert expected_label in out
        assert result["pearson_r"] == pytest.approx(correlation)
        assert result["spearman_rho"] == pytest.approx(0.33)


class TestVerifyDiversityModelMain:
    """Tests for ``main()``."""

    def test_main_exits_when_no_datasets_found(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            verify_diversity_model,
            "__file__",
            str(tmp_path / "verify_diversity_model.py"),
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["verify_diversity_model"],
        )

        with pytest.raises(SystemExit, match="1"):
            verify_diversity_model.main()

        out = capsys.readouterr().out
        assert "No datasets found." in out

    def test_main_runs_with_sentence_transformer_comparison(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _install_sentence_transformers(monkeypatch)
        monkeypatch.setattr(
            verify_diversity_model,
            "__file__",
            str(tmp_path / "verify_diversity_model.py"),
        )
        _write_jsonl(
            tmp_path / "harmful_100.jsonl",
            [{"prompt": "harm-a"}, {"prompt": "harm-b"}],
        )
        _write_jsonl(
            tmp_path / "harmless_100.jsonl",
            [{"prompt": "safe-a"}, {"prompt": "safe-b"}],
        )
        _write_jsonl(
            tmp_path / "harmful_infix_100.jsonl",
            [{"prompt": "infix-a"}, {"prompt": "infix-b"}],
        )
        (tmp_path / "diversity_report.json").write_text("{}")

        def _fake_extract_activations(
            model_path: str,
            prompts: list[str],
            layer_index: int | None = None,
        ) -> tuple[np.ndarray, np.ndarray, int, list[float]]:
            del model_path, prompts, layer_index
            activations = np.array(
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                    [0.1, 0.9],
                    [0.8, 0.2],
                    [0.7, 0.3],
                ],
                dtype=float,
            )
            projections = np.array(
                [2.0, 1.0, -1.0, 0.0, 0.2, 0.3],
                dtype=float,
            )
            return activations, projections, 3, [0.1, 0.9, 0.2]

        compared_sets: list[str] = []

        def _fake_analyze_set(
            name: str,
            entries: list[dict[str, str]],
            activations: np.ndarray,
            projections: np.ndarray,
            threshold: float,
        ) -> verify_diversity_model.ModelSection:
            del entries, activations, projections, threshold
            mapping: dict[str, verify_diversity_model.ModelSection] = {
                "harmful_100": {
                    "n_prompts": 2,
                    "act_mean_sim": 0.25,
                    "act_near_duplicates": 0,
                    "proj_range": 1.5,
                    "category_proj_spread": 0.2,
                },
                "harmless_100": {
                    "n_prompts": 2,
                    "act_mean_sim": 0.35,
                    "act_near_duplicates": 0,
                    "proj_range": 0.5,
                    "category_proj_spread": 0.0,
                },
                "harmful_infix_100": {
                    "n_prompts": 2,
                    "act_mean_sim": 0.8,
                    "act_near_duplicates": 3,
                    "proj_range": 0.7,
                    "category_proj_spread": 0.1,
                },
            }
            return mapping[name]

        def _fake_compare_spaces(
            name: str,
            act_sim: np.ndarray,
            st_sim: np.ndarray,
            n: int,
        ) -> verify_diversity_model.ModelSection:
            del act_sim, st_sim, n
            compared_sets.append(name)
            return {
                "pearson_r": 0.5,
                "spearman_rho": 0.4,
                "spearman_p": 0.01,
            }

        times = iter([10.0, 12.2])

        monkeypatch.setattr(
            verify_diversity_model,
            "extract_activations",
            _fake_extract_activations,
        )
        monkeypatch.setattr(
            verify_diversity_model,
            "analyze_set",
            _fake_analyze_set,
        )
        monkeypatch.setattr(
            verify_diversity_model,
            "compare_spaces",
            _fake_compare_spaces,
        )
        monkeypatch.setattr(
            verify_diversity_model.time,
            "time",
            lambda: next(times),
        )

        output_path = tmp_path / "model-report.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "verify_diversity_model",
                "--model",
                "fake-model",
                "--output",
                str(output_path),
            ],
        )

        verify_diversity_model.main()

        out = capsys.readouterr().out
        report = json.loads(output_path.read_text())
        assert report["model"] == "fake-model"
        assert report["best_layer"] == 3
        assert report["extraction_time_s"] == pytest.approx(2.2)
        assert compared_sets == [
            "harmful_100",
            "harmless_100",
            "harmful_infix_100",
        ]
        assert report["comparison_harmful_100"]["pearson_r"] == pytest.approx(
            0.5,
        )
        assert report["cross_harmful_harmless"]["cohens_d"] > 0.8
        assert "[PASS]" in out
        assert "[REVIEW]" in out
        assert "[FAIL]" in out
        assert "[GOOD" in out

    def test_main_loads_cached_sentence_transformer_report(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            verify_diversity_model,
            "__file__",
            str(tmp_path / "verify_diversity_model.py"),
        )
        _write_jsonl(
            tmp_path / "harmful_100.jsonl",
            [{"prompt": "harm-a"}, {"prompt": "harm-b"}],
        )
        _write_jsonl(
            tmp_path / "harmless_100.jsonl",
            [{"prompt": "safe-a"}, {"prompt": "safe-b"}],
        )
        _write_jsonl(
            tmp_path / "harmful_infix_100.jsonl",
            [{"prompt": "infix-a"}, {"prompt": "infix-b"}],
        )
        (tmp_path / "diversity_report.json").write_text(
            json.dumps({"cached": 1}),
        )

        def _fake_extract_activations(
            model_path: str,
            prompts: list[str],
            layer_index: int | None = None,
        ) -> tuple[np.ndarray, np.ndarray, int, list[float]]:
            del model_path, prompts, layer_index
            activations = np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.9, 0.1],
                    [0.1, 0.9],
                    [0.8, 0.2],
                    [0.2, 0.8],
                ],
                dtype=float,
            )
            projections = np.array(
                [1.0, 0.0, 0.9, -0.1, 0.2, 0.1],
                dtype=float,
            )
            return activations, projections, 1, [0.6, 0.7]

        def _fake_analyze_set(
            name: str,
            entries: list[dict[str, str]],
            activations: np.ndarray,
            projections: np.ndarray,
            threshold: float,
        ) -> verify_diversity_model.ModelSection:
            del name, entries, activations, projections, threshold
            return {
                "n_prompts": 2,
                "act_mean_sim": 0.4,
                "act_near_duplicates": 0,
                "proj_range": 1.2,
                "category_proj_spread": 0.0,
            }

        monkeypatch.setitem(
            sys.modules,
            "sentence_transformers",
            ModuleType("sentence_transformers"),
        )
        monkeypatch.setattr(
            verify_diversity_model,
            "extract_activations",
            _fake_extract_activations,
        )
        monkeypatch.setattr(
            verify_diversity_model,
            "analyze_set",
            _fake_analyze_set,
        )

        output_path = tmp_path / "cached-report.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "verify_diversity_model",
                "--output",
                str(output_path),
            ],
        )

        verify_diversity_model.main()

        out = capsys.readouterr().out
        report = json.loads(output_path.read_text())
        assert report["st_report_loaded"] is True
        assert report["cross_harmful_harmless"]["cohens_d"] == pytest.approx(0.2)
        assert "Loaded cached ST report" in out
        assert "[WEAK" in out

    def test_main_serializes_numpy_scalars_and_skips_non_dict_entries(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            verify_diversity_model,
            "__file__",
            str(tmp_path / "verify_diversity_model.py"),
        )
        _write_jsonl(
            tmp_path / "harmful_100.jsonl",
            [{"prompt": "harm-a"}, {"prompt": "harm-b"}],
        )
        _write_jsonl(
            tmp_path / "harmless_100.jsonl",
            [{"prompt": "safe-a"}, {"prompt": "safe-b"}],
        )
        _write_jsonl(
            tmp_path / "harmful_infix_100.jsonl",
            [{"prompt": "infix-a"}, {"prompt": "infix-b"}],
        )

        def _fake_extract_activations(
            model_path: str,
            prompts: list[str],
            layer_index: int | None = None,
        ) -> tuple[np.ndarray, np.ndarray, int, list[float]]:
            del model_path, prompts, layer_index
            activations = np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.8, 0.2],
                    [0.2, 0.8],
                    [0.7, 0.3],
                    [0.6, 0.4],
                ],
                dtype=float,
            )
            projections = np.array(
                [1.0, 0.0, 0.6, -0.4, 0.1, 0.0],
                dtype=float,
            )
            return (
                activations,
                projections,
                cast("int", np.int64(2)),
                [cast("float", np.float64(0.1))],
            )

        def _fake_analyze_set(
            name: str,
            entries: list[dict[str, str]],
            activations: np.ndarray,
            projections: np.ndarray,
            threshold: float,
        ) -> verify_diversity_model.ModelSection:
            del entries, activations, projections, threshold
            if name == "harmful_infix_100":
                return cast("verify_diversity_model.ModelSection", "skip-me")
            return cast(
                "verify_diversity_model.ModelSection",
                {
                    "n_prompts": np.int64(2),
                    "act_mean_sim": 0.4,
                    "act_near_duplicates": np.int64(0),
                    "proj_range": 1.1,
                    "category_proj_spread": 0.0,
                    "proj_values": [np.float64(0.1), np.float64(0.2)],
                },
            )

        times = iter([np.float64(5.0), np.float64(6.2)])

        monkeypatch.setattr(
            verify_diversity_model,
            "extract_activations",
            _fake_extract_activations,
        )
        monkeypatch.setattr(
            verify_diversity_model,
            "analyze_set",
            _fake_analyze_set,
        )

        output_path = tmp_path / "serialized-report.json"
        monkeypatch.setattr(
            verify_diversity_model.time,
            "time",
            lambda: next(times),
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "verify_diversity_model",
                "--output",
                str(output_path),
                "--st-report",
                str(tmp_path / "missing-st-report.json"),
            ],
        )

        verify_diversity_model.main()

        out = capsys.readouterr().out
        report = json.loads(output_path.read_text())
        assert report["best_layer"] == 2
        assert report["extraction_time_s"] == pytest.approx(1.2)
        assert report["harmful_100"]["n_prompts"] == 2
        assert report["harmful_100"]["proj_values"] == [0.1, 0.2]
        assert report["harmful_infix_100"] == "skip-me"
        assert "MEDIUM separation" in out

    def test_module_entrypoint_runs_from_copied_script(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        direction_result = FakeDirectionResult(
            direction=np.array([1.0, 0.0], dtype=float),
            layer_index=0,
            d_model=2,
            cosine_scores=[0.6],
        )
        _install_fake_model_stack(
            monkeypatch,
            FakeTokenizer(),
            direction_result,
            [np.array([1.0, 0.0], dtype=float)],
        )

        output_path = tmp_path / "entrypoint-model-report.json"
        monkeypatch.setattr("time.time", lambda: 100.0)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "verify_diversity_model",
                "--model",
                "fake-model",
                "--output",
                str(output_path),
                "--st-report",
                str(tmp_path / "missing-st-report.json"),
            ],
        )

        runpy.run_path(
            str(verify_diversity_model.__file__),
            run_name="__main__",
        )

        report = json.loads(output_path.read_text())
        assert report["model"] == "fake-model"
