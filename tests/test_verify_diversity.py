# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``vauban.data.verify_diversity``."""

from __future__ import annotations

import json
import runpy
import sys
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import pytest

import vauban.data.verify_diversity as verify_diversity

if TYPE_CHECKING:
    from pathlib import Path


class FakeSentenceTransformer:
    """Small sentence-transformer stub with deterministic embeddings."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode(
        self,
        prompts: list[str],
        *,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        """Encode prompts into simple 2D vectors."""
        del show_progress_bar, convert_to_numpy
        mapping: dict[str, list[float]] = {
            "harm-a": [1.0, 0.0],
            "harm-b": [0.0, 1.0],
            "safe-a": [0.8, 0.2],
            "safe-b": [0.2, 0.8],
            "infix-a": [1.0, 0.0],
            "infix-b": [1.0, 0.01],
        }
        vectors: list[list[float]] = []
        for index, prompt in enumerate(prompts):
            if prompt in mapping:
                vectors.append(mapping[prompt])
            else:
                vectors.append([float(len(prompt)), float(index) + 1.0])
        return np.array(vectors, dtype=float)


class FakeSentenceTransformersModule(ModuleType):
    """Typed fake ``sentence_transformers`` module."""

    SentenceTransformer: type[FakeSentenceTransformer]


def _write_jsonl(path: Path, entries: list[dict[str, str]]) -> None:
    """Write JSONL entries to disk."""
    path.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries),
    )


def _install_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Install a fake ``sentence_transformers`` module."""
    module = FakeSentenceTransformersModule("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


class TestVerifyDiversityHelpers:
    """Tests for helper utilities."""

    def test_load_jsonl_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "sample.jsonl"
        path.write_text('{"prompt":"a"}\n\n{"prompt":"b"}\n')

        entries = verify_diversity.load_jsonl(path)

        assert entries == [{"prompt": "a"}, {"prompt": "b"}]

    def test_cosine_similarity_matrix(self) -> None:
        embeddings = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])

        sim = verify_diversity.cosine_similarity_matrix(embeddings)

        assert sim.shape == (3, 3)
        assert sim[0, 0] == pytest.approx(1.0)
        assert sim[0, 1] == pytest.approx(0.0)
        assert sim[0, 2] == pytest.approx(1 / np.sqrt(2))

    def test_report_pairwise_stats_detects_near_duplicates(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        prompts = ["first prompt", "first prompt paraphrase", "other topic"]
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
            ],
        )

        stats = verify_diversity.report_pairwise_stats(
            "sample",
            prompts,
            embeddings,
            0.9,
        )

        out = capsys.readouterr().out
        assert stats["n_prompts"] == 3
        assert stats["n_near_duplicates"] == 1
        near_duplicates = stats["near_duplicates"]
        assert isinstance(near_duplicates, list)
        assert near_duplicates == [(0, 1, pytest.approx(0.9999, abs=1e-4))]
        assert "Top near-duplicates" in out

    def test_report_category_stats_handles_singletons_and_centroids(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        entries = [
            {"prompt": "a", "category": "alpha"},
            {"prompt": "b", "category": "alpha"},
            {"prompt": "c", "category": "beta"},
        ]
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
            ],
        )

        cat_stats = verify_diversity.report_category_stats(
            "sample",
            entries,
            embeddings,
        )

        out = capsys.readouterr().out
        assert "Inter-category centroid similarity" in out
        assert "alpha" in cat_stats
        assert "beta" not in cat_stats
        assert cat_stats["alpha"]["n"] == 2

    def test_report_cross_set_finds_cross_duplicates(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        emb_a = np.array([[1.0, 0.0], [0.0, 1.0]])
        emb_b = np.array([[1.0, 0.0], [1.0, 0.1]])

        stats = verify_diversity.report_cross_set(
            "set_a",
            "set_b",
            emb_a,
            emb_b,
            ["alpha", "beta"],
            ["alpha clone", "alpha variant"],
            0.95,
        )

        out = capsys.readouterr().out
        assert stats["n_cross_duplicates"] == 2
        assert "Top cross-set matches" in out


class TestVerifyDiversityMain:
    """Tests for ``main()``."""

    def test_main_exits_when_sentence_transformers_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        module = ModuleType("sentence_transformers")
        monkeypatch.setitem(sys.modules, "sentence_transformers", module)
        monkeypatch.setattr(
            sys,
            "argv",
            ["verify_diversity"],
        )

        with pytest.raises(SystemExit, match="1"):
            verify_diversity.main()

        out = capsys.readouterr().out
        assert "sentence-transformers not installed" in out

    def test_main_exits_when_no_datasets_found(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _install_sentence_transformers(monkeypatch)
        monkeypatch.setattr(
            verify_diversity,
            "__file__",
            str(tmp_path / "verify_diversity.py"),
        )
        monkeypatch.setattr(sys, "argv", ["verify_diversity"])

        with pytest.raises(SystemExit, match="1"):
            verify_diversity.main()

        out = capsys.readouterr().out
        assert "No datasets found." in out

    def test_main_runs_and_serializes_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _install_sentence_transformers(monkeypatch)
        monkeypatch.setattr(
            verify_diversity,
            "__file__",
            str(tmp_path / "verify_diversity.py"),
        )

        data_sets: dict[str, list[dict[str, str]]] = {
            "harmful.jsonl": [
                {"prompt": "harm-a", "category": "red"},
                {"prompt": "harm-b", "category": "blue"},
            ],
            "harmless.jsonl": [
                {"prompt": "safe-a", "category": "green"},
                {"prompt": "safe-b", "category": "green"},
            ],
            "harmful_100.jsonl": [
                {"prompt": "harm-a"},
                {"prompt": "harm-b"},
            ],
            "harmless_100.jsonl": [
                {"prompt": "safe-a"},
                {"prompt": "safe-b"},
            ],
            "harmful_infix_100.jsonl": [
                {"prompt": "infix-a"},
                {"prompt": "infix-b"},
            ],
            "harmful_infix.jsonl": [
                {"prompt": "infix-a"},
                {"prompt": "infix-b"},
            ],
        }
        for filename, entries in data_sets.items():
            _write_jsonl(tmp_path / filename, entries)

        category_calls: list[str] = []

        def _fake_report_pairwise_stats(
            name: str,
            prompts: list[str],
            embeddings: np.ndarray,
            threshold: float,
        ) -> verify_diversity.DiversityStats:
            del prompts, embeddings, threshold
            mapping: dict[str, verify_diversity.DiversityStats] = {
                "harmful_original": {
                    "n_prompts": 2,
                    "mean_similarity": 0.5,
                    "n_near_duplicates": 0,
                    "near_duplicates": [],
                },
                "harmless_original": {
                    "n_prompts": 2,
                    "mean_similarity": 0.75,
                    "n_near_duplicates": 2,
                    "near_duplicates": [(0, 1, 0.97)],
                },
                "harmful_100": {
                    "n_prompts": 2,
                    "mean_similarity": 0.95,
                    "n_near_duplicates": 6,
                    "near_duplicates": [(0, 1, 0.99)],
                },
                "harmless_100": {
                    "n_prompts": 2,
                    "mean_similarity": 0.6,
                    "n_near_duplicates": 0,
                    "near_duplicates": [],
                },
                "harmful_infix_100": {
                    "n_prompts": 2,
                    "mean_similarity": 0.65,
                    "n_near_duplicates": 1,
                    "near_duplicates": [(0, 1, 0.96)],
                },
                "harmful_infix_original": {
                    "n_prompts": 2,
                    "mean_similarity": 0.55,
                    "n_near_duplicates": 0,
                    "near_duplicates": [],
                },
            }
            return mapping[name]

        def _fake_report_category_stats(
            name: str,
            entries: list[dict[str, str]],
            embeddings: np.ndarray,
        ) -> dict[str, dict[str, float]]:
            del entries, embeddings
            category_calls.append(name)
            return {"category": {"intra_mean": 0.1, "intra_std": 0.2, "n": 2}}

        def _fake_report_cross_set(
            name_a: str,
            name_b: str,
            emb_a: np.ndarray,
            emb_b: np.ndarray,
            prompts_a: list[str],
            prompts_b: list[str],
            threshold: float,
        ) -> verify_diversity.DiversityStats:
            del emb_a, emb_b, prompts_a, prompts_b, threshold
            return {
                "mean_cross_similarity": 0.12,
                "max_cross_similarity": 0.34,
                "n_cross_duplicates": 1 if name_a == "harmful_original" else 0,
            }

        monkeypatch.setattr(
            verify_diversity,
            "report_pairwise_stats",
            _fake_report_pairwise_stats,
        )
        monkeypatch.setattr(
            verify_diversity,
            "report_category_stats",
            _fake_report_category_stats,
        )
        monkeypatch.setattr(
            verify_diversity,
            "report_cross_set",
            _fake_report_cross_set,
        )

        output_path = tmp_path / "report.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "verify_diversity",
                "--model",
                "fake-model",
                "--threshold",
                "0.9",
                "--output",
                str(output_path),
            ],
        )

        verify_diversity.main()

        report = json.loads(output_path.read_text())
        assert "surface" not in report
        assert category_calls == ["harmful_original", "harmless_original"]
        assert report["harmless_original"]["near_duplicates"] == [[0, 1, 0.97]]
        assert report["cross_harmful_original_vs_harmful_100"] == {
            "mean_cross_similarity": 0.12,
            "max_cross_similarity": 0.34,
            "n_cross_duplicates": 1,
        }

    def test_module_entrypoint_runs_from_copied_script(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _install_sentence_transformers(monkeypatch)

        output_path = tmp_path / "entrypoint-report.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "verify_diversity",
                "--model",
                "fake-model",
                "--output",
                str(output_path),
            ],
        )

        runpy.run_path(str(verify_diversity.__file__), run_name="__main__")

        report = json.loads(output_path.read_text())
        assert "harmful_original" in report
