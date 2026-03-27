# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban.geometry: direction composition analysis."""

import pytest

from vauban import _ops as ops
from vauban.geometry import analyze_directions


class TestAnalyzeDirections:
    def test_identical_directions(self) -> None:
        d = ops.array([1.0, 0.0, 0.0])
        ops.eval(d)
        result = analyze_directions({"a": d, "b": d})
        assert len(result.pairwise) == 1
        assert result.pairwise[0].cosine_similarity == pytest.approx(1.0, abs=1e-4)
        assert result.pairwise[0].shared_variance == pytest.approx(1.0, abs=1e-4)
        assert result.pairwise[0].independent is False

    def test_orthogonal_directions(self) -> None:
        a = ops.array([1.0, 0.0, 0.0])
        b = ops.array([0.0, 1.0, 0.0])
        ops.eval(a, b)
        result = analyze_directions({"a": a, "b": b})
        assert len(result.pairwise) == 1
        assert result.pairwise[0].cosine_similarity == pytest.approx(0.0, abs=1e-4)
        assert result.pairwise[0].shared_variance == pytest.approx(0.0, abs=1e-4)
        assert result.pairwise[0].independent is True

    def test_three_directions_three_pairs(self) -> None:
        a = ops.array([1.0, 0.0, 0.0])
        b = ops.array([0.0, 1.0, 0.0])
        c = ops.array([0.0, 0.0, 1.0])
        ops.eval(a, b, c)
        result = analyze_directions({"a": a, "b": b, "c": c})
        assert len(result.pairwise) == 3
        assert len(result.direction_names) == 3
        # All orthogonal -> all independent
        assert result.mean_independence == 1.0

    def test_single_direction_zero_pairs(self) -> None:
        d = ops.array([1.0, 0.0, 0.0])
        ops.eval(d)
        result = analyze_directions({"only": d})
        assert len(result.pairwise) == 0
        assert result.most_aligned_pair is None
        assert result.most_orthogonal_pair is None
        assert result.mean_independence == 0.0

    def test_cosine_matrix_shape(self) -> None:
        a = ops.array([1.0, 0.0])
        b = ops.array([0.0, 1.0])
        ops.eval(a, b)
        result = analyze_directions({"a": a, "b": b})
        assert len(result.cosine_matrix) == 2
        assert len(result.cosine_matrix[0]) == 2
        # Diagonal is 1.0
        assert result.cosine_matrix[0][0] == pytest.approx(1.0)
        assert result.cosine_matrix[1][1] == pytest.approx(1.0)

    def test_most_aligned_and_orthogonal(self) -> None:
        a = ops.array([1.0, 0.0, 0.0])
        b = ops.array([0.9, 0.436, 0.0])  # close to a
        c = ops.array([0.0, 0.0, 1.0])  # orthogonal to both
        ops.eval(a, b, c)
        result = analyze_directions({"a": a, "b": b, "c": c})
        assert result.most_aligned_pair is not None
        assert {"a", "b"} == {
            result.most_aligned_pair.name_a,
            result.most_aligned_pair.name_b,
        }
        assert result.most_orthogonal_pair is not None
        # One of the pairs with c should be most orthogonal
        assert "c" in {
            result.most_orthogonal_pair.name_a,
            result.most_orthogonal_pair.name_b,
        }

    def test_to_dict_output(self) -> None:
        a = ops.array([1.0, 0.0])
        b = ops.array([0.0, 1.0])
        ops.eval(a, b)
        result = analyze_directions({"a": a, "b": b})
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "direction_names" in d
        assert "pairwise" in d
        assert "cosine_matrix" in d
        assert "mean_independence" in d
        assert "most_aligned_pair" in d
        assert "most_orthogonal_pair" in d
        # Pairwise should be serializable dicts
        pairwise = d["pairwise"]
        assert isinstance(pairwise, list)
        assert len(pairwise) == 1
        assert isinstance(pairwise[0], dict)
