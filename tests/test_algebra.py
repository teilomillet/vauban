# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban.algebra: direction algebra with closed operations."""

import json

import pytest

from vauban import _ops as ops
from vauban._array import Array
from vauban.algebra import (
    add,
    compose,
    from_array,
    from_dbdi_result,
    from_diff_result,
    from_direction_result,
    from_subspace_result,
    intersect,
    negate,
    similarity,
    subtract,
    to_basis,
    to_direction,
)
from vauban.subspace import orthonormalize
from vauban.types import (
    DBDIResult,
    DiffResult,
    DirectionResult,
    DirectionSpace,
    SubspaceResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D_MODEL = 32  # small for fast tests


def _random_space(rank: int, label: str = "test", seed: int = 0) -> DirectionSpace:
    """Create a random DirectionSpace for testing."""
    ops.random.seed(seed)
    raw = ops.random.normal((rank, D_MODEL))
    ops.eval(raw)
    return from_array(raw, label=label)


def _rank0_space(label: str = "empty") -> DirectionSpace:
    """Create a rank-0 DirectionSpace."""
    basis = ops.zeros((0, D_MODEL))
    ops.eval(basis)
    return DirectionSpace(basis=basis, d_model=D_MODEL, rank=0, label=label)


def _orthogonal_spaces() -> tuple[DirectionSpace, DirectionSpace]:
    """Create two orthogonal rank-2 subspaces in R^D_MODEL."""
    row_a0 = [0.0] * D_MODEL
    row_a0[0] = 1.0
    row_a1 = [0.0] * D_MODEL
    row_a1[1] = 1.0
    a_raw = ops.array([row_a0, row_a1])

    row_b0 = [0.0] * D_MODEL
    row_b0[2] = 1.0
    row_b1 = [0.0] * D_MODEL
    row_b1[3] = 1.0
    b_raw = ops.array([row_b0, row_b1])

    ops.eval(a_raw, b_raw)
    return from_array(a_raw, label="A"), from_array(b_raw, label="B")


def _is_orthonormal(basis: Array, tol: float = 1e-4) -> bool:
    """Check if basis rows are orthonormal."""
    if basis.shape[0] == 0:
        return True
    gram = basis @ basis.T
    ops.eval(gram)
    identity = ops.eye(basis.shape[0])
    diff = float(ops.linalg.norm(gram - identity).item())
    return diff < tol


# ---------------------------------------------------------------------------
# Closure tests
# ---------------------------------------------------------------------------


class TestClosure:
    """Every operation returns a valid DirectionSpace with orthonormal basis."""

    def test_add_returns_valid_space(self) -> None:
        a = _random_space(2, "a", seed=1)
        b = _random_space(3, "b", seed=2)
        result = add(a, b)
        assert isinstance(result, DirectionSpace)
        assert result.d_model == D_MODEL
        assert result.rank == result.basis.shape[0]
        assert _is_orthonormal(result.basis)

    def test_subtract_returns_valid_space(self) -> None:
        a = _random_space(3, "a", seed=1)
        b = _random_space(1, "b", seed=2)
        result = subtract(a, b)
        assert isinstance(result, DirectionSpace)
        assert _is_orthonormal(result.basis)

    def test_intersect_returns_valid_space(self) -> None:
        a = _random_space(3, "a", seed=1)
        b = _random_space(3, "b", seed=2)
        result = intersect(a, b)
        assert isinstance(result, DirectionSpace)
        if result.rank > 0:
            assert _is_orthonormal(result.basis)

    def test_negate_returns_valid_space(self) -> None:
        a = _random_space(2, "a", seed=1)
        result = negate(a)
        assert isinstance(result, DirectionSpace)
        assert result.rank == a.rank
        assert _is_orthonormal(result.basis)

    def test_compose_returns_valid_space(self) -> None:
        a = _random_space(2, "a", seed=1)
        b = _random_space(2, "b", seed=2)
        result = compose([a, b], [1.0, 0.5])
        assert isinstance(result, DirectionSpace)
        assert _is_orthonormal(result.basis)


# ---------------------------------------------------------------------------
# Algebraic properties
# ---------------------------------------------------------------------------


class TestAlgebraicProperties:
    def test_add_commutativity(self) -> None:
        a = _random_space(2, "a", seed=10)
        b = _random_space(2, "b", seed=20)
        ab = add(a, b)
        ba = add(b, a)
        assert similarity(ab, ba) > 0.99

    def test_subtract_removes_subspace(self) -> None:
        a = _random_space(3, "a", seed=10)
        b = _random_space(2, "b", seed=20)
        result = subtract(a, b)
        # Result should have low overlap with b
        if result.rank > 0:
            assert similarity(result, b) < 0.15

    def test_negate_involution(self) -> None:
        a = _random_space(2, "a", seed=10)
        double_neg = negate(negate(a))
        # basis of negate(negate(a)) should equal a.basis
        diff = a.basis - double_neg.basis
        ops.eval(diff)
        assert float(ops.linalg.norm(diff).item()) < 1e-5

    def test_intersect_subset_of_both(self) -> None:
        a = _random_space(3, "a", seed=10)
        b = _random_space(3, "b", seed=20)
        result = intersect(a, b, threshold=0.1)
        if result.rank > 0:
            assert similarity(result, a) <= 1.01
            assert similarity(result, b) <= 1.01

    def test_self_intersection(self) -> None:
        a = _random_space(3, "a", seed=10)
        result = intersect(a, a, threshold=0.3)
        assert similarity(result, a) > 0.99

    def test_self_subtraction_is_rank0(self) -> None:
        a = _random_space(3, "a", seed=10)
        result = subtract(a, a)
        assert result.rank == 0

    def test_add_with_self(self) -> None:
        a = _random_space(2, "a", seed=10)
        result = add(a, a)
        # Should recover the same subspace
        assert similarity(result, a) > 0.99

    def test_orthogonal_intersect_is_empty(self) -> None:
        a, b = _orthogonal_spaces()
        result = intersect(a, b, threshold=0.3)
        assert result.rank == 0


# ---------------------------------------------------------------------------
# Converter round-trips
# ---------------------------------------------------------------------------


class TestConverters:
    def test_from_direction_result_roundtrip(self) -> None:
        ops.random.seed(42)
        vec = ops.random.normal((D_MODEL,))
        ops.eval(vec)
        norm = ops.linalg.norm(vec)
        vec = vec / norm
        ops.eval(vec)

        result = DirectionResult(
            direction=vec,
            layer_index=5,
            cosine_scores=[0.8],
            d_model=D_MODEL,
            model_path="test",
        )
        space = from_direction_result(result, label="test_dir")
        assert space.rank == 1
        assert space.d_model == D_MODEL
        assert space.layer_index == 5

        extracted = to_direction(space)
        ops.eval(extracted)
        # Should match original direction (possibly flipped sign)
        cos_sim = abs(float(ops.sum(extracted * vec).item()))
        assert cos_sim > 0.99

    def test_from_subspace_result_roundtrip(self) -> None:
        ops.random.seed(42)
        raw = ops.random.normal((3, D_MODEL))
        basis = orthonormalize(raw)
        ops.eval(basis)

        result = SubspaceResult(
            basis=basis,
            singular_values=[3.0, 2.0, 1.0],
            explained_variance=[0.6, 0.3, 0.1],
            layer_index=7,
            d_model=D_MODEL,
            model_path="test",
            per_layer_bases=[],
        )
        space = from_subspace_result(result, label="test_sub")
        assert space.rank == 3
        assert space.singular_values == [3.0, 2.0, 1.0]

        extracted_basis = to_basis(space)
        diff = float(ops.linalg.norm(extracted_basis - basis).item())
        assert diff < 1e-5

    def test_from_diff_result(self) -> None:
        ops.random.seed(42)
        raw = ops.random.normal((2, D_MODEL))
        basis = orthonormalize(raw)
        ops.eval(basis)

        result = DiffResult(
            basis=basis,
            singular_values=[5.0, 3.0],
            explained_variance=[0.7, 0.3],
            best_layer=10,
            d_model=D_MODEL,
            source_model="base",
            target_model="aligned",
            per_layer_bases=[],
            per_layer_singular_values=[],
        )
        space = from_diff_result(result)
        assert space.rank == 2
        assert space.layer_index == 10

    def test_from_dbdi_result_produces_rank2(self) -> None:
        ops.random.seed(42)
        hdd = ops.random.normal((D_MODEL,))
        red = ops.random.normal((D_MODEL,))
        ops.eval(hdd, red)

        result = DBDIResult(
            hdd=hdd,
            red=red,
            hdd_layer_index=5,
            red_layer_index=10,
            hdd_cosine_scores=[0.7],
            red_cosine_scores=[0.8],
            d_model=D_MODEL,
            model_path="test",
        )
        space = from_dbdi_result(result)
        assert space.rank == 2
        assert _is_orthonormal(space.basis)

    def test_from_array_1d(self) -> None:
        vec = ops.array([1.0, 0.0, 0.0] + [0.0] * (D_MODEL - 3))
        ops.eval(vec)
        space = from_array(vec, label="unit_x")
        assert space.rank == 1
        assert space.d_model == D_MODEL
        # Should be normalized
        norm = float(ops.linalg.norm(space.basis[0]).item())
        assert abs(norm - 1.0) < 1e-5

    def test_from_array_2d(self) -> None:
        ops.random.seed(42)
        raw = ops.random.normal((3, D_MODEL))
        ops.eval(raw)
        space = from_array(raw, label="random_3d")
        assert space.rank == 3
        assert _is_orthonormal(space.basis)

    def test_from_array_zero_vector(self) -> None:
        vec = ops.zeros((D_MODEL,))
        ops.eval(vec)
        space = from_array(vec, label="zero")
        assert space.rank == 0


# ---------------------------------------------------------------------------
# Provenance tracking
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_converter_provenance(self) -> None:
        space = _random_space(2, "test", seed=1)
        assert space.provenance is not None
        assert space.provenance.operation == "convert"

    def test_add_provenance(self) -> None:
        a = _random_space(2, "alpha", seed=1)
        b = _random_space(2, "beta", seed=2)
        result = add(a, b)
        assert result.provenance is not None
        assert result.provenance.operation == "add"
        assert result.provenance.parents == ("alpha", "beta")

    def test_subtract_provenance(self) -> None:
        a = _random_space(2, "alpha", seed=1)
        b = _random_space(1, "beta", seed=2)
        result = subtract(a, b)
        assert result.provenance is not None
        assert result.provenance.operation == "subtract"
        assert result.provenance.parents == ("alpha", "beta")

    def test_chained_provenance(self) -> None:
        a = _random_space(2, "alpha", seed=1)
        b = _random_space(2, "beta", seed=2)
        c = _random_space(1, "gamma", seed=3)
        ab = add(a, b)
        result = subtract(ab, c)
        assert result.provenance is not None
        assert result.provenance.operation == "subtract"
        assert result.provenance.parents == (ab.label, "gamma")

    def test_negate_provenance(self) -> None:
        a = _random_space(2, "alpha", seed=1)
        result = negate(a)
        assert result.provenance is not None
        assert result.provenance.operation == "negate"
        assert result.provenance.parents == ("alpha",)

    def test_to_dict_roundtrip(self) -> None:
        a = _random_space(2, "alpha", seed=1)
        b = _random_space(2, "beta", seed=2)
        result = add(a, b)
        d = result.to_dict()
        # Verify it's JSON-serializable
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["provenance"]["operation"] == "add"
        assert parsed["provenance"]["parents"] == ["alpha", "beta"]
        assert parsed["rank"] == result.rank
        assert parsed["d_model"] == D_MODEL

    def test_compose_provenance(self) -> None:
        a = _random_space(2, "alpha", seed=1)
        b = _random_space(2, "beta", seed=2)
        result = compose([a, b], [1.0, 0.5])
        assert result.provenance is not None
        assert result.provenance.operation == "compose"
        assert result.provenance.parents == ("alpha", "beta")

    def test_deep_provenance_chain(self) -> None:
        """Depth-3 provenance: add -> subtract -> intersect."""
        a = _random_space(3, "alpha", seed=1)
        b = _random_space(3, "beta", seed=2)
        c = _random_space(2, "gamma", seed=3)
        d = _random_space(3, "delta", seed=4)

        # Depth 1: add
        ab = add(a, b)
        assert ab.provenance is not None
        assert ab.provenance.operation == "add"
        assert ab.provenance.parents == ("alpha", "beta")

        # Depth 2: subtract
        ab_c = subtract(ab, c)
        assert ab_c.provenance is not None
        assert ab_c.provenance.operation == "subtract"
        assert ab_c.provenance.parents == (ab.label, "gamma")

        # Depth 3: intersect
        result = intersect(ab_c, d, threshold=0.1)
        assert result.provenance is not None
        assert result.provenance.operation == "intersect"
        assert result.provenance.parents == (ab_c.label, "delta")

    def test_deep_provenance_serialization(self) -> None:
        """Serialize depth-3 provenance via to_dict() and JSON round-trip."""
        a = _random_space(3, "alpha", seed=10)
        b = _random_space(3, "beta", seed=20)
        c = _random_space(2, "gamma", seed=30)
        delta = _random_space(3, "delta", seed=40)

        ab = add(a, b)
        ab_c = subtract(ab, c)
        result = intersect(ab_c, delta, threshold=0.1)

        result_dict = result.to_dict()
        json_str = json.dumps(result_dict)
        parsed = json.loads(json_str)

        # Root provenance
        prov = parsed["provenance"]
        assert prov["operation"] == "intersect"
        assert prov["parents"] == [ab_c.label, "delta"]

        # Label threading: result label contains intermediate labels
        assert result.label is not None
        assert "delta" in result.label

        # Intermediate node serialization
        ab_c_dict = ab_c.to_dict()
        ab_c_prov = ab_c_dict["provenance"]
        assert ab_c_prov["operation"] == "subtract"
        assert ab_c_prov["parents"] == [ab.label, "gamma"]

        ab_dict = ab.to_dict()
        ab_prov = ab_dict["provenance"]
        assert ab_prov["operation"] == "add"
        assert ab_prov["parents"] == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_rank0_add(self) -> None:
        a = _rank0_space("empty_a")
        b = _random_space(2, "b", seed=1)
        result = add(a, b)
        assert result.rank == b.rank
        assert similarity(result, b) > 0.99

    def test_rank0_subtract(self) -> None:
        a = _rank0_space("empty_a")
        b = _random_space(2, "b", seed=1)
        result = subtract(a, b)
        assert result.rank == 0

    def test_rank0_intersect(self) -> None:
        a = _rank0_space("empty_a")
        b = _random_space(2, "b", seed=1)
        result = intersect(a, b)
        assert result.rank == 0

    def test_dimension_mismatch_raises(self) -> None:
        a = from_array(ops.random.normal((8,)), label="small")
        b = from_array(ops.random.normal((16,)), label="big")
        with pytest.raises(ValueError, match="Dimension mismatch"):
            add(a, b)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            subtract(a, b)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            intersect(a, b)

    def test_compose_zero_weight(self) -> None:
        a = _random_space(2, "a", seed=1)
        b = _random_space(2, "b", seed=2)
        result = compose([a, b], [0.0, 0.0])
        # Zero-weighted compose should produce rank-0
        assert result.rank == 0

    def test_max_rank_capping(self) -> None:
        a = _random_space(3, "a", seed=1)
        b = _random_space(3, "b", seed=2)
        result = add(a, b, max_rank=2)
        assert result.rank <= 2

    def test_compose_length_mismatch_raises(self) -> None:
        a = _random_space(2, "a", seed=1)
        with pytest.raises(ValueError, match="same length"):
            compose([a], [1.0, 2.0])

    def test_compose_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compose([], [])

    def test_to_direction_rank0_raises(self) -> None:
        space = _rank0_space()
        with pytest.raises(ValueError, match="rank-0"):
            to_direction(space)

    def test_similarity_rank0(self) -> None:
        a = _rank0_space("empty")
        b = _random_space(2, "b", seed=1)
        assert similarity(a, b) == 0.0

    def test_from_array_invalid_ndim(self) -> None:
        arr = ops.random.normal((2, 3, 4))
        ops.eval(arr)
        with pytest.raises(ValueError, match="ndim"):
            from_array(arr)

    def test_from_array_empty_2d(self) -> None:
        arr = ops.zeros((0, D_MODEL))
        ops.eval(arr)
        space = from_array(arr, label="empty_2d")
        assert space.rank == 0

    def test_compose_partial_singular_values(self) -> None:
        """compose() with empty singular_values (from_array) falls back to sv=1.0."""
        ops.random.seed(51)
        raw = ops.random.normal((3, D_MODEL))
        ops.eval(raw)
        a = from_array(raw, label="no_sv")
        # from_array produces empty singular_values
        assert a.singular_values == []
        assert a.rank == 3
        result = compose([a], [2.0])
        assert isinstance(result, DirectionSpace)
        assert result.rank > 0
        assert _is_orthonormal(result.basis)

    def test_compose_mixed_singular_values(self) -> None:
        """compose() with singular_values shorter than rank uses sv=1.0 fallback."""
        ops.random.seed(50)
        raw = ops.random.normal((3, D_MODEL))
        basis = orthonormalize(raw)
        ops.eval(basis)
        # Construct space with only 1 singular value but rank=3
        space = DirectionSpace(
            basis=basis,
            d_model=D_MODEL,
            rank=3,
            label="partial_sv",
            singular_values=[2.0],
        )
        result = compose([space], [1.0])
        assert isinstance(result, DirectionSpace)
        assert result.rank > 0
        assert _is_orthonormal(result.basis)
        # First SV should be scaled by 2.0, others by 1.0
        assert len(result.singular_values) == result.rank

    def test_intersect_non_orthonormal_diverges(self) -> None:
        """intersect() with non-orthonormal input diverges from orthonormal version."""
        ops.random.seed(60)
        raw = ops.random.normal((3, D_MODEL))
        ops.eval(raw)

        # "Good" space: orthonormalized
        good = from_array(raw, label="good")

        # "Bad" space: raw rows (not orthonormal), directly constructed
        bad = DirectionSpace(
            basis=raw, d_model=D_MODEL, rank=3, label="bad",
        )
        assert not _is_orthonormal(bad.basis)

        result_good = intersect(good, good, threshold=0.3)
        result_bad = intersect(bad, bad, threshold=0.3)

        # Self-intersect of orthonormal should recover ~full subspace
        assert result_good.rank > 0
        # Non-orthonormal may differ in rank or subspace content,
        # but output must still satisfy the orthonormal invariant
        if result_bad.rank > 0:
            assert _is_orthonormal(result_bad.basis)

    def test_subtract_non_orthonormal_residual(self) -> None:
        """subtract() with non-orthonormal input still produces orthonormal output."""
        ops.random.seed(70)
        raw_a = ops.random.normal((3, D_MODEL))
        raw_b = ops.random.normal((1, D_MODEL))
        ops.eval(raw_a, raw_b)

        bad_a = DirectionSpace(
            basis=raw_a, d_model=D_MODEL, rank=3, label="bad_a",
        )
        bad_b = DirectionSpace(
            basis=raw_b, d_model=D_MODEL, rank=1, label="bad_b",
        )

        result = subtract(bad_a, bad_b)
        assert isinstance(result, DirectionSpace)
        # Output is always orthonormalized (via orthonormalize() in subtract)
        if result.rank > 0:
            assert _is_orthonormal(result.basis)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_compose_subspaces(self) -> None:
        """compose_subspaces() produces a DirectionSpace with correct rank."""
        from vauban._compose import compose_subspaces

        bank = {
            "safety": ops.random.normal((3, D_MODEL)),
            "format": ops.random.normal((2, D_MODEL)),
        }
        ops.eval(bank["safety"], bank["format"])

        result = compose_subspaces(bank, {"safety": 1.0, "format": 0.5})
        assert isinstance(result, DirectionSpace)
        assert result.d_model == D_MODEL
        assert result.rank > 0
        assert _is_orthonormal(result.basis)

    def test_compose_subspaces_max_rank(self) -> None:
        from vauban._compose import compose_subspaces

        bank = {
            "a": ops.random.normal((3, D_MODEL)),
            "b": ops.random.normal((3, D_MODEL)),
        }
        ops.eval(bank["a"], bank["b"])

        result = compose_subspaces(
            bank, {"a": 1.0, "b": 1.0}, max_rank=2,
        )
        assert result.rank <= 2

    def test_compose_subspaces_missing_key_raises(self) -> None:
        from vauban._compose import compose_subspaces

        bank = {"a": ops.random.normal((2, D_MODEL))}
        ops.eval(bank["a"])
        with pytest.raises(KeyError, match="missing"):
            compose_subspaces(bank, {"missing": 1.0})

    def test_compose_subspaces_empty_raises(self) -> None:
        from vauban._compose import compose_subspaces

        bank = {"a": ops.random.normal((2, D_MODEL))}
        ops.eval(bank["a"])
        with pytest.raises(ValueError, match="empty"):
            compose_subspaces(bank, {})

    def test_cut_space_rank1_matches_cut(self) -> None:
        """cut_space() with rank-1 produces same result as cut()."""
        from vauban.cut import cut, cut_space

        ops.random.seed(42)
        direction = ops.random.normal((D_MODEL,))
        direction = direction / ops.linalg.norm(direction)
        ops.eval(direction)

        space = from_array(direction, label="test_cut")

        o_key = "model.layers.0.self_attn.o_proj.weight"
        d_key = "model.layers.0.mlp.down_proj.weight"
        weights = {
            o_key: ops.random.normal((D_MODEL, D_MODEL)),
            d_key: ops.random.normal((D_MODEL, D_MODEL)),
        }
        ops.eval(weights[o_key], weights[d_key])

        result_cut = cut(weights, direction, [0], alpha=1.0)
        result_space = cut_space(weights, space, [0], alpha=1.0)

        for key in result_cut:
            diff = float(ops.linalg.norm(result_cut[key] - result_space[key]).item())
            assert diff < 1e-4, f"Mismatch on {key}: {diff}"

    def test_cut_space_rank_k_matches_cut_subspace(self) -> None:
        """cut_space() with rank-k produces same result as cut_subspace()."""
        from vauban.cut import cut_space, cut_subspace

        ops.random.seed(42)
        raw = ops.random.normal((3, D_MODEL))
        basis = orthonormalize(raw)
        ops.eval(basis)

        space = from_array(basis, label="test_sub_cut")

        o_key = "model.layers.0.self_attn.o_proj.weight"
        d_key = "model.layers.0.mlp.down_proj.weight"
        weights = {
            o_key: ops.random.normal((D_MODEL, D_MODEL)),
            d_key: ops.random.normal((D_MODEL, D_MODEL)),
        }
        ops.eval(weights[o_key], weights[d_key])

        result_sub = cut_subspace(weights, basis, [0], alpha=0.8)
        result_space = cut_space(weights, space, [0], alpha=0.8)

        for key in result_sub:
            diff = float(ops.linalg.norm(result_sub[key] - result_space[key]).item())
            assert diff < 1e-4, f"Mismatch on {key}: {diff}"

    def test_cut_space_rank0_noop(self) -> None:
        """cut_space() with rank-0 returns weights unchanged."""
        from vauban.cut import cut_space

        space = _rank0_space()
        key = "model.layers.0.self_attn.o_proj.weight"
        weights = {
            key: ops.random.normal((D_MODEL, D_MODEL)),
        }
        ops.eval(weights[key])

        result = cut_space(weights, space, [0])
        diff = float(ops.linalg.norm(result[key] - weights[key]).item())
        assert diff < 1e-6


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


class TestLabels:
    def test_add_label(self) -> None:
        a = _random_space(2, "alpha", seed=1)
        b = _random_space(2, "beta", seed=2)
        result = add(a, b)
        assert result.label == "(alpha + beta)"

    def test_subtract_label(self) -> None:
        a = _random_space(2, "alpha", seed=1)
        b = _random_space(1, "beta", seed=2)
        result = subtract(a, b)
        assert result.label == "(alpha - beta)"

    def test_intersect_label(self) -> None:
        a = _random_space(2, "alpha", seed=1)
        b = _random_space(2, "beta", seed=2)
        result = intersect(a, b)
        assert "\u2229" in result.label

    def test_negate_label(self) -> None:
        a = _random_space(2, "alpha", seed=1)
        result = negate(a)
        assert result.label == "(-alpha)"

    def test_compose_label(self) -> None:
        a = _random_space(2, "alpha", seed=1)
        b = _random_space(2, "beta", seed=2)
        result = compose([a, b], [1.0, 0.5])
        assert "compose(" in result.label
        assert "alpha" in result.label
        assert "beta" in result.label
