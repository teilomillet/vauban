# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban.cut: orthogonalization math (pure linear algebra)."""

from vauban import _ops as ops
from vauban.cut import (
    _biprojected_direction,
    _orthogonalize_matrix,
    _orthogonalize_norm_preserve,
    cut,
    cut_false_refusal_ortho,
    cut_subspace,
    target_weight_keys,
)
from vauban.subspace import orthonormalize


class TestOrthogonalizeMatrix:
    def test_removes_direction_component(self) -> None:
        ops.random.seed(42)
        w = ops.random.normal((8, 16))
        d = ops.random.normal((8,))
        d = d / ops.linalg.norm(d)
        ops.eval(w, d)

        w_new = _orthogonalize_matrix(w, d, alpha=1.0)
        ops.eval(w_new)

        # After orthogonalization, d^T @ W' should be ~0
        proj = d @ w_new
        ops.eval(proj)
        for i in range(16):
            assert abs(float(proj[i].item())) < 1e-5

    def test_alpha_scaling(self) -> None:
        ops.random.seed(42)
        w = ops.random.normal((4, 8))
        d = ops.random.normal((4,))
        d = d / ops.linalg.norm(d)
        ops.eval(w, d)

        w_half = _orthogonalize_matrix(w, d, alpha=0.5)
        w_full = _orthogonalize_matrix(w, d, alpha=1.0)
        ops.eval(w_half, w_full)

        # Half alpha should leave more of the original projection
        proj_half = ops.linalg.norm(d @ w_half)
        proj_full = ops.linalg.norm(d @ w_full)
        ops.eval(proj_half, proj_full)
        assert float(proj_half.item()) > float(proj_full.item())


class TestOrthogonalizeNormPreserve:
    def test_preserves_row_norms(self) -> None:
        ops.random.seed(42)
        w = ops.random.normal((8, 16))
        d = ops.random.normal((8,))
        d = d / ops.linalg.norm(d)
        ops.eval(w, d)

        original_norms = ops.linalg.norm(w, axis=1)
        w_new = _orthogonalize_norm_preserve(w, d, alpha=1.0)
        new_norms = ops.linalg.norm(w_new, axis=1)
        ops.eval(original_norms, new_norms)

        for i in range(8):
            orig = float(original_norms[i].item())
            new = float(new_norms[i].item())
            assert abs(orig - new) < 1e-4


class TestBiprojectedDirection:
    def test_orthogonal_to_harmless(self) -> None:
        ops.random.seed(42)
        refusal = ops.random.normal((16,))
        refusal = refusal / ops.linalg.norm(refusal)
        harmless = ops.random.normal((16,))
        harmless = harmless / ops.linalg.norm(harmless)
        ops.eval(refusal, harmless)

        result = _biprojected_direction(refusal, harmless)
        ops.eval(result)

        dot = float(ops.sum(result * harmless).item())
        assert abs(dot) < 1e-5

    def test_is_unit_vector(self) -> None:
        ops.random.seed(42)
        refusal = ops.random.normal((16,))
        refusal = refusal / ops.linalg.norm(refusal)
        harmless = ops.random.normal((16,))
        harmless = harmless / ops.linalg.norm(harmless)
        ops.eval(refusal, harmless)

        result = _biprojected_direction(refusal, harmless)
        norm = float(ops.linalg.norm(result).item())
        assert abs(norm - 1.0) < 1e-4


class TestOrthogonalizeMatrix3D:
    """Tests for batched (MoE expert) 3D weight matrices."""

    def test_removes_direction_from_all_experts(self) -> None:
        ops.random.seed(42)
        num_experts = 4
        d_model = 8
        features = 16
        w = ops.random.normal((num_experts, d_model, features))
        d = ops.random.normal((d_model,))
        d = d / ops.linalg.norm(d)
        ops.eval(w, d)

        w_new = _orthogonalize_matrix(w, d, alpha=1.0)
        ops.eval(w_new)

        # d^T @ W'[e] should be ~0 for each expert
        for e in range(num_experts):
            proj = d @ w_new[e]
            ops.eval(proj)
            for i in range(features):
                assert abs(float(proj[i].item())) < 1e-5

    def test_norm_preserve_3d(self) -> None:
        ops.random.seed(42)
        w = ops.random.normal((4, 8, 16))
        d = ops.random.normal((8,))
        d = d / ops.linalg.norm(d)
        ops.eval(w, d)

        original_norms = ops.linalg.norm(w, axis=-1)
        w_new = _orthogonalize_norm_preserve(w, d, alpha=1.0)
        new_norms = ops.linalg.norm(w_new, axis=-1)
        ops.eval(original_norms, new_norms)

        for e in range(4):
            for i in range(8):
                orig = float(original_norms[e, i].item())
                new = float(new_norms[e, i].item())
                assert abs(orig - new) < 1e-4


class TestTargetWeightKeys:
    def test_selects_correct_keys(self) -> None:
        keys = [
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.1.self_attn.o_proj.weight",
            "model.layers.1.mlp.down_proj.weight",
        ]
        result = target_weight_keys(keys, [0])
        assert result == [
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ]

    def test_selects_moe_keys(self) -> None:
        keys = [
            "model.layers.2.self_attn.o_proj.weight",
            "model.layers.2.mlp.shared_experts.down_proj.weight",
            "model.layers.2.mlp.experts.down_proj.weight",
            "model.layers.2.mlp.router.gate.weight",
        ]
        result = target_weight_keys(keys, [2])
        assert result == [
            "model.layers.2.self_attn.o_proj.weight",
            "model.layers.2.mlp.shared_experts.down_proj.weight",
            "model.layers.2.mlp.experts.down_proj.weight",
        ]

    def test_missing_keys_skipped(self) -> None:
        keys = ["model.layers.0.self_attn.o_proj.weight"]
        result = target_weight_keys(keys, [0, 5])
        assert len(result) == 1


class TestCut:
    def test_modifies_target_weights(self) -> None:
        ops.random.seed(42)
        d = 16
        o_key = "model.layers.0.self_attn.o_proj.weight"
        d_key = "model.layers.0.mlp.down_proj.weight"
        q_key = "model.layers.0.self_attn.q_proj.weight"
        weights = {
            o_key: ops.random.normal((d, d)),
            d_key: ops.random.normal((d, d * 2)),
            q_key: ops.random.normal((d, d)),
        }
        direction = ops.random.normal((d,))
        direction = direction / ops.linalg.norm(direction)
        ops.eval(direction)
        for v in weights.values():
            ops.eval(v)

        result = cut(weights, direction, target_layers=[0])

        # o_proj and down_proj should be different
        assert not ops.array_equal(
            result[o_key], weights[o_key],
        )
        # q_proj should be unchanged (same object)
        assert result[q_key] is weights[q_key]


class TestCutSubspace:
    def test_removes_subspace_components(self) -> None:
        ops.random.seed(42)
        d = 16
        o_key = "model.layers.0.self_attn.o_proj.weight"
        weights = {o_key: ops.random.normal((d, d))}
        ops.eval(weights[o_key])

        basis = orthonormalize(ops.random.normal((3, d)))
        ops.eval(basis)

        result = cut_subspace(weights, basis, target_layers=[0])

        # After subspace cut, each basis direction should be ~removed
        w_new = result[o_key]
        for i in range(3):
            proj = basis[i] @ w_new
            ops.eval(proj)
            assert float(ops.linalg.norm(proj).item()) < 1e-4

    def test_norm_preserve(self) -> None:
        ops.random.seed(42)
        d = 16
        o_key = "model.layers.0.self_attn.o_proj.weight"
        weights = {o_key: ops.random.normal((d, d))}
        ops.eval(weights[o_key])

        basis = orthonormalize(ops.random.normal((2, d)))
        ops.eval(basis)

        original_norms = ops.linalg.norm(weights[o_key], axis=1)
        result = cut_subspace(
            weights, basis, target_layers=[0], norm_preserve=True,
        )
        new_norms = ops.linalg.norm(result[o_key], axis=1)
        ops.eval(original_norms, new_norms)

        for i in range(d):
            orig = float(original_norms[i].item())
            new = float(new_norms[i].item())
            assert abs(orig - new) < 1e-3


class TestCutFalseRefusalOrtho:
    def test_effective_direction_orthogonal_to_false_refusal(self) -> None:
        ops.random.seed(42)
        d = 16
        o_key = "model.layers.0.self_attn.o_proj.weight"
        weights = {o_key: ops.random.normal((d, d))}

        refusal = ops.random.normal((d,))
        refusal = refusal / ops.linalg.norm(refusal)
        false_refusal = ops.random.normal((d,))
        false_refusal = false_refusal / ops.linalg.norm(false_refusal)
        ops.eval(weights[o_key], refusal, false_refusal)

        cut_false_refusal_ortho(
            weights, refusal, false_refusal, target_layers=[0],
        )

        # Verify the effective cut direction is orthogonal to false_refusal
        ortho_dir = _biprojected_direction(refusal, false_refusal)
        ops.eval(ortho_dir)
        dot = float(ops.abs(ops.sum(ortho_dir * false_refusal)).item())
        assert dot < 1e-5

    def test_weights_are_modified(self) -> None:
        ops.random.seed(42)
        d = 16
        o_key = "model.layers.0.self_attn.o_proj.weight"
        weights = {o_key: ops.random.normal((d, d))}

        refusal = ops.random.normal((d,))
        refusal = refusal / ops.linalg.norm(refusal)
        false_refusal = ops.random.normal((d,))
        false_refusal = false_refusal / ops.linalg.norm(false_refusal)
        ops.eval(weights[o_key], refusal, false_refusal)

        result = cut_false_refusal_ortho(
            weights, refusal, false_refusal, target_layers=[0],
        )

        assert not ops.array_equal(result[o_key], weights[o_key])


class TestDBDIBothCut:
    """Test DBDI 'both' target: sequential RED + HDD cuts."""

    def test_both_directions_applied(self) -> None:
        ops.random.seed(42)
        d = 16
        o_key = "model.layers.0.self_attn.o_proj.weight"
        weights = {o_key: ops.random.normal((d, d))}

        red = ops.random.normal((d,))
        red = red / ops.linalg.norm(red)
        hdd = ops.random.normal((d,))
        hdd = hdd / ops.linalg.norm(hdd)
        ops.eval(weights[o_key], red, hdd)

        # Apply RED cut first, then HDD cut (simulating "both" mode)
        after_red = cut(weights, red, target_layers=[0])
        after_both = cut(after_red, hdd, target_layers=[0])

        # After both cuts, HDD projection should be ~removed (last cut)
        proj_hdd = hdd @ after_both[o_key]
        ops.eval(proj_hdd)
        assert float(ops.linalg.norm(proj_hdd).item()) < 1e-4

        # The combined result should differ from just RED cut
        assert not ops.array_equal(after_both[o_key], after_red[o_key])

    def test_both_differs_from_single(self) -> None:
        ops.random.seed(42)
        d = 16
        o_key = "model.layers.0.self_attn.o_proj.weight"
        weights = {o_key: ops.random.normal((d, d))}

        red = ops.random.normal((d,))
        red = red / ops.linalg.norm(red)
        hdd = ops.random.normal((d,))
        hdd = hdd / ops.linalg.norm(hdd)
        ops.eval(weights[o_key], red, hdd)

        after_red_only = cut(weights, red, target_layers=[0])
        after_red = cut(weights, red, target_layers=[0])
        after_both = cut(after_red, hdd, target_layers=[0])

        # "both" should differ from "red only"
        assert not ops.array_equal(
            after_both[o_key], after_red_only[o_key],
        )


class TestFalseRefusalOrthoPreservesDirection:
    """Test that false-refusal ortho preserves the false-refusal direction."""

    def test_false_refusal_direction_preserved_in_weights(self) -> None:
        ops.random.seed(42)
        d = 16
        o_key = "model.layers.0.self_attn.o_proj.weight"
        weights = {o_key: ops.random.normal((d, d))}

        # Use orthogonal directions to maximize the difference
        refusal_data = [0.0] * d
        refusal_data[0] = 1.0
        refusal = ops.array(refusal_data)
        false_refusal_data = [0.0] * d
        false_refusal_data[1] = 1.0
        false_refusal = ops.array(false_refusal_data)
        ops.eval(weights[o_key], refusal, false_refusal)

        # Cut with false-refusal ortho
        result_ortho = cut_false_refusal_ortho(
            weights, refusal, false_refusal, target_layers=[0],
        )

        # The ortho direction should be orthogonal to false_refusal,
        # so the false_refusal component of weights should be fully
        # preserved. Verify: false_refusal @ W_ortho == false_refusal @ W_orig
        proj_ortho = false_refusal @ result_ortho[o_key]
        proj_orig = false_refusal @ weights[o_key]
        ops.eval(proj_ortho, proj_orig)

        diff = float(ops.linalg.norm(proj_ortho - proj_orig).item())
        assert diff < 1e-5
