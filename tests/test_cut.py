"""Tests for vauban.cut: orthogonalization math (pure linear algebra)."""

import mlx.core as mx

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
        mx.random.seed(42)
        w = mx.random.normal((8, 16))
        d = mx.random.normal((8,))
        d = d / mx.linalg.norm(d)
        mx.eval(w, d)

        w_new = _orthogonalize_matrix(w, d, alpha=1.0)
        mx.eval(w_new)

        # After orthogonalization, d^T @ W' should be ~0
        proj = d @ w_new
        mx.eval(proj)
        for i in range(16):
            assert abs(float(proj[i].item())) < 1e-5

    def test_alpha_scaling(self) -> None:
        mx.random.seed(42)
        w = mx.random.normal((4, 8))
        d = mx.random.normal((4,))
        d = d / mx.linalg.norm(d)
        mx.eval(w, d)

        w_half = _orthogonalize_matrix(w, d, alpha=0.5)
        w_full = _orthogonalize_matrix(w, d, alpha=1.0)
        mx.eval(w_half, w_full)

        # Half alpha should leave more of the original projection
        proj_half = mx.linalg.norm(d @ w_half)
        proj_full = mx.linalg.norm(d @ w_full)
        mx.eval(proj_half, proj_full)
        assert float(proj_half.item()) > float(proj_full.item())


class TestOrthogonalizeNormPreserve:
    def test_preserves_row_norms(self) -> None:
        mx.random.seed(42)
        w = mx.random.normal((8, 16))
        d = mx.random.normal((8,))
        d = d / mx.linalg.norm(d)
        mx.eval(w, d)

        original_norms = mx.linalg.norm(w, axis=1)
        w_new = _orthogonalize_norm_preserve(w, d, alpha=1.0)
        new_norms = mx.linalg.norm(w_new, axis=1)
        mx.eval(original_norms, new_norms)

        for i in range(8):
            orig = float(original_norms[i].item())
            new = float(new_norms[i].item())
            assert abs(orig - new) < 1e-4


class TestBiprojectedDirection:
    def test_orthogonal_to_harmless(self) -> None:
        mx.random.seed(42)
        refusal = mx.random.normal((16,))
        refusal = refusal / mx.linalg.norm(refusal)
        harmless = mx.random.normal((16,))
        harmless = harmless / mx.linalg.norm(harmless)
        mx.eval(refusal, harmless)

        result = _biprojected_direction(refusal, harmless)
        mx.eval(result)

        dot = float(mx.sum(result * harmless).item())
        assert abs(dot) < 1e-5

    def test_is_unit_vector(self) -> None:
        mx.random.seed(42)
        refusal = mx.random.normal((16,))
        refusal = refusal / mx.linalg.norm(refusal)
        harmless = mx.random.normal((16,))
        harmless = harmless / mx.linalg.norm(harmless)
        mx.eval(refusal, harmless)

        result = _biprojected_direction(refusal, harmless)
        norm = float(mx.linalg.norm(result).item())
        assert abs(norm - 1.0) < 1e-4


class TestOrthogonalizeMatrix3D:
    """Tests for batched (MoE expert) 3D weight matrices."""

    def test_removes_direction_from_all_experts(self) -> None:
        mx.random.seed(42)
        num_experts = 4
        d_model = 8
        features = 16
        w = mx.random.normal((num_experts, d_model, features))
        d = mx.random.normal((d_model,))
        d = d / mx.linalg.norm(d)
        mx.eval(w, d)

        w_new = _orthogonalize_matrix(w, d, alpha=1.0)
        mx.eval(w_new)

        # d^T @ W'[e] should be ~0 for each expert
        for e in range(num_experts):
            proj = d @ w_new[e]
            mx.eval(proj)
            for i in range(features):
                assert abs(float(proj[i].item())) < 1e-5

    def test_norm_preserve_3d(self) -> None:
        mx.random.seed(42)
        w = mx.random.normal((4, 8, 16))
        d = mx.random.normal((8,))
        d = d / mx.linalg.norm(d)
        mx.eval(w, d)

        original_norms = mx.linalg.norm(w, axis=-1)
        w_new = _orthogonalize_norm_preserve(w, d, alpha=1.0)
        new_norms = mx.linalg.norm(w_new, axis=-1)
        mx.eval(original_norms, new_norms)

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
        mx.random.seed(42)
        d = 16
        o_key = "model.layers.0.self_attn.o_proj.weight"
        d_key = "model.layers.0.mlp.down_proj.weight"
        q_key = "model.layers.0.self_attn.q_proj.weight"
        weights = {
            o_key: mx.random.normal((d, d)),
            d_key: mx.random.normal((d, d * 2)),
            q_key: mx.random.normal((d, d)),
        }
        direction = mx.random.normal((d,))
        direction = direction / mx.linalg.norm(direction)
        mx.eval(direction)
        for v in weights.values():
            mx.eval(v)

        result = cut(weights, direction, target_layers=[0])

        # o_proj and down_proj should be different
        assert not mx.array_equal(
            result[o_key], weights[o_key],
        )
        # q_proj should be unchanged (same object)
        assert result[q_key] is weights[q_key]


class TestCutSubspace:
    def test_removes_subspace_components(self) -> None:
        mx.random.seed(42)
        d = 16
        o_key = "model.layers.0.self_attn.o_proj.weight"
        weights = {o_key: mx.random.normal((d, d))}
        mx.eval(weights[o_key])

        basis = orthonormalize(mx.random.normal((3, d)))
        mx.eval(basis)

        result = cut_subspace(weights, basis, target_layers=[0])

        # After subspace cut, each basis direction should be ~removed
        w_new = result[o_key]
        for i in range(3):
            proj = basis[i] @ w_new
            mx.eval(proj)
            assert float(mx.linalg.norm(proj).item()) < 1e-4

    def test_norm_preserve(self) -> None:
        mx.random.seed(42)
        d = 16
        o_key = "model.layers.0.self_attn.o_proj.weight"
        weights = {o_key: mx.random.normal((d, d))}
        mx.eval(weights[o_key])

        basis = orthonormalize(mx.random.normal((2, d)))
        mx.eval(basis)

        original_norms = mx.linalg.norm(weights[o_key], axis=1)
        result = cut_subspace(
            weights, basis, target_layers=[0], norm_preserve=True,
        )
        new_norms = mx.linalg.norm(result[o_key], axis=1)
        mx.eval(original_norms, new_norms)

        for i in range(d):
            orig = float(original_norms[i].item())
            new = float(new_norms[i].item())
            assert abs(orig - new) < 1e-3


class TestCutFalseRefusalOrtho:
    def test_effective_direction_orthogonal_to_false_refusal(self) -> None:
        mx.random.seed(42)
        d = 16
        o_key = "model.layers.0.self_attn.o_proj.weight"
        weights = {o_key: mx.random.normal((d, d))}

        refusal = mx.random.normal((d,))
        refusal = refusal / mx.linalg.norm(refusal)
        false_refusal = mx.random.normal((d,))
        false_refusal = false_refusal / mx.linalg.norm(false_refusal)
        mx.eval(weights[o_key], refusal, false_refusal)

        cut_false_refusal_ortho(
            weights, refusal, false_refusal, target_layers=[0],
        )

        # Verify the effective cut direction is orthogonal to false_refusal
        ortho_dir = _biprojected_direction(refusal, false_refusal)
        mx.eval(ortho_dir)
        dot = float(mx.abs(mx.sum(ortho_dir * false_refusal)).item())
        assert dot < 1e-5

    def test_weights_are_modified(self) -> None:
        mx.random.seed(42)
        d = 16
        o_key = "model.layers.0.self_attn.o_proj.weight"
        weights = {o_key: mx.random.normal((d, d))}

        refusal = mx.random.normal((d,))
        refusal = refusal / mx.linalg.norm(refusal)
        false_refusal = mx.random.normal((d,))
        false_refusal = false_refusal / mx.linalg.norm(false_refusal)
        mx.eval(weights[o_key], refusal, false_refusal)

        result = cut_false_refusal_ortho(
            weights, refusal, false_refusal, target_layers=[0],
        )

        assert not mx.array_equal(result[o_key], weights[o_key])
