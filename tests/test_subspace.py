"""Tests for vauban.subspace: pure linear algebra geometry tools."""

import math

import mlx.core as mx

from vauban.subspace import (
    effective_rank,
    explained_variance_ratio,
    grassmann_distance,
    orthonormalize,
    principal_angles,
    project_subspace,
    remove_subspace,
    subspace_overlap,
)


class TestPrincipalAngles:
    def test_identical_subspaces_give_zero_angles(self) -> None:
        mx.random.seed(42)
        u = orthonormalize(mx.random.normal((3, 16)))
        mx.eval(u)
        angles = principal_angles(u, u)
        mx.eval(angles)
        for i in range(3):
            assert abs(float(angles[i].item())) < 1e-3

    def test_orthogonal_subspaces_give_pi_over_2(self) -> None:
        # Construct two orthogonal 2D subspaces in R^8
        u = mx.zeros((2, 8))
        u[0, 0] = 1.0
        u[0, 1] = 0.0
        u[1, 0] = 0.0
        u[1, 1] = 1.0
        v = mx.zeros((2, 8))
        v[0, 2] = 1.0
        v[0, 3] = 0.0
        v[1, 2] = 0.0
        v[1, 3] = 1.0
        mx.eval(u, v)
        angles = principal_angles(u, v)
        mx.eval(angles)
        for i in range(2):
            assert abs(float(angles[i].item()) - math.pi / 2) < 1e-4

    def test_symmetry(self) -> None:
        mx.random.seed(42)
        u = orthonormalize(mx.random.normal((2, 16)))
        v = orthonormalize(mx.random.normal((2, 16)))
        mx.eval(u, v)
        angles_uv = principal_angles(u, v)
        angles_vu = principal_angles(v, u)
        mx.eval(angles_uv, angles_vu)
        for i in range(2):
            assert abs(
                float(angles_uv[i].item()) - float(angles_vu[i].item()),
            ) < 1e-4


class TestGrassmannDistance:
    def test_identical_subspaces_zero_distance(self) -> None:
        mx.random.seed(42)
        u = orthonormalize(mx.random.normal((3, 16)))
        mx.eval(u)
        dist = grassmann_distance(u, u)
        assert dist < 1e-3

    def test_orthogonal_subspaces_positive_distance(self) -> None:
        u = mx.zeros((2, 8))
        u[0, 0] = 1.0
        u[1, 1] = 1.0
        v = mx.zeros((2, 8))
        v[0, 2] = 1.0
        v[1, 3] = 1.0
        mx.eval(u, v)
        dist = grassmann_distance(u, v)
        # sqrt(2 * (pi/2)^2) = pi/2 * sqrt(2) ≈ 2.22
        assert dist > 1.0


class TestSubspaceOverlap:
    def test_identical_subspaces_overlap_one(self) -> None:
        mx.random.seed(42)
        u = orthonormalize(mx.random.normal((3, 16)))
        mx.eval(u)
        overlap = subspace_overlap(u, u)
        assert abs(overlap - 1.0) < 1e-3

    def test_orthogonal_subspaces_overlap_zero(self) -> None:
        u = mx.zeros((2, 8))
        u[0, 0] = 1.0
        u[1, 1] = 1.0
        v = mx.zeros((2, 8))
        v[0, 2] = 1.0
        v[1, 3] = 1.0
        mx.eval(u, v)
        overlap = subspace_overlap(u, v)
        assert overlap < 1e-3


class TestProjectSubspace:
    def test_projection_is_idempotent(self) -> None:
        mx.random.seed(42)
        basis = orthonormalize(mx.random.normal((3, 16)))
        x = mx.random.normal((16,))
        mx.eval(basis, x)

        p1 = project_subspace(x, basis)
        p2 = project_subspace(p1, basis)
        mx.eval(p1, p2)

        diff = float(mx.linalg.norm(p1 - p2).item())
        assert diff < 1e-4

    def test_projection_in_subspace(self) -> None:
        mx.random.seed(42)
        basis = orthonormalize(mx.random.normal((2, 16)))
        x = mx.random.normal((16,))
        mx.eval(basis, x)

        proj = project_subspace(x, basis)
        # Project again — should be the same
        proj2 = project_subspace(proj, basis)
        mx.eval(proj, proj2)
        assert float(mx.linalg.norm(proj - proj2).item()) < 1e-4


class TestRemoveSubspace:
    def test_removal_orthogonal_to_basis(self) -> None:
        mx.random.seed(42)
        basis = orthonormalize(mx.random.normal((3, 16)))
        x = mx.random.normal((16,))
        mx.eval(basis, x)

        removed = remove_subspace(x, basis)
        mx.eval(removed)

        # removed should be orthogonal to each basis vector
        for i in range(3):
            dot = float(mx.sum(removed * basis[i]).item())
            assert abs(dot) < 1e-4

    def test_removal_plus_projection_equals_original(self) -> None:
        mx.random.seed(42)
        basis = orthonormalize(mx.random.normal((2, 16)))
        x = mx.random.normal((16,))
        mx.eval(basis, x)

        proj = project_subspace(x, basis)
        removed = remove_subspace(x, basis)
        reconstructed = proj + removed
        mx.eval(reconstructed)

        diff = float(mx.linalg.norm(x - reconstructed).item())
        assert diff < 1e-4


class TestOrthonormalize:
    def test_output_is_orthonormal(self) -> None:
        mx.random.seed(42)
        vectors = mx.random.normal((3, 16))
        mx.eval(vectors)

        basis = orthonormalize(vectors)
        mx.eval(basis)

        # Check orthonormality: basis @ basis^T should be identity
        gram = basis @ basis.T
        mx.eval(gram)
        identity = mx.eye(3)
        diff = float(mx.linalg.norm(gram - identity).item())
        assert diff < 1e-4

    def test_preserves_span(self) -> None:
        mx.random.seed(42)
        vectors = mx.random.normal((2, 16))
        mx.eval(vectors)

        basis = orthonormalize(vectors)
        mx.eval(basis)

        # Original vectors should be in the span of the basis
        for i in range(2):
            proj = project_subspace(vectors[i], basis)
            diff = float(mx.linalg.norm(vectors[i] - proj).item())
            assert diff < 1e-4


class TestExplainedVarianceRatio:
    def test_sums_to_one(self) -> None:
        ratios = explained_variance_ratio([3.0, 2.0, 1.0])
        assert abs(sum(ratios) - 1.0) < 1e-6

    def test_zero_values(self) -> None:
        ratios = explained_variance_ratio([0.0, 0.0])
        assert all(r == 0.0 for r in ratios)


class TestEffectiveRank:
    def test_rank_one(self) -> None:
        # Single dominant singular value
        rank = effective_rank([10.0, 0.0, 0.0])
        assert abs(rank - 1.0) < 1e-4

    def test_uniform_spectrum(self) -> None:
        # All equal singular values → effective rank = n
        rank = effective_rank([1.0, 1.0, 1.0])
        assert abs(rank - 3.0) < 1e-4
