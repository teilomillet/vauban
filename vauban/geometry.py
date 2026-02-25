"""Geometric analysis of multiple extracted directions."""

from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True, slots=True)
class DirectionPair:
    """Pairwise relationship between two named directions."""

    name_a: str
    name_b: str
    cosine_similarity: float
    shared_variance: float  # cos^2
    independent: bool  # True if shared_variance < threshold


@dataclass(frozen=True, slots=True)
class DirectionGeometryResult:
    """Geometric relationships between multiple directions."""

    direction_names: list[str]
    pairwise: list[DirectionPair]
    cosine_matrix: list[list[float]]  # N x N matrix as nested lists
    mean_independence: float
    most_aligned_pair: DirectionPair | None
    most_orthogonal_pair: DirectionPair | None

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dict."""
        return {
            "direction_names": self.direction_names,
            "pairwise": [
                {
                    "name_a": p.name_a,
                    "name_b": p.name_b,
                    "cosine_similarity": p.cosine_similarity,
                    "shared_variance": p.shared_variance,
                    "independent": p.independent,
                }
                for p in self.pairwise
            ],
            "cosine_matrix": self.cosine_matrix,
            "mean_independence": self.mean_independence,
            "most_aligned_pair": (
                {
                    "name_a": self.most_aligned_pair.name_a,
                    "name_b": self.most_aligned_pair.name_b,
                    "cosine_similarity": self.most_aligned_pair.cosine_similarity,
                }
                if self.most_aligned_pair is not None
                else None
            ),
            "most_orthogonal_pair": (
                {
                    "name_a": self.most_orthogonal_pair.name_a,
                    "name_b": self.most_orthogonal_pair.name_b,
                    "cosine_similarity": self.most_orthogonal_pair.cosine_similarity,
                }
                if self.most_orthogonal_pair is not None
                else None
            ),
        }


def analyze_directions(
    directions: dict[str, mx.array],
    independence_threshold: float = 0.1,
) -> DirectionGeometryResult:
    """Analyze geometric relationships between multiple directions.

    Computes pairwise cosine similarities, shared variance (cos^2),
    and independence flags for all direction pairs.

    Args:
        directions: Mapping from direction name to unit-length direction vector.
        independence_threshold: Shared variance below this marks a pair as
            independent (approximately orthogonal).

    Returns:
        A DirectionGeometryResult with all pairwise analysis.
    """
    names = sorted(directions.keys())
    n = len(names)

    # Build cosine matrix
    cosine_matrix: list[list[float]] = []
    for i in range(n):
        row: list[float] = []
        for j in range(n):
            if i == j:
                row.append(1.0)
            else:
                di = directions[names[i]]
                dj = directions[names[j]]
                norm_i = mx.linalg.norm(di)
                norm_j = mx.linalg.norm(dj)
                cos = mx.sum(di * dj) / (norm_i * norm_j + 1e-8)
                mx.eval(cos)
                row.append(float(cos.item()))
        cosine_matrix.append(row)

    # Build pairwise list (upper triangle only)
    pairwise: list[DirectionPair] = []
    for i in range(n):
        for j in range(i + 1, n):
            cos_val = cosine_matrix[i][j]
            shared_var = cos_val * cos_val
            pairwise.append(
                DirectionPair(
                    name_a=names[i],
                    name_b=names[j],
                    cosine_similarity=cos_val,
                    shared_variance=shared_var,
                    independent=shared_var < independence_threshold,
                ),
            )

    # Compute mean independence (fraction of pairs that are independent)
    mean_independence = (
        sum(1.0 for p in pairwise if p.independent) / len(pairwise)
        if pairwise
        else 0.0
    )

    # Find most aligned and most orthogonal pairs
    most_aligned: DirectionPair | None = None
    most_orthogonal: DirectionPair | None = None
    if pairwise:
        most_aligned = max(pairwise, key=lambda p: abs(p.cosine_similarity))
        most_orthogonal = min(pairwise, key=lambda p: abs(p.cosine_similarity))

    return DirectionGeometryResult(
        direction_names=names,
        pairwise=pairwise,
        cosine_matrix=cosine_matrix,
        mean_independence=mean_independence,
        most_aligned_pair=most_aligned,
        most_orthogonal_pair=most_orthogonal,
    )
