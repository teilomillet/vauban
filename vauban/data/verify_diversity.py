"""Semantic diversity verification for prompt datasets.

Uses a sentence-transformer model to embed all prompts, then computes
pairwise cosine similarity to verify that prompts are semantically diverse
and non-redundant. Produces a diversity report with per-category and
cross-set statistics.

Usage::

    uv run --with sentence-transformers \
        python -m vauban.data.verify_diversity [--threshold 0.85]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

type DuplicatePair = tuple[int, int, float]
type DiversityMetric = float | int | list[DuplicatePair]
type DiversityStats = dict[str, DiversityMetric]
type SerializedDiversityMetric = float | int | list[list[int | float]]
type SerializedDiversityStats = dict[str, SerializedDiversityMetric]


def load_jsonl(path: Path) -> list[dict[str, str]]:
    """Load a JSONL file into a list of dicts."""
    entries: list[dict[str, str]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = embeddings / norms
    return normalized @ normalized.T


def report_pairwise_stats(
    name: str,
    prompts: list[str],
    embeddings: np.ndarray,
    threshold: float,
) -> DiversityStats:
    """Compute and report pairwise similarity statistics for a prompt set."""
    sim = cosine_similarity_matrix(embeddings)
    n = len(prompts)

    # Extract upper triangle (exclude self-similarity)
    upper_idx = np.triu_indices(n, k=1)
    pairwise = sim[upper_idx]

    stats: DiversityStats = {
        "n_prompts": n,
        "mean_similarity": float(np.mean(pairwise)),
        "median_similarity": float(np.median(pairwise)),
        "min_similarity": float(np.min(pairwise)),
        "max_similarity": float(np.max(pairwise)),
        "std_similarity": float(np.std(pairwise)),
    }

    # Flag near-duplicates
    near_dupes: list[DuplicatePair] = []
    dup_i, dup_j = np.where(sim > threshold)
    for i, j in zip(dup_i, dup_j, strict=True):
        if i < j:
            near_dupes.append((int(i), int(j), float(sim[i, j])))
    near_dupes.sort(key=lambda x: -x[2])
    stats["n_near_duplicates"] = len(near_dupes)
    stats["near_duplicates"] = near_dupes[:20]  # top 20

    print(f"\n{'='*60}")
    print(f"  {name} (n={n})")
    print(f"{'='*60}")
    print(f"  Mean similarity:   {stats['mean_similarity']:.4f}")
    print(f"  Median similarity: {stats['median_similarity']:.4f}")
    print(f"  Min similarity:    {stats['min_similarity']:.4f}")
    print(f"  Max similarity:    {stats['max_similarity']:.4f}")
    print(f"  Std deviation:     {stats['std_similarity']:.4f}")
    print(f"  Near-duplicates (>{threshold}): {len(near_dupes)}")

    if near_dupes:
        print("\n  Top near-duplicates:")
        for i, j, s in near_dupes[:10]:
            p_i = (
                prompts[i][:60] + "..."
                if len(prompts[i]) > 60
                else prompts[i]
            )
            p_j = (
                prompts[j][:60] + "..."
                if len(prompts[j]) > 60
                else prompts[j]
            )
            print(f"    [{i:3d}] vs [{j:3d}] sim={s:.4f}")
            print(f"          {p_i}")
            print(f"          {p_j}")

    return stats


def report_category_stats(
    name: str,
    entries: list[dict[str, str]],
    embeddings: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute per-category intra-similarity and inter-category separation."""
    categories: dict[str, list[int]] = {}
    for i, entry in enumerate(entries):
        cat = entry.get("category", "unknown")
        categories.setdefault(cat, []).append(i)

    print(f"\n  Per-category analysis ({name}):")
    print(f"  {'Category':<20} {'N':>4} {'Intra-sim':>10} {'Spread':>8}")
    print(f"  {'-'*44}")

    cat_stats: dict[str, dict[str, float]] = {}
    cat_centroids: dict[str, np.ndarray] = {}

    for cat, indices in sorted(categories.items()):
        cat_emb = embeddings[indices]
        cat_centroids[cat] = np.mean(cat_emb, axis=0)

        if len(indices) < 2:
            print(
                f"  {cat:<20} {len(indices):>4}"
                f" {'N/A':>10} {'N/A':>8}",
            )
            continue

        sim = cosine_similarity_matrix(cat_emb)
        upper_idx = np.triu_indices(len(indices), k=1)
        pairwise = sim[upper_idx]

        intra_mean = float(np.mean(pairwise))
        intra_std = float(np.std(pairwise))
        cat_stats[cat] = {
            "intra_mean": intra_mean,
            "intra_std": intra_std,
            "n": len(indices),
        }

        print(
            f"  {cat:<20} {len(indices):>4}"
            f" {intra_mean:>10.4f} {intra_std:>8.4f}",
        )

    # Inter-category centroid distances
    cat_names = sorted(cat_centroids.keys())
    if len(cat_names) > 1:
        print("\n  Inter-category centroid similarity:")
        print(f"  {'':>20}", end="")
        for c in cat_names:
            print(f" {c[:8]:>8}", end="")
        print()

        centroid_array = np.array(
            [cat_centroids[c] for c in cat_names],
        )
        centroid_sim = cosine_similarity_matrix(centroid_array)

        for i, c_i in enumerate(cat_names):
            print(f"  {c_i:<20}", end="")
            for j, _c_j in enumerate(cat_names):
                if j <= i:
                    print(f" {'':>8}", end="")
                else:
                    print(f" {centroid_sim[i, j]:>8.4f}", end="")
            print()

        # Summary: mean inter-category separation
        upper_idx = np.triu_indices(len(cat_names), k=1)
        inter_sims = centroid_sim[upper_idx]
        print(
            "\n  Mean inter-category centroid similarity:"
            f" {np.mean(inter_sims):.4f}",
        )
        print(
            "  Min inter-category centroid similarity: "
            f" {np.min(inter_sims):.4f}",
        )

    return cat_stats


def report_cross_set(
    name_a: str,
    name_b: str,
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    prompts_a: list[str],
    prompts_b: list[str],
    threshold: float,
) -> DiversityStats:
    """Compare two prompt sets for cross-set overlap."""
    # Normalize
    norm_a = emb_a / np.maximum(
        np.linalg.norm(emb_a, axis=1, keepdims=True), 1e-10,
    )
    norm_b = emb_b / np.maximum(
        np.linalg.norm(emb_b, axis=1, keepdims=True), 1e-10,
    )

    cross_sim = norm_a @ norm_b.T  # (n_a, n_b)

    print(f"\n{'='*60}")
    print(f"  Cross-set: {name_a} vs {name_b}")
    print(f"{'='*60}")
    print(f"  Mean cross-similarity: {np.mean(cross_sim):.4f}")
    print(f"  Max cross-similarity:  {np.max(cross_sim):.4f}")

    # Find cross-set near-duplicates
    cross_dupes: list[tuple[int, int, float]] = []
    dup_i, dup_j = np.where(cross_sim > threshold)
    for i, j in zip(dup_i, dup_j, strict=True):
        cross_dupes.append(
            (int(i), int(j), float(cross_sim[i, j])),
        )
    cross_dupes.sort(key=lambda x: -x[2])

    print(
        f"  Cross near-duplicates (>{threshold}):"
        f" {len(cross_dupes)}",
    )
    if cross_dupes:
        print("\n  Top cross-set matches:")
        for i, j, s in cross_dupes[:10]:
            p_i = (
                prompts_a[i][:55] + "..."
                if len(prompts_a[i]) > 55
                else prompts_a[i]
            )
            p_j = (
                prompts_b[j][:55] + "..."
                if len(prompts_b[j]) > 55
                else prompts_b[j]
            )
            print(
                f"    [{name_a}:{i:3d}] vs"
                f" [{name_b}:{j:3d}] sim={s:.4f}",
            )
            print(f"      A: {p_i}")
            print(f"      B: {p_j}")

    return {
        "mean_cross_similarity": float(np.mean(cross_sim)),
        "max_cross_similarity": float(np.max(cross_sim)),
        "n_cross_duplicates": len(cross_dupes),
    }


def main() -> None:
    """Run diversity verification on all prompt datasets."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed.")
        print(
            "Run: uv run --with sentence-transformers"
            " python -m vauban.data.verify_diversity",
        )
        sys.exit(1)

    import argparse

    parser = argparse.ArgumentParser(
        description="Verify semantic diversity of prompt datasets",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.85,
        help="Cosine similarity threshold for near-duplicates",
    )
    parser.add_argument(
        "--model", type=str, default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON report",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).parent

    # Define datasets to check
    datasets: dict[str, Path] = {
        "harmful_original": data_dir / "harmful.jsonl",
        "harmless_original": data_dir / "harmless.jsonl",
        "harmful_100": data_dir / "harmful_100.jsonl",
        "harmless_100": data_dir / "harmless_100.jsonl",
        "harmful_infix_100": data_dir / "harmful_infix_100.jsonl",
        "harmful_infix_original": data_dir / "harmful_infix.jsonl",
        "surface": data_dir / "surface.jsonl",
    }

    # Load all datasets
    loaded: dict[str, list[dict[str, str]]] = {}
    for name, path in datasets.items():
        if path.exists():
            loaded[name] = load_jsonl(path)
            print(
                f"Loaded {name}: {len(loaded[name])} prompts"
                f" from {path.name}",
            )
        else:
            print(f"Skipping {name}: {path} not found")

    if not loaded:
        print("No datasets found.")
        sys.exit(1)

    # Load model
    print(f"\nLoading sentence-transformer model: {args.model}")
    model = SentenceTransformer(args.model)

    # Embed all prompts
    all_embeddings: dict[str, np.ndarray] = {}
    all_prompts: dict[str, list[str]] = {}

    for name, entries in loaded.items():
        prompts = [e["prompt"] for e in entries]
        all_prompts[name] = prompts
        print(
            f"Embedding {name} ({len(prompts)} prompts)...",
            end=" ", flush=True,
        )
        emb = model.encode(
            prompts, show_progress_bar=False, convert_to_numpy=True,
        )
        all_embeddings[name] = emb
        print("done.")

    report: dict[str, DiversityStats] = {}

    # Per-set diversity analysis
    for name in loaded:
        stats = report_pairwise_stats(
            name,
            all_prompts[name],
            all_embeddings[name],
            args.threshold,
        )
        report[name] = stats

        # Category analysis for datasets with categories
        if any("category" in e for e in loaded[name]):
            report_category_stats(
                name, loaded[name], all_embeddings[name],
            )

    # Cross-set comparisons (check overlap between old and new)
    cross_pairs = [
        ("harmful_original", "harmful_100"),
        ("harmless_original", "harmless_100"),
        ("harmful_100", "harmless_100"),
        ("harmful_infix_original", "harmful_infix_100"),
        ("harmful_100", "harmful_infix_100"),
    ]

    for name_a, name_b in cross_pairs:
        if name_a in all_embeddings and name_b in all_embeddings:
            cross_stats = report_cross_set(
                name_a, name_b,
                all_embeddings[name_a],
                all_embeddings[name_b],
                all_prompts[name_a],
                all_prompts[name_b],
                args.threshold,
            )
            report[f"cross_{name_a}_vs_{name_b}"] = cross_stats

    # Summary verdict
    print(f"\n{'='*60}")
    print("  SUMMARY VERDICT")
    print(f"{'='*60}")

    for name in loaded:
        stats = report[name]
        n_dupes = stats.get("n_near_duplicates", 0)
        mean_sim = stats.get("mean_similarity", 0.0)
        if n_dupes == 0 and isinstance(mean_sim, float) and mean_sim < 0.7:
            verdict = "PASS"
        elif isinstance(n_dupes, int) and n_dupes < 5:
            verdict = "REVIEW"
        else:
            verdict = "FAIL"
        print(
            f"  {name:<25} mean={mean_sim:.3f}"
            f"  dupes={n_dupes:>3}  [{verdict}]",
        )

    # Save report
    if args.output:
        output_path = Path(args.output)

        def _serialize(
            obj: DiversityStats,
        ) -> SerializedDiversityStats:
            """Convert tuples to lists for JSON serialization."""
            result: SerializedDiversityStats = {}
            for k, v in obj.items():
                if isinstance(v, list):
                    result[k] = [
                        [int(i), int(j), float(score)]
                        for i, j, score in v
                    ]
                else:
                    result[k] = v
            return result

        serializable: dict[str, SerializedDiversityStats] = {
            k: _serialize(v) for k, v in report.items()
        }
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()
