"""Model-space diversity verification for prompt datasets.

Embeds prompts through the target model's residual stream and measures
diversity in activation space + projection onto the refusal direction.
Compares with sentence-transformer embeddings for a combined report.

Usage::

    uv run python -m vauban.data.verify_diversity_model \\
        --model mlx-community/Qwen2.5-1.5B-Instruct-bf16 \\
        --output vauban/data/model_diversity_report.json

Requires: the vauban package (uses _model_io, measure, _ops).
Optional: sentence-transformers for combined comparison.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

type ModelMetric = float | int | list[float]
type ModelSection = dict[str, ModelMetric]
type ModelReportEntry = ModelSection | str | int | float | list[float] | bool


def load_jsonl(path: Path) -> list[dict[str, str]]:
    """Load a JSONL file into a list of dicts."""
    entries: list[dict[str, str]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def cosine_similarity_matrix(
    embeddings: np.ndarray,
) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = embeddings / norms
    return normalized @ normalized.T


def extract_activations(
    model_path: str,
    prompts: list[str],
    layer_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int, list[float]]:
    """Extract model activations and refusal projections for prompts.

    Args:
        model_path: HuggingFace model ID or local path.
        prompts: List of prompt strings.
        layer_index: Specific layer to use. If None, extracts refusal
            direction first and uses the best layer.

    Returns:
        (activations, projections, best_layer, cosine_scores)
        - activations: shape (n_prompts, d_model) at best layer
        - projections: shape (n_prompts,) onto refusal direction
        - best_layer: layer index used
        - cosine_scores: per-layer separation scores
    """
    from vauban import _ops as ops
    from vauban._forward import (
        embed_and_mask,
        force_eval,
        get_transformer,
        make_ssm_mask,
        select_mask,
    )
    from vauban._model_io import load_model
    from vauban.measure import measure

    print(f"Loading model: {model_path}")
    model, tokenizer = load_model(model_path)

    # Extract refusal direction to get best layer + direction vector
    print(
        "Extracting refusal direction"
        " (using bundled harmful/harmless)...",
    )
    data_dir = Path(__file__).parent
    harmful_prompts = [
        e["prompt"]
        for e in load_jsonl(data_dir / "harmful.jsonl")
    ]
    harmless_prompts = [
        e["prompt"]
        for e in load_jsonl(data_dir / "harmless.jsonl")
    ]

    direction_result = measure(
        model=model,
        tokenizer=tokenizer,
        harmful_prompts=harmful_prompts[:64],
        harmless_prompts=harmless_prompts[:64],
    )

    best_layer = (
        layer_index
        if layer_index is not None
        else direction_result.layer_index
    )
    direction = direction_result.direction
    d_model = direction_result.d_model
    print(f"Best layer: {best_layer}, d_model: {d_model}")
    print(
        "Cosine scores:"
        f" {[f'{s:.3f}' for s in direction_result.cosine_scores]}",
    )

    # Collect per-prompt activations at best layer
    print(
        f"Collecting activations for {len(prompts)} prompts"
        f" at layer {best_layer}...",
    )
    transformer = get_transformer(model)
    activations_list: list[np.ndarray] = []
    projections_list: list[float] = []

    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False,
        )
        if not isinstance(text, str):
            msg = (
                "apply_chat_template must return str"
                " when tokenize=False"
            )
            raise TypeError(msg)
        token_ids = ops.array(tokenizer.encode(text))[None, :]

        h, mask = embed_and_mask(transformer, token_ids)
        ssm_mask = make_ssm_mask(transformer, h)
        for layer_idx, layer in enumerate(transformer.layers):
            h = layer(h, select_mask(layer, mask, ssm_mask))
            if layer_idx == best_layer:
                last_token = h[0, -1, :].astype(ops.float32)
                proj = ops.sum(last_token * direction)
                force_eval(last_token, proj)
                activations_list.append(
                    np.array(last_token.tolist()),
                )
                projections_list.append(float(proj.item()))
                break

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(prompts)} done")

    activations = np.array(activations_list)
    projections = np.array(projections_list)

    return (
        activations,
        projections,
        best_layer,
        direction_result.cosine_scores,
    )


def analyze_set(
    name: str,
    entries: list[dict[str, str]],
    activations: np.ndarray,
    projections: np.ndarray,
    threshold: float,
) -> ModelSection:
    """Analyze diversity of a prompt set in model activation space."""
    n = len(entries)
    prompts = [e["prompt"] for e in entries]

    print(f"\n{'='*60}")
    print(f"  {name} \u2014 Model Space (n={n})")
    print(f"{'='*60}")

    # Pairwise similarity in activation space
    sim = cosine_similarity_matrix(activations)
    upper_idx = np.triu_indices(n, k=1)
    pairwise = sim[upper_idx]

    stats: ModelSection = {
        "n_prompts": n,
        "act_mean_sim": float(np.mean(pairwise)),
        "act_median_sim": float(np.median(pairwise)),
        "act_min_sim": float(np.min(pairwise)),
        "act_max_sim": float(np.max(pairwise)),
        "act_std_sim": float(np.std(pairwise)),
    }

    print("  Activation similarity:")
    print(f"    Mean:   {stats['act_mean_sim']:.4f}")
    print(f"    Median: {stats['act_median_sim']:.4f}")
    print(f"    Min:    {stats['act_min_sim']:.4f}")
    print(f"    Max:    {stats['act_max_sim']:.4f}")
    print(f"    Std:    {stats['act_std_sim']:.4f}")

    # Near-duplicates in activation space
    dup_i, dup_j = np.where(sim > threshold)
    near_dupes = [
        (int(i), int(j), float(sim[i, j]))
        for i, j in zip(dup_i, dup_j, strict=True)
        if i < j
    ]
    near_dupes.sort(key=lambda x: -x[2])
    stats["act_near_duplicates"] = len(near_dupes)
    print(
        f"    Near-duplicates (>{threshold}): {len(near_dupes)}",
    )
    if near_dupes:
        for i, j, s in near_dupes[:5]:
            print(f"      [{i}] vs [{j}] sim={s:.4f}")
            print(f"        {prompts[i][:60]}...")
            print(f"        {prompts[j][:60]}...")

    # Refusal projection statistics
    stats["proj_mean"] = float(np.mean(projections))
    stats["proj_std"] = float(np.std(projections))
    stats["proj_min"] = float(np.min(projections))
    stats["proj_max"] = float(np.max(projections))
    stats["proj_range"] = float(
        np.max(projections) - np.min(projections),
    )
    stats["proj_values"] = [float(p) for p in projections]

    print("\n  Refusal projection (dot with refusal direction):")
    print(f"    Mean:  {stats['proj_mean']:+.4f}")
    print(f"    Std:   {stats['proj_std']:.4f}")
    print(
        f"    Range: [{stats['proj_min']:+.4f},"
        f" {stats['proj_max']:+.4f}]",
    )

    # Per-category breakdown
    categories: dict[str, list[int]] = {}
    for i, entry in enumerate(entries):
        cat = entry.get("category", "unknown")
        categories.setdefault(cat, []).append(i)

    if len(categories) > 1:
        print("\n  Per-category refusal projection:")
        print(
            f"  {'Category':<20} {'N':>4}"
            f" {'Proj mean':>10} {'Proj std':>9}"
            f" {'Intra-sim':>10}",
        )
        print(f"  {'-'*55}")

        cat_stats: dict[str, dict[str, float]] = {}
        for cat, indices in sorted(categories.items()):
            cat_proj = projections[indices]
            cat_act = activations[indices]

            cat_info: dict[str, float] = {
                "proj_mean": float(np.mean(cat_proj)),
                "proj_std": float(np.std(cat_proj)),
            }

            if len(indices) >= 2:
                cat_sim = cosine_similarity_matrix(cat_act)
                cat_upper = np.triu_indices(len(indices), k=1)
                cat_info["intra_sim"] = float(
                    np.mean(cat_sim[cat_upper]),
                )
            else:
                cat_info["intra_sim"] = float("nan")

            cat_stats[cat] = cat_info
            print(
                f"  {cat:<20} {len(indices):>4} "
                f"{cat_info['proj_mean']:>+10.4f} "
                f"{cat_info['proj_std']:>9.4f} "
                f"{cat_info['intra_sim']:>10.4f}",
            )

        # Category separation
        cat_means = [
            cat_stats[c]["proj_mean"]
            for c in sorted(categories)
        ]
        stats["category_proj_spread"] = float(np.std(cat_means))
        print(
            "\n  Category centroid projection spread (std):"
            f" {stats['category_proj_spread']:.4f}",
        )

    return stats


def compare_spaces(
    name: str,
    act_sim: np.ndarray,
    st_sim: np.ndarray,
    n: int,
) -> ModelSection:
    """Compare activation-space and sentence-transformer similarity."""
    upper_idx = np.triu_indices(n, k=1)
    act_pairwise = act_sim[upper_idx]
    st_pairwise = st_sim[upper_idx]

    # Correlation between the two similarity measures
    correlation = float(
        np.corrcoef(act_pairwise, st_pairwise)[0, 1],
    )

    # Rank correlation (Spearman)
    from scipy.stats import spearmanr
    rho, p_value = spearmanr(act_pairwise, st_pairwise)

    print(f"\n  {name} \u2014 Space Comparison:")
    print(f"    Pearson r:  {correlation:.4f}")
    print(
        f"    Spearman \u03c1: {float(rho):.4f}"
        f" (p={float(p_value):.2e})",
    )
    print("    Interpretation: ", end="")
    if abs(correlation) < 0.3:
        print("WEAK correlation \u2014 spaces capture different structure")
    elif abs(correlation) < 0.6:
        print("MODERATE correlation \u2014 partial overlap")
    else:
        print("STRONG correlation \u2014 spaces agree")

    return {
        "pearson_r": correlation,
        "spearman_rho": float(rho),
        "spearman_p": float(p_value),
    }


def main() -> None:
    """Run model-space diversity verification."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Model-space prompt diversity verification",
    )
    parser.add_argument(
        "--model", type=str,
        default="mlx-community/Qwen2.5-1.5B-Instruct-bf16",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.95,
        help="Activation cosine similarity threshold",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON report",
    )
    parser.add_argument(
        "--st-report", type=str, default=None,
        help="Path to sentence-transformer report for comparison",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).parent

    # Datasets to analyze
    datasets: dict[str, Path] = {
        "harmful_100": data_dir / "harmful_100.jsonl",
        "harmless_100": data_dir / "harmless_100.jsonl",
        "harmful_infix_100": data_dir / "harmful_infix_100.jsonl",
    }

    # Load all entries
    loaded: dict[str, list[dict[str, str]]] = {}
    all_prompts: list[str] = []
    set_ranges: dict[str, tuple[int, int]] = {}

    for name, path in datasets.items():
        if path.exists():
            entries = load_jsonl(path)
            loaded[name] = entries
            start = len(all_prompts)
            all_prompts.extend(e["prompt"] for e in entries)
            set_ranges[name] = (start, len(all_prompts))
            print(f"Loaded {name}: {len(entries)} prompts")

    if not loaded:
        print("No datasets found.")
        sys.exit(1)

    # Run all prompts through model in one pass
    t0 = time.time()
    all_activations, all_projections, best_layer, cosine_scores = (
        extract_activations(args.model, all_prompts)
    )
    elapsed = time.time() - t0
    print(
        f"\nActivation extraction: {elapsed:.1f}s"
        f" for {len(all_prompts)} prompts",
    )

    report: dict[str, ModelReportEntry] = {
        "model": args.model,
        "best_layer": best_layer,
        "cosine_scores": cosine_scores,
        "total_prompts": len(all_prompts),
        "extraction_time_s": round(elapsed, 1),
    }

    # Per-set analysis
    for name, entries in loaded.items():
        start, end = set_ranges[name]
        act = all_activations[start:end]
        proj = all_projections[start:end]
        stats = analyze_set(
            name, entries, act, proj, args.threshold,
        )
        report[name] = stats

    # Cross-set: harmful vs harmless in model space
    if "harmful_100" in loaded and "harmless_100" in loaded:
        h_start, h_end = set_ranges["harmful_100"]
        s_start, s_end = set_ranges["harmless_100"]
        h_act = all_activations[h_start:h_end]
        s_act = all_activations[s_start:s_end]
        h_proj = all_projections[h_start:h_end]
        s_proj = all_projections[s_start:s_end]

        # Activation-space cross-similarity
        norm_h = h_act / np.maximum(
            np.linalg.norm(h_act, axis=1, keepdims=True),
            1e-10,
        )
        norm_s = s_act / np.maximum(
            np.linalg.norm(s_act, axis=1, keepdims=True),
            1e-10,
        )
        cross_sim = norm_h @ norm_s.T

        print(f"\n{'='*60}")
        print(
            "  Cross-set: harmful_100 vs harmless_100"
            " (Model Space)",
        )
        print(f"{'='*60}")
        print(
            f"  Mean cross-similarity: {np.mean(cross_sim):.4f}",
        )
        print(
            f"  Max cross-similarity:  {np.max(cross_sim):.4f}",
        )

        # Refusal projection separation
        h_mean = float(np.mean(h_proj))
        s_mean = float(np.mean(s_proj))
        h_std = float(np.std(h_proj))
        s_std = float(np.std(s_proj))
        pooled_std = np.sqrt((h_std**2 + s_std**2) / 2)
        cohens_d = (
            (h_mean - s_mean) / pooled_std
            if pooled_std > 0
            else 0.0
        )

        print("\n  Refusal projection separation:")
        print(
            f"    Harmful mean:  {h_mean:+.4f}"
            f" (std={h_std:.4f})",
        )
        print(
            f"    Harmless mean: {s_mean:+.4f}"
            f" (std={s_std:.4f})",
        )
        print(f"    Cohen's d:     {cohens_d:+.4f}")
        print("    Interpretation: ", end="")
        if abs(cohens_d) > 0.8:
            print(
                "LARGE separation \u2014 model clearly"
                " distinguishes harmful from harmless",
            )
        elif abs(cohens_d) > 0.5:
            print("MEDIUM separation")
        else:
            print(
                "SMALL separation \u2014 prompts may be"
                " too similar in refusal space",
            )

        report["cross_harmful_harmless"] = {
            "act_mean_cross_sim": float(np.mean(cross_sim)),
            "act_max_cross_sim": float(np.max(cross_sim)),
            "harmful_proj_mean": h_mean,
            "harmless_proj_mean": s_mean,
            "cohens_d": cohens_d,
        }

    # Compare with sentence-transformer if available
    st_report_path = args.st_report or (
        data_dir / "diversity_report.json"
    )
    if Path(st_report_path).exists():
        print(f"\n{'='*60}")
        print(
            "  Combined: Model Space vs Sentence-Transformer",
        )
        print(f"{'='*60}")

        try:
            from sentence_transformers import SentenceTransformer

            st_model = SentenceTransformer("all-MiniLM-L6-v2")

            for name, entries in loaded.items():
                start, end = set_ranges[name]
                act = all_activations[start:end]
                prompts = [e["prompt"] for e in entries]

                # Get ST embeddings
                st_emb = st_model.encode(
                    prompts,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )

                act_sim = cosine_similarity_matrix(act)
                st_sim = cosine_similarity_matrix(st_emb)

                comparison = compare_spaces(
                    name, act_sim, st_sim, len(prompts),
                )
                report[f"comparison_{name}"] = comparison

        except ImportError:
            print(
                "  sentence-transformers not available"
                " \u2014 loading cached report",
            )
            with open(st_report_path) as f:
                st_data = json.load(f)
            print(
                "  Loaded cached ST report"
                f" with {len(st_data)} entries",
            )
            report["st_report_loaded"] = True

    # Summary verdict
    print(f"\n{'='*60}")
    print("  COMBINED VERDICT")
    print(f"{'='*60}")

    for name in loaded:
        stats = report.get(name, {})
        if not isinstance(stats, dict):
            continue
        act_mean = stats.get("act_mean_sim", 0.0)
        act_dupes = stats.get("act_near_duplicates", 0)
        proj_range = stats.get("proj_range", 0.0)
        cat_spread = stats.get("category_proj_spread", 0.0)

        # Verdict based on activation diversity + projection
        act_ok = act_dupes == 0
        proj_ok = (
            isinstance(proj_range, float) and proj_range > 1.0
        )

        if (
            isinstance(act_mean, float)
            and isinstance(act_dupes, int)
        ):
            if act_ok and proj_ok:
                verdict = "PASS"
            elif act_ok:
                verdict = "REVIEW"
            else:
                verdict = "FAIL"
            print(
                f"  {name:<25} "
                f"act_sim={act_mean:.3f} "
                f"dupes={act_dupes} "
                f"proj_range={proj_range:.2f} "
                f"cat_spread={cat_spread:.3f} "
                f"[{verdict}]",
            )

    if "cross_harmful_harmless" in report:
        cross = report["cross_harmful_harmless"]
        if isinstance(cross, dict):
            d = cross.get("cohens_d", 0.0)
            print(
                "\n  Harmful/harmless separation:"
                f" Cohen's d = {d:+.3f}",
                end="",
            )
            if isinstance(d, float) and abs(d) > 0.8:
                print(" [GOOD \u2014 large effect]")
            else:
                print(" [WEAK \u2014 may need harder prompts]")

    # Save report
    if args.output:
        output_path = Path(args.output)

        def _serialize(
            obj: ModelSection,
        ) -> ModelSection:
            """Ensure JSON-serializable types."""
            result: ModelSection = {}
            for k, v in obj.items():
                if isinstance(v, np.floating):
                    result[k] = float(v)
                elif isinstance(v, np.integer):
                    result[k] = int(v)
                elif isinstance(v, list):
                    result[k] = [
                        float(x)
                        if isinstance(x, np.floating)
                        else x
                        for x in v
                    ]
                else:
                    result[k] = v
            return result

        serializable: dict[str, ModelReportEntry] = {}
        for k, v in report.items():
            if isinstance(v, dict):
                serializable[k] = _serialize(v)
            elif isinstance(v, np.floating):
                serializable[k] = float(v)
            elif isinstance(v, np.integer):
                serializable[k] = int(v)
            elif isinstance(v, list):
                serializable[k] = [
                    float(x)
                    if isinstance(x, (np.floating, float))
                    else x
                    for x in v
                ]
            else:
                serializable[k] = v

        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()
