"""REPL convenience API for interactive exploration.

Import as ``from vauban import quick`` — not in ``__all__``.

Provides one-liner wrappers around the core pipeline for Jupyter / REPL use.
"""

from pathlib import Path

import mlx.core as mx

from vauban.dequantize import dequantize_model, is_quantized
from vauban.geometry import DirectionGeometryResult
from vauban.types import (
    CausalLM,
    DirectionResult,
    EvalResult,
    ProbeResult,
    SteerResult,
    SurfaceResult,
    Tokenizer,
)

_DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"


def load(
    model_path: str = _DEFAULT_MODEL,
) -> tuple[CausalLM, Tokenizer]:
    """Load a model and auto-dequantize if quantized.

    Args:
        model_path: HuggingFace model ID or local path.

    Returns:
        (model, tokenizer) tuple ready for measure/probe/steer.
    """
    import mlx_lm

    model, tokenizer = mlx_lm.load(model_path)  # type: ignore[invalid-assignment]
    if is_quantized(model):
        dequantize_model(model)
    return model, tokenizer  # type: ignore[return-value]


def measure_direction(
    model: CausalLM,
    tokenizer: Tokenizer,
    harmful: list[str] | None = None,
    harmless: list[str] | None = None,
) -> DirectionResult:
    """Measure the refusal direction. Uses bundled prompts if None.

    Args:
        model: Loaded causal language model.
        tokenizer: Tokenizer with chat template support.
        harmful: Harmful prompt list, or None for bundled defaults.
        harmless: Harmless prompt list, or None for bundled defaults.

    Returns:
        DirectionResult with the extracted refusal direction.
    """
    from vauban.measure import default_prompt_paths, load_prompts, measure

    if harmful is None or harmless is None:
        h_path, hl_path = default_prompt_paths()
        if harmful is None:
            harmful = load_prompts(h_path)
        if harmless is None:
            harmless = load_prompts(hl_path)

    return measure(model, tokenizer, harmful, harmless)


def probe_prompt(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    direction: DirectionResult | mx.array,
) -> ProbeResult:
    """Probe one prompt against a direction.

    Args:
        model: Loaded causal language model.
        tokenizer: Tokenizer with chat template support.
        prompt: The prompt to probe.
        direction: A DirectionResult or raw mx.array direction vector.

    Returns:
        ProbeResult with per-layer projections.
    """
    from vauban.probe import probe

    if isinstance(direction, DirectionResult):
        direction = direction.direction

    return probe(model, tokenizer, prompt, direction)


def steer_prompt(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    direction: DirectionResult | mx.array,
    alpha: float = 1.0,
    max_tokens: int = 100,
) -> SteerResult:
    """Generate text with direction removal (steered generation).

    Args:
        model: Loaded causal language model.
        tokenizer: Tokenizer with chat template support.
        prompt: The prompt to generate from.
        direction: A DirectionResult or raw mx.array direction vector.
        alpha: Steering strength.
        max_tokens: Maximum tokens to generate.

    Returns:
        SteerResult with generated text and projections.
    """
    from vauban.probe import steer

    if isinstance(direction, DirectionResult):
        direction = direction.direction

    layers = list(range(len(model.model.layers)))
    return steer(model, tokenizer, prompt, direction, layers, alpha, max_tokens)


def abliterate(
    model: CausalLM,
    tokenizer: Tokenizer,
    model_path: str,
    output_dir: str | Path = "output",
    alpha: float = 1.0,
    harmful: list[str] | None = None,
    harmless: list[str] | None = None,
) -> DirectionResult:
    """Full measure -> cut -> export in one call.

    Args:
        model: Loaded causal language model.
        tokenizer: Tokenizer with chat template support.
        model_path: Original model path (needed for export to copy configs).
        output_dir: Where to write the abliterated model.
        alpha: Cut strength.
        harmful: Harmful prompts, or None for bundled defaults.
        harmless: Harmless prompts, or None for bundled defaults.

    Returns:
        The DirectionResult used for cutting.
    """
    from mlx.utils import tree_flatten

    from vauban.cut import cut
    from vauban.export import export_model

    direction_result = measure_direction(model, tokenizer, harmful, harmless)

    flat_weights: dict[str, mx.array] = dict(tree_flatten(model.parameters()))  # type: ignore[attr-defined]
    target_layers = list(range(len(model.model.layers)))

    modified_weights = cut(
        flat_weights, direction_result.direction, target_layers, alpha,
    )

    export_model(model_path, modified_weights, output_dir)

    return direction_result


def compare(dir_a: str | Path, dir_b: str | Path) -> str:
    """Compare JSON reports from two output directories.

    Wraps diff_reports + format_diff for REPL use.

    Args:
        dir_a: First output directory.
        dir_b: Second output directory.

    Returns:
        Formatted diff string.
    """
    from vauban._diff import diff_reports, format_diff

    a = Path(dir_a)
    b = Path(dir_b)
    reports = diff_reports(a, b)
    return format_diff(a, b, reports)


def scan(
    model: CausalLM,
    tokenizer: Tokenizer,
    direction: DirectionResult | mx.array,
    direction_layer: int | None = None,
) -> SurfaceResult:
    """One-liner surface scan with bundled default prompts.

    Args:
        model: Loaded causal language model.
        tokenizer: Tokenizer with chat template support.
        direction: A DirectionResult or raw mx.array direction vector.
        direction_layer: Layer index for direction projection.
            Inferred from DirectionResult if not provided; defaults to 0
            for raw arrays.

    Returns:
        SurfaceResult from the default surface prompt set.
    """
    from vauban.surface import default_surface_path, load_surface_prompts, map_surface

    if isinstance(direction, DirectionResult):
        if direction_layer is None:
            direction_layer = direction.layer_index
        direction_vec = direction.direction
    else:
        if direction_layer is None:
            direction_layer = 0
        direction_vec = direction

    prompts = load_surface_prompts(default_surface_path())
    return map_surface(
        model, tokenizer, prompts, direction_vec, direction_layer,
        progress=False,
    )


def evaluate(
    original: CausalLM,
    modified: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str] | None = None,
    *,
    max_tokens: int = 100,
) -> EvalResult:
    """Evaluate original vs modified model. Uses bundled harmful prompts if None.

    Args:
        original: Original (unmodified) causal language model.
        modified: Modified (abliterated) causal language model.
        tokenizer: Tokenizer with chat template support.
        prompts: Eval prompt list, or None to use first 20 bundled harmful prompts.
        max_tokens: Maximum tokens to generate per prompt.

    Returns:
        EvalResult with refusal rates, perplexity, and KL divergence.
    """
    from vauban.evaluate import evaluate as _evaluate
    from vauban.measure import default_prompt_paths, load_prompts

    if prompts is None:
        h_path, _ = default_prompt_paths()
        prompts = load_prompts(h_path)[:20]

    return _evaluate(original, modified, tokenizer, prompts, max_tokens=max_tokens)  # type: ignore[arg-type]


def analyze_geometry(
    directions: dict[str, DirectionResult | mx.array],
    independence_threshold: float = 0.1,
) -> DirectionGeometryResult:
    """Analyze geometric relationships between multiple directions.

    Convenience wrapper that accepts DirectionResult or raw mx.array values.

    Args:
        directions: Mapping from direction name to DirectionResult or mx.array.
        independence_threshold: Shared variance below this marks a pair as
            independent.

    Returns:
        DirectionGeometryResult with pairwise analysis.
    """
    from vauban.geometry import analyze_directions

    raw: dict[str, mx.array] = {}
    for name, d in directions.items():
        if isinstance(d, DirectionResult):
            raw[name] = d.direction
        else:
            raw[name] = d

    return analyze_directions(raw, independence_threshold)
