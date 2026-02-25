"""REPL convenience API for interactive exploration.

Import as ``from vauban import quick`` — not in ``__all__``.

Provides one-liner wrappers around the core pipeline for Jupyter / REPL use.
"""

from pathlib import Path

import mlx.core as mx

from vauban.dequantize import dequantize_model, is_quantized
from vauban.types import CausalLM, DirectionResult, ProbeResult, SteerResult, Tokenizer

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
