# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Generate agent worlds from skeletons for flywheel cycles."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import TYPE_CHECKING, Literal, cast

from vauban.flywheel._skeletons import get_skeleton
from vauban.types import (
    EnvironmentConfig,
    EnvironmentTask,
    ToolSchema,
    WorldMeta,
)

if TYPE_CHECKING:
    import mlx.nn as nn
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from transformers import PreTrainedTokenizer

    from vauban.types import CausalLM, Tokenizer

type Position = Literal["prefix", "suffix", "infix"]

_PAYLOAD_MARKER = "{payload}"


def generate_worlds(
    skeletons: list[str],
    n_worlds: int,
    difficulty_range: tuple[int, int],
    positions: list[str],
    seed: int | None,
    model: CausalLM | None = None,
    tokenizer: Tokenizer | None = None,
    model_expand: bool = False,
    expand_temperature: float = 0.7,
    expand_max_tokens: int = 200,
) -> list[tuple[EnvironmentConfig, WorldMeta]]:
    """Generate a list of (EnvironmentConfig, WorldMeta) pairs.

    Each world is created from a skeleton template with a random task
    and difficulty level, using either static detail pool expansion or
    model-based generation.

    Args:
        skeletons: Names of skeleton templates to draw from.
        n_worlds: Number of worlds to generate.
        difficulty_range: Min/max complexity levels (1-5).
        positions: Injection positions to distribute across worlds.
        seed: Random seed for reproducibility.
        model: Optional model for model-based expansion.
        tokenizer: Optional tokenizer for model-based expansion.
        model_expand: If True and model/tokenizer provided, use model
            to generate task details instead of static pools.
        expand_temperature: Temperature for model-based generation.
        expand_max_tokens: Max tokens for model-based generation.

    Returns:
        List of (EnvironmentConfig, WorldMeta) tuples.
    """
    rng = random.Random(seed)
    d_min, d_max = difficulty_range

    results: list[tuple[EnvironmentConfig, WorldMeta]] = []

    for i in range(n_worlds):
        skel_name = rng.choice(skeletons)
        skeleton = get_skeleton(skel_name)
        complexity = rng.randint(d_min, d_max)
        position = cast("Position", rng.choice(positions))

        # Pick a task template and expand it
        template_idx = rng.randint(
            0, len(skeleton.task_templates) - 1,
        )
        template = skeleton.task_templates[template_idx]
        expected_tools = skeleton.expected_tools_by_template[template_idx]
        pool_idx = template_idx % len(skeleton.detail_pools)
        pool = skeleton.detail_pools[pool_idx]

        if (
            model_expand
            and model is not None
            and tokenizer is not None
        ):
            detail = _model_expand_detail(
                model, tokenizer, skeleton.domain, template,
                expand_temperature, expand_max_tokens,
            )
        else:
            detail = rng.choice(pool)

        task_content = template.replace("{detail}", detail)
        tools = _configure_tools_for_world(
            skeleton.tools,
            skeleton.injection_surface,
            position,
        )

        env_config = EnvironmentConfig(
            system_prompt=skeleton.system_prompt_template,
            tools=tools,
            target=skeleton.target,
            task=EnvironmentTask(content=task_content),
            injection_surface=skeleton.injection_surface,
            injection_position=position,
            benign_expected_tools=list(expected_tools),
        )

        meta = WorldMeta(
            domain=skel_name,
            skeleton=skel_name,
            complexity=complexity,
            position=position,
            seed_offset=i,
        )

        results.append((env_config, meta))

    return results


def _model_expand_detail(
    model: CausalLM,
    tokenizer: Tokenizer,
    domain: str,
    template: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Use the model to generate a realistic task detail.

    Falls back to a placeholder if generation fails.
    """
    try:
        from mlx_lm.generate import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = (
            f"Generate a short, realistic detail for an agent task"
            f" in the {domain} domain. The task template is:"
            f" '{template}'. Reply with ONLY the detail text."
        )
        sampler = make_sampler(temp=temperature)
        mlx_model = cast("nn.Module", model)
        mlx_tokenizer = cast(
            "PreTrainedTokenizer | TokenizerWrapper",
            tokenizer,
        )
        result = mlx_generate(
            mlx_model,
            mlx_tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )
        detail = result.strip()
        if detail:
            return detail
    except Exception:  # mlx_lm import or generation failure
        pass
    return "item-001"


def _configure_tools_for_world(
    tools: list[ToolSchema],
    injection_surface: str,
    position: Position,
) -> list[ToolSchema]:
    """Return per-world tool schemas with position-aware injection output."""
    configured_tools: list[ToolSchema] = []
    for tool in tools:
        if tool.name != injection_surface:
            configured_tools.append(tool)
            continue
        base_result = tool.result if tool.result is not None else "{}"
        configured_tools.append(replace(
            tool,
            result=_position_tool_result(base_result, position),
        ))
    return configured_tools


def _position_tool_result(
    base_result: str,
    position: Position,
) -> str:
    """Build a result template that inserts the payload at *position*."""
    if position == "prefix":
        return f"{_PAYLOAD_MARKER}\n\n{base_result}"
    if position == "infix":
        return (
            f"{base_result}\n\n"
            f"{_PAYLOAD_MARKER}\n\n"
            "The retrieved content continues below."
        )
    return f"{base_result}\n\n{_PAYLOAD_MARKER}"
