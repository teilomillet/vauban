# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Core field parsing for the [softprompt] config section."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vauban.config._parse_helpers import SectionReader

if TYPE_CHECKING:
    from vauban.config._types import TomlDict


@dataclass(frozen=True, slots=True)
class _SoftPromptCoreSection:
    """Parsed core softprompt configuration values."""

    mode: str
    n_tokens: int
    n_steps: int
    learning_rate: float
    init_scale: float
    batch_size: int
    top_k: int
    target_prefixes: list[str]
    max_gen_tokens: int
    seed: int | None
    patience: int
    lr_schedule: str
    n_restarts: int
    prompt_strategy: str
    direction_mode: str
    direction_layers: list[int] | None
    init_tokens: list[int] | None


def _parse_softprompt_core(sec: TomlDict) -> _SoftPromptCoreSection:
    """Parse the core [softprompt] fields."""
    reader = SectionReader("[softprompt]", sec)

    mode = reader.literal(
        "mode",
        ("continuous", "gcg", "egd", "cold", "amplecgc"),
        default="continuous",
    )

    n_tokens = reader.integer("n_tokens", default=16)
    if n_tokens < 1:
        msg = f"[softprompt].n_tokens must be >= 1, got {n_tokens}"
        raise ValueError(msg)

    n_steps = reader.integer("n_steps", default=200)
    if n_steps < 1:
        msg = f"[softprompt].n_steps must be >= 1, got {n_steps}"
        raise ValueError(msg)

    learning_rate = reader.number("learning_rate", default=0.01)
    if learning_rate <= 0:
        msg = f"[softprompt].learning_rate must be > 0, got {learning_rate}"
        raise ValueError(msg)

    init_scale = reader.number("init_scale", default=0.1)

    batch_size = reader.integer("batch_size", default=64)
    if batch_size < 1:
        msg = f"[softprompt].batch_size must be >= 1, got {batch_size}"
        raise ValueError(msg)

    top_k = reader.integer("top_k", default=256)
    if top_k < 1:
        msg = f"[softprompt].top_k must be >= 1, got {top_k}"
        raise ValueError(msg)

    target_prefixes = reader.string_list(
        "target_prefixes",
        default=["Sure", "Here"],
        coerce=True,
    )

    max_gen_tokens = reader.integer("max_gen_tokens", default=100)
    seed = reader.optional_integer("seed")

    patience = reader.integer("patience", default=0)
    if patience < 0:
        msg = f"[softprompt].patience must be >= 0, got {patience}"
        raise ValueError(msg)

    lr_schedule = reader.literal(
        "lr_schedule",
        ("constant", "cosine"),
        default="constant",
    )

    n_restarts = reader.integer("n_restarts", default=1)
    if n_restarts < 1:
        msg = f"[softprompt].n_restarts must be >= 1, got {n_restarts}"
        raise ValueError(msg)

    prompt_strategy = reader.literal(
        "prompt_strategy",
        ("all", "cycle", "first", "worst_k", "sample"),
        default="all",
    )
    direction_mode = reader.literal(
        "direction_mode",
        ("last", "raid", "all_positions"),
        default="last",
    )

    direction_layers = reader.optional_int_list("direction_layers")
    init_tokens = reader.optional_int_list("init_tokens")

    return _SoftPromptCoreSection(
        mode=mode,
        n_tokens=n_tokens,
        n_steps=n_steps,
        learning_rate=learning_rate,
        init_scale=init_scale,
        batch_size=batch_size,
        top_k=top_k,
        target_prefixes=target_prefixes,
        max_gen_tokens=max_gen_tokens,
        seed=seed,
        patience=patience,
        lr_schedule=lr_schedule,
        n_restarts=n_restarts,
        prompt_strategy=prompt_strategy,
        direction_mode=direction_mode,
        direction_layers=direction_layers,
        init_tokens=init_tokens,
    )
