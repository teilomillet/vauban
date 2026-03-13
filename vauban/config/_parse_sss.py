"""Parse the optional ``[sss]`` TOML section into an :class:`SSSConfig`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vauban.config._parse_helpers import SectionReader
from vauban.types import SSSConfig

if TYPE_CHECKING:
    from vauban.config._types import TomlDict


def _parse_sss(raw: TomlDict) -> SSSConfig | None:
    """Parse the optional [sss] section into an SSSConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("sss")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[sss] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    reader = SectionReader("[sss]", sec)

    # -- prompts (required) --
    prompts = reader.string_list("prompts")
    if not prompts:
        msg = "[sss].prompts must be non-empty"
        raise ValueError(msg)

    # -- layers (optional) --
    layers = reader.optional_int_list("layers")

    # -- alpha (optional, default 1.0) --
    alpha = reader.number("alpha", default=1.0)

    # -- max_tokens (optional, default 100) --
    max_tokens = reader.integer("max_tokens", default=100)
    if max_tokens < 1:
        msg = f"[sss].max_tokens must be >= 1, got {max_tokens}"
        raise ValueError(msg)

    # -- calibration_prompt (optional, default "Hello") --
    calibration_prompt = reader.optional_string("calibration_prompt") or "Hello"

    # -- n_power_iterations (optional, default 5) --
    n_power_iterations = reader.integer("n_power_iterations", default=5)
    if n_power_iterations < 1:
        msg = f"[sss].n_power_iterations must be >= 1, got {n_power_iterations}"
        raise ValueError(msg)

    # -- fd_epsilon (optional, default 1e-4) --
    fd_epsilon = reader.number("fd_epsilon", default=1e-4)
    if fd_epsilon <= 0:
        msg = f"[sss].fd_epsilon must be > 0, got {fd_epsilon}"
        raise ValueError(msg)

    # -- seed_floor (optional, default 0.01) --
    seed_floor = reader.number("seed_floor", default=0.01)

    # -- valley_window (optional, default 3) --
    valley_window = reader.integer("valley_window", default=3)
    if valley_window < 1:
        msg = f"[sss].valley_window must be >= 1, got {valley_window}"
        raise ValueError(msg)

    # -- top_k_valleys (optional, default 3) --
    top_k_valleys = reader.integer("top_k_valleys", default=3)
    if top_k_valleys < 1:
        msg = f"[sss].top_k_valleys must be >= 1, got {top_k_valleys}"
        raise ValueError(msg)

    return SSSConfig(
        prompts=prompts,
        layers=layers,
        alpha=alpha,
        max_tokens=max_tokens,
        calibration_prompt=calibration_prompt,
        n_power_iterations=n_power_iterations,
        fd_epsilon=fd_epsilon,
        seed_floor=seed_floor,
        valley_window=valley_window,
        top_k_valleys=top_k_valleys,
    )
