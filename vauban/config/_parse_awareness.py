# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Parse the optional ``[awareness]`` TOML section into an :class:`AwarenessConfig`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vauban.config._parse_helpers import SectionReader, require_toml_table
from vauban.types import AwarenessConfig

if TYPE_CHECKING:
    from vauban.config._types import TomlDict


def _parse_awareness(raw: TomlDict) -> AwarenessConfig | None:
    """Parse the optional [awareness] section into an AwarenessConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("awareness")
    if sec is None:
        return None
    reader = SectionReader("[awareness]", require_toml_table("[awareness]", sec))

    # -- prompts (required) --
    prompts = reader.string_list("prompts")
    if not prompts:
        msg = "[awareness].prompts must be non-empty"
        raise ValueError(msg)

    # -- calibration_prompt (optional, default "Hello") --
    calibration_prompt = reader.optional_string("calibration_prompt")
    if calibration_prompt is None:
        calibration_prompt = "Hello"
    elif not calibration_prompt:
        msg = "[awareness].calibration_prompt must be non-empty"
        raise ValueError(msg)

    # -- mode (optional, default "full") --
    mode = reader.literal("mode", ("fast", "full"), default="full")

    # -- n_power_iterations (optional, default 5) --
    n_power_iterations = reader.integer("n_power_iterations", default=5)
    if n_power_iterations < 1:
        msg = f"[awareness].n_power_iterations must be >= 1, got {n_power_iterations}"
        raise ValueError(msg)

    # -- fd_epsilon (optional, default 1e-4) --
    fd_epsilon = reader.number("fd_epsilon", default=1e-4)
    if fd_epsilon <= 0:
        msg = f"[awareness].fd_epsilon must be > 0, got {fd_epsilon}"
        raise ValueError(msg)

    # -- valley_window (optional, default 3) --
    valley_window = reader.integer("valley_window", default=3)
    if valley_window < 1:
        msg = f"[awareness].valley_window must be >= 1, got {valley_window}"
        raise ValueError(msg)

    # -- top_k_valleys (optional, default 3) --
    top_k_valleys = reader.integer("top_k_valleys", default=3)
    if top_k_valleys < 1:
        msg = f"[awareness].top_k_valleys must be >= 1, got {top_k_valleys}"
        raise ValueError(msg)

    # -- gain_ratio_threshold (optional, default 2.0) --
    gain_ratio_threshold = reader.number("gain_ratio_threshold", default=2.0)
    if gain_ratio_threshold <= 0:
        msg = (
            f"[awareness].gain_ratio_threshold must be > 0,"
            f" got {gain_ratio_threshold}"
        )
        raise ValueError(msg)

    # -- rank_ratio_threshold (optional, default 0.5) --
    rank_ratio_threshold = reader.number("rank_ratio_threshold", default=0.5)
    if rank_ratio_threshold <= 0:
        msg = (
            f"[awareness].rank_ratio_threshold must be > 0,"
            f" got {rank_ratio_threshold}"
        )
        raise ValueError(msg)

    # -- correlation_delta_threshold (optional, default 0.3) --
    correlation_delta_threshold = reader.number(
        "correlation_delta_threshold", default=0.3,
    )
    if correlation_delta_threshold <= 0:
        msg = (
            f"[awareness].correlation_delta_threshold must be > 0,"
            f" got {correlation_delta_threshold}"
        )
        raise ValueError(msg)

    # -- min_anomalous_layers (optional, default 2) --
    min_anomalous_layers = reader.integer("min_anomalous_layers", default=2)
    if min_anomalous_layers < 1:
        msg = (
            f"[awareness].min_anomalous_layers must be >= 1,"
            f" got {min_anomalous_layers}"
        )
        raise ValueError(msg)

    # -- confidence_threshold (optional, default 0.5) --
    confidence_threshold = reader.number("confidence_threshold", default=0.5)
    if confidence_threshold < 0:
        msg = (
            f"[awareness].confidence_threshold must be >= 0,"
            f" got {confidence_threshold}"
        )
        raise ValueError(msg)

    return AwarenessConfig(
        prompts=prompts,
        calibration_prompt=calibration_prompt,
        mode=mode,
        n_power_iterations=n_power_iterations,
        fd_epsilon=fd_epsilon,
        valley_window=valley_window,
        top_k_valleys=top_k_valleys,
        gain_ratio_threshold=gain_ratio_threshold,
        rank_ratio_threshold=rank_ratio_threshold,
        correlation_delta_threshold=correlation_delta_threshold,
        min_anomalous_layers=min_anomalous_layers,
        confidence_threshold=confidence_threshold,
    )
