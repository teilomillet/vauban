# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Parse the [intent] section of a TOML config."""

from typing import Literal, cast

from vauban.config._types import TomlDict
from vauban.types import IntentConfig


def _parse_intent(raw: TomlDict) -> IntentConfig | None:
    """Parse the optional [intent] section into an IntentConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("intent")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[intent] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    intent_dict = cast("dict[str, object]", sec)

    # -- mode --
    mode_raw = intent_dict.get("mode", "embedding")
    if not isinstance(mode_raw, str):
        msg = (
            f"[intent].mode must be a string,"
            f" got {type(mode_raw).__name__}"
        )
        raise TypeError(msg)
    if mode_raw not in ("embedding", "judge"):
        msg = (
            f"[intent].mode must be one of ('embedding', 'judge'),"
            f" got {mode_raw!r}"
        )
        raise ValueError(msg)
    mode: Literal["embedding", "judge"] = (
        "embedding" if mode_raw == "embedding" else "judge"
    )

    # -- target_layer --
    target_layer_raw = intent_dict.get("target_layer")
    target_layer: int | None = None
    if target_layer_raw is not None:
        if not isinstance(target_layer_raw, int):
            msg = (
                f"[intent].target_layer must be an integer,"
                f" got {type(target_layer_raw).__name__}"
            )
            raise TypeError(msg)
        target_layer = target_layer_raw

    # -- similarity_threshold --
    sim_raw = intent_dict.get("similarity_threshold", 0.7)
    if not isinstance(sim_raw, int | float):
        msg = (
            f"[intent].similarity_threshold must be a number,"
            f" got {type(sim_raw).__name__}"
        )
        raise TypeError(msg)

    # -- max_tokens --
    max_tokens_raw = intent_dict.get("max_tokens", 10)
    if not isinstance(max_tokens_raw, int):
        msg = (
            f"[intent].max_tokens must be an integer,"
            f" got {type(max_tokens_raw).__name__}"
        )
        raise TypeError(msg)

    return IntentConfig(
        mode=mode,
        target_layer=target_layer,
        similarity_threshold=float(sim_raw),
        max_tokens=max_tokens_raw,
    )
