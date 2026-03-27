# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Architecture auto-detection for transformer models.

Probes model objects at load time using ordered fallback lists to
normalize any HuggingFace architecture (Llama, GPT-2, Phi, Mistral,
Qwen, etc.) to a canonical interface with ``embed_tokens``, ``layers``,
and ``norm`` attributes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vauban.types import LayerComponents

# Ordered fallback lists — covers Llama/Mistral/Qwen/Gemma/
# StarCoder/Phi/GPT-2/GPTNeoX/Cohere/Qwen3.5-VL
_INNER_ATTRS: tuple[str, ...] = ("model", "transformer", "gpt_neox", "language_model")
_EMBED_ATTRS: tuple[str, ...] = (
    "embed_tokens", "wte", "embed", "embedding",
)
_LAYERS_ATTRS: tuple[str, ...] = (
    "layers", "h", "blocks", "decoder_layers",
)
_NORM_ATTRS: tuple[str, ...] = (
    "norm", "ln_f", "final_layernorm", "final_ln", "model_norm",
)


def _find_attr(obj: object, candidates: tuple[str, ...]) -> object:
    """Find the first matching attribute from a list of candidates.

    Args:
        obj: The object to probe.
        candidates: Ordered tuple of attribute names to try.

    Raises:
        AttributeError: If none of the candidates are found.
    """
    for attr in candidates:
        val = getattr(obj, attr, None)
        if val is not None:
            return val
    msg = f"Cannot find any of {candidates} on {type(obj).__name__}"
    raise AttributeError(msg)


def _has_canonical_attrs(obj: object) -> bool:
    """Check whether *obj* has the canonical transformer interface."""
    return (
        hasattr(obj, "embed_tokens")
        and hasattr(obj, "layers")
        and hasattr(obj, "norm")
    )


def get_inner_model(model: object, *, _depth: int = 0) -> object:
    """Find the innermost transformer from a CausalLM wrapper.

    Recursively probes ``model``, ``transformer``, ``gpt_neox``,
    ``language_model`` attributes up to 3 levels deep to reach the
    canonical transformer (the one with ``embed_tokens``, ``layers``,
    ``norm``).

    Handles nested wrappers like Qwen3.5 VL where the path is
    ``Model → language_model → TextModel → model → Qwen3_5TextModel``.

    Raises:
        AttributeError: If no inner transformer is found.
    """
    if _depth > 3:
        msg = f"Recursion limit reached probing {type(model).__name__}"
        raise AttributeError(msg)

    fallback: object | None = None
    for attr in _INNER_ATTRS:
        val = getattr(model, attr, None)
        if val is None:
            continue
        # Found a canonical transformer — return immediately
        if _has_canonical_attrs(val):
            return val
        # Not canonical — try to go deeper
        try:
            return get_inner_model(val, _depth=_depth + 1)
        except AttributeError:
            # Couldn't go deeper — save as fallback, try remaining candidates
            if fallback is None:
                fallback = val

    if fallback is not None:
        return fallback

    # Flat architecture: layers live directly on the model
    if hasattr(model, "layers"):
        return model
    msg = f"Cannot find any of {_INNER_ATTRS} on {type(model).__name__}"
    raise AttributeError(msg)


def normalize_transformer(inner: object) -> object:
    """Return an object with canonical ``.embed_tokens``, ``.layers``, ``.norm``.

    If the inner model already has all three attributes, return as-is.
    Otherwise wrap with ``TransformerAdapter``.
    """
    if (
        hasattr(inner, "embed_tokens")
        and hasattr(inner, "layers")
        and hasattr(inner, "norm")
    ):
        return inner
    return TransformerAdapter(inner)


class TransformerAdapter:
    """Thin proxy that normalizes architecture-specific attribute names.

    Resolves ``embed_tokens``, ``layers``, and ``norm`` once at init
    using ordered fallback lists, then proxies all other attribute
    access to the underlying model.
    """

    def __init__(self, inner: object) -> None:
        # Use object.__setattr__ to avoid triggering __getattr__
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "embed_tokens", _find_attr(inner, _EMBED_ATTRS))
        object.__setattr__(self, "layers", _find_attr(inner, _LAYERS_ATTRS))
        object.__setattr__(self, "norm", _find_attr(inner, _NORM_ATTRS))

    def __getattr__(self, name: str) -> object:
        """Proxy all other attribute access to the underlying model."""
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# Layer component detection for circuit tracing
# ---------------------------------------------------------------------------

# Ordered fallback lists for layer internal components
_ATTN_ATTRS: tuple[str, ...] = (
    "self_attn", "attn", "attention", "self_attention",
)
_MLP_ATTRS: tuple[str, ...] = (
    "mlp", "feed_forward", "ff", "ffn",
)
_INPUT_NORM_ATTRS: tuple[str, ...] = (
    "input_layernorm", "ln_1", "norm1", "pre_attn_norm",
    "attn_norm", "layer_norm1",
)
_POST_ATTN_NORM_ATTRS: tuple[str, ...] = (
    "post_attention_layernorm", "ln_2", "norm2", "post_attn_norm",
    "ffn_norm", "layer_norm2",
)


def detect_layer_components(layer: object) -> LayerComponents:
    """Probe a transformer layer for its internal components.

    Detects self_attn, mlp, input_layernorm, and post_attention_layernorm
    using ordered fallback attribute lists to handle different model
    architectures (Llama, GPT-2, Phi, Mistral, Qwen, etc.).

    Args:
        layer: A single transformer layer object.

    Returns:
        LayerComponents with references to the detected sub-modules.

    Raises:
        AttributeError: If any required component cannot be found.
    """
    from vauban.types import LayerComponents as _LayerComponents

    return _LayerComponents(
        input_norm=_find_attr(layer, _INPUT_NORM_ATTRS),
        self_attn=_find_attr(layer, _ATTN_ATTRS),
        post_attn_norm=_find_attr(layer, _POST_ATTN_NORM_ATTRS),
        mlp=_find_attr(layer, _MLP_ATTRS),
    )
