"""Architecture auto-detection for transformer models.

Probes model objects at load time using ordered fallback lists to
normalize any HuggingFace architecture (Llama, GPT-2, Phi, Mistral,
Qwen, etc.) to a canonical interface with ``embed_tokens``, ``layers``,
and ``norm`` attributes.
"""

# Ordered fallback lists — covers Llama/Mistral/Qwen/Gemma/
# StarCoder/Phi/GPT-2/GPTNeoX/Cohere
_INNER_ATTRS: tuple[str, ...] = ("model", "transformer", "gpt_neox")
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


def get_inner_model(model: object) -> object:
    """Find the inner transformer from a CausalLM wrapper.

    Probes ``model.model``, ``model.transformer``, ``model.gpt_neox``
    in order, returning the first that exists.

    Raises:
        AttributeError: If no inner transformer is found.
    """
    return _find_attr(model, _INNER_ATTRS)


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
