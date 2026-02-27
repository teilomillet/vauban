"""Export modified model weights as a loadable mlx-lm model directory.

The bare ``save_weights()`` in ``cut.py`` writes a single safetensors
file. That's not loadable by ``mlx_lm.load()`` — it needs config.json,
tokenizer files, etc. This module copies all non-weight files from the
source model and writes the modified weights, producing a directory
that ``mlx_lm.load()`` can open directly.
"""

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from vauban._array import Array

if TYPE_CHECKING:
    from vauban._backend import get_backend as _get_backend

    _BACKEND = _get_backend()
else:
    from vauban._backend import get_backend

    _BACKEND = get_backend()


def export_model(
    model_path: str,
    weights: dict[str, Array],
    output_dir: str | Path,
) -> Path:
    """Export modified weights as a complete mlx-lm model directory.

    Resolves the source model directory (local path or HuggingFace cache),
    copies all non-safetensors files (config.json, tokenizer, etc.), and
    writes the full weight dict as ``model.safetensors``.

    Args:
        model_path: HuggingFace model ID or local path (same as passed
            to ``mlx_lm.load()``).
        weights: Complete weight dict (all keys, not just modified ones).
        output_dir: Destination directory for the exported model.

    Returns:
        The output directory path.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    source_dir = _resolve_source_dir(model_path)

    # Copy all non-safetensors files from source
    for src_file in source_dir.iterdir():
        if src_file.is_file() and not src_file.name.endswith(".safetensors"):
            shutil.copy2(src_file, output / src_file.name)

    # Write the modified weights
    _save_weights(output / "model.safetensors", weights)

    return output


if TYPE_CHECKING or _BACKEND == "mlx":
    def _save_weights(path: Path, weights: dict[str, Array]) -> None:
        """Save weight dict to safetensors using MLX."""
        import mlx.core as mx

        mx.save_safetensors(str(path), weights)

elif _BACKEND == "torch":
    def _save_weights(path: Path, weights: dict[str, Array]) -> None:
        """Save weight dict to safetensors using PyTorch."""
        from safetensors.torch import save_file

        save_file(weights, str(path))

else:
    msg = f"Unknown backend: {_BACKEND!r}"
    raise ValueError(msg)


def _resolve_source_dir(model_path: str) -> Path:
    """Resolve a model path or HuggingFace ID to a local directory.

    Uses ``mlx_lm.utils._download()`` on MLX or ``huggingface_hub``
    on PyTorch. Both handle local paths and HuggingFace Hub downloads.
    """
    local = Path(model_path)
    if local.is_dir():
        return local
    if _BACKEND == "mlx":
        from mlx_lm.utils import _download

        return _download(model_path)

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_path))
