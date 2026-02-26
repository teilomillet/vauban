"""Export modified model weights as a loadable mlx-lm model directory.

The bare ``save_weights()`` in ``cut.py`` writes a single safetensors
file. That's not loadable by ``mlx_lm.load()`` — it needs config.json,
tokenizer files, etc. This module copies all non-weight files from the
source model and writes the modified weights, producing a directory
that ``mlx_lm.load()`` can open directly.
"""

import shutil
from pathlib import Path

import mlx.core as mx

from vauban._array import Array


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
    mx.save_safetensors(str(output / "model.safetensors"), weights)

    return output


def _resolve_source_dir(model_path: str) -> Path:
    """Resolve a model path or HuggingFace ID to a local directory.

    Uses ``mlx_lm.utils._download()`` which handles both local paths
    and HuggingFace Hub downloads.
    """
    from mlx_lm.utils import _download

    return _download(model_path)
