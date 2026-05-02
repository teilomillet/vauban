# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the Torch-first runtime surface contract."""

from __future__ import annotations

from pathlib import Path

from vauban.runtime import torch_capabilities

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PRIMITIVES: tuple[str, ...] = (
    "load_model",
    "tokenize",
    "forward",
    "runtime_trace",
    "generation",
    "logits",
    "logprobs",
    "activations",
    "interventions",
    "kv_cache",
    "weight_access",
    "mutable_weights",
    "safetensors_io",
    "peft_lora_export",
    "device_profile_cuda",
    "device_profile_mps",
    "profile_sweep",
)

TORCH_CAPABILITY_ROWS: tuple[str, ...] = (
    "logits",
    "logprobs",
    "activations",
    "interventions",
    "kv_cache",
    "weight_access",
    "mutable_weights",
)

BLOCKED_DEFAULT_TOKENS: tuple[str, ...] = (
    "mlx-community",
    "mlx_lm",
    "import mlx",
    "from mlx",
    'runtime_backend = "mlx"',
    'backend = "mlx"',
)


def test_torch_surface_matrix_covers_required_primitives() -> None:
    """Require the documented Torch matrix to cover the full runtime surface."""
    primitives = _surface_matrix_primitives()
    missing = sorted(set(REQUIRED_PRIMITIVES).difference(primitives))
    duplicates = sorted(
        primitive for primitive in set(primitives) if primitives.count(primitive) > 1
    )

    assert not missing
    assert not duplicates


def test_torch_capabilities_match_surface_matrix() -> None:
    """Keep declared Torch evidence capabilities aligned with the matrix."""
    capabilities = torch_capabilities()

    assert capabilities.name == "torch"
    assert set(capabilities.device_kinds).issuperset({"cpu", "cuda", "mps"})
    for capability in TORCH_CAPABILITY_ROWS:
        assert capabilities.support_level(capability) == "full"


def test_mps_boundary_names_activation_projection_primitive() -> None:
    """Pin custom MPS work to one evidence primitive, not a new product surface."""
    text = (ROOT / "docs/research/mps-primitive-boundary.md").read_text()

    assert "activation projection" in text
    assert "intervention inside the trace path" in text
    assert "TraceArtifact" in text
    assert "StageProfile" in text
    assert "Torch reference" in text


def test_default_facing_docs_and_examples_do_not_use_mlx_defaults() -> None:
    """Prevent public quickstarts/examples from drifting back to MLX defaults."""
    offenders: list[str] = []
    for path in _default_facing_paths():
        text = path.read_text()
        for token in BLOCKED_DEFAULT_TOKENS:
            if token in text:
                offenders.append(f"{path.relative_to(ROOT)} contains {token!r}")

    assert not offenders


def test_mkdocs_nav_lists_runtime_research_docs() -> None:
    """Expose the Torch and MPS runtime direction in the documentation nav."""
    text = (ROOT / "mkdocs.yml").read_text()

    assert "Torch Surface Matrix: research/torch_surface_matrix.md" in text
    assert "MPS Primitive Boundary: research/mps-primitive-boundary.md" in text


def _surface_matrix_primitives() -> list[str]:
    """Return primitive names listed in the Torch surface matrix table."""
    text = (ROOT / "docs/research/torch_surface_matrix.md").read_text()
    primitives: list[str] = []
    for line in text.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        first_cell = cells[0]
        if first_cell.startswith("`") and first_cell.endswith("`"):
            primitives.append(first_cell.strip("`"))
    return primitives


def _default_facing_paths() -> list[Path]:
    """Return docs and examples where MLX should not appear as the default."""
    paths: list[Path] = [ROOT / "README.md"]

    for directory in (ROOT / "docs", ROOT / "examples"):
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix not in {".md", ".toml"}:
                continue
            if _is_legacy_or_reproduction_path(path):
                continue
            paths.append(path)

    paths.extend(
        [
            ROOT / "vauban/_init.py",
            ROOT / "vauban/session.py",
        ]
    )
    return paths


def _is_legacy_or_reproduction_path(path: Path) -> bool:
    """Return whether a doc/example path is intentionally MLX-specific."""
    parts = path.relative_to(ROOT).parts
    if len(parts) >= 2 and parts[0] == "docs" and parts[1] == "class":
        return True
    if len(parts) >= 3 and parts[:3] == ("docs", "research", "reports"):
        return True
    return len(parts) >= 2 and parts[0] == "examples" and parts[1] == "reproductions"
