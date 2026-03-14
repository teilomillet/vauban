"""Backend-agnostic orchestrator for remote model probing."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from vauban.types import RemoteActivationResult, RemoteBackend, RemoteConfig


def run_remote_probe(
    *,
    cfg: RemoteConfig,
    api_key: str,
    output_dir: Path,
    verbose: bool = True,
    t0: float = 0.0,
    backend: RemoteBackend | None = None,
) -> dict[str, object]:
    """Run a remote probe against all configured models.

    Resolves the backend from the registry if *backend* is not provided.
    Returns the full report payload as a JSON-serialisable dict.
    Activation tensors (if requested) are saved as ``.npy`` files.

    Args:
        cfg: Remote configuration from TOML.
        api_key: API key for the backend.
        output_dir: Directory for activation files and reports.
        verbose: Whether to log progress.
        t0: Reference time for elapsed logging.
        backend: Optional pre-built backend (for testing). If ``None``,
            resolved from ``cfg.backend`` via the registry.

    Returns:
        Report dict with ``backend``, ``models``, etc.
    """
    if backend is None:
        from vauban.remote._registry import get_backend

        backend = get_backend(cfg.backend, api_key)

    return asyncio.run(_run_probe_async(
        cfg=cfg,
        backend=backend,
        output_dir=output_dir,
        verbose=verbose,
        t0=t0,
    ))


async def _run_probe_async(
    *,
    cfg: RemoteConfig,
    backend: RemoteBackend,
    output_dir: Path,
    verbose: bool,
    t0: float,
) -> dict[str, object]:
    """Async probe logic -- backend-agnostic."""
    from vauban._pipeline._context import log

    models_data: dict[str, object] = {}

    for model_id in cfg.models:
        log(
            f"  Probing {model_id} ...",
            verbose=verbose,
            elapsed=time.monotonic() - t0,
        )

        # --- Chat completions ---
        chat_results = await backend.chat(model_id, cfg.prompts, cfg.max_tokens)

        responses: list[dict[str, str]] = []
        for r in chat_results:
            if r.error is not None:
                responses.append({"prompt": r.prompt, "error": r.error})
            else:
                responses.append({"prompt": r.prompt, "response": r.response or ""})

        log(
            f"    {len(responses)} responses collected",
            verbose=verbose,
            elapsed=time.monotonic() - t0,
        )

        model_entry: dict[str, object] = {
            "model_id": model_id,
            "responses": responses,
        }

        # --- Activations (optional) ---
        if cfg.activations and cfg.activation_layers:
            expanded_modules: list[str] = []
            for layer in cfg.activation_layers:
                for pattern in cfg.activation_modules:
                    expanded_modules.append(pattern.format(layer=layer))

            log(
                f"    Requesting activations ({len(cfg.prompts)} prompts"
                f" x {len(expanded_modules)} modules) ...",
                verbose=verbose,
                elapsed=time.monotonic() - t0,
            )

            act_results = await backend.activations(
                model_id, cfg.prompts, expanded_modules,
            )

            saved_files = _save_activations(act_results, model_id, output_dir)
            model_entry["activation_files"] = saved_files

            log(
                f"    Saved {len(saved_files)} activation files",
                verbose=verbose,
                elapsed=time.monotonic() - t0,
            )

        models_data[model_id] = model_entry

    return {
        "backend": cfg.backend,
        "n_models": len(cfg.models),
        "n_prompts": len(cfg.prompts),
        "activations_requested": cfg.activations,
        "models": models_data,
    }


def _save_activations(
    results: list[RemoteActivationResult],
    model_id: str,
    output_dir: Path,
) -> list[str]:
    """Save activation results as .npy files.

    Args:
        results: Activation results from the backend.
        model_id: Model identifier (used in filenames).
        output_dir: Directory to save files.

    Returns:
        List of saved filenames (relative to output_dir).
    """
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    for act in results:
        safe_model = model_id.replace("/", "_").replace("-", "_")
        safe_module = act.module_name.replace(".", "_")
        filename = f"act_{safe_model}_p{act.prompt_index:03d}_{safe_module}.npy"
        filepath = output_dir / filename
        np.save(str(filepath), np.array(act.data, dtype=np.float32))
        saved.append(filename)

    return saved
