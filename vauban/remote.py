"""Remote model probing via batch inference APIs.

Currently supports the ``jsinfer`` backend (Jane Street batch inference).
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from pathlib import Path

    from vauban.types import RemoteConfig

# Base URL for jsinfer API
_JSINFER_BASE = "https://jsinfer.janestreet.com"


def run_remote_probe(
    *,
    cfg: RemoteConfig,
    api_key: str,
    output_dir: Path,
    verbose: bool = True,
    t0: float = 0.0,
) -> dict[str, object]:
    """Run a remote probe against all configured models.

    Returns the full report payload as a JSON-serialisable dict.
    Activation tensors (if requested) are saved as ``.npy`` files.
    """
    from vauban._pipeline._context import log

    models_data: dict[str, object] = {}

    for model_id in cfg.models:
        log(
            f"  Probing {model_id} ...",
            verbose=verbose,
            elapsed=time.monotonic() - t0,
        )

        # --- Chat completions ---
        responses = _query_model(
            api_key=api_key,
            model_id=model_id,
            prompts=cfg.prompts,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
        )

        model_entry: dict[str, object] = {
            "model_id": model_id,
            "responses": responses,
        }

        # --- Activations (optional) ---
        if cfg.activations and cfg.activation_layers:
            act_files = _fetch_activations(
                api_key=api_key,
                model_id=model_id,
                prompts=cfg.prompts,
                layers=cfg.activation_layers,
                modules=cfg.activation_modules,
                output_dir=output_dir,
                timeout=cfg.timeout,
            )
            model_entry["activation_files"] = act_files

        models_data[model_id] = model_entry

    return {
        "backend": cfg.backend,
        "n_models": len(cfg.models),
        "n_prompts": len(cfg.prompts),
        "activations_requested": cfg.activations,
        "models": models_data,
    }


def _query_model(
    *,
    api_key: str,
    model_id: str,
    prompts: list[str],
    max_tokens: int,
    timeout: int,
) -> list[dict[str, str]]:
    """Send chat completions to a single model and return responses."""
    results: list[dict[str, str]] = []

    for prompt in prompts:
        body = json.dumps({
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }).encode()

        req = Request(
            f"{_JSINFER_BASE}/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            choices = data.get("choices", [])
            content = (
                choices[0].get("message", {}).get("content", "")
                if choices
                else ""
            )
            results.append({"prompt": prompt, "response": content})
        except Exception as exc:
            results.append({"prompt": prompt, "error": str(exc)})

    return results


def _fetch_activations(
    *,
    api_key: str,
    model_id: str,
    prompts: list[str],
    layers: list[int],
    modules: list[str],
    output_dir: Path,
    timeout: int,
) -> list[str]:
    """Fetch activation tensors and save as .npy files.

    Returns a list of saved file paths (relative to output_dir).
    """
    import numpy as np

    saved_files: list[str] = []

    # Expand module patterns for each layer
    expanded_modules: list[str] = []
    for layer in layers:
        for pattern in modules:
            expanded_modules.append(pattern.format(layer=layer))

    for prompt_idx, prompt in enumerate(prompts):
        body = json.dumps({
            "model": model_id,
            "prompt": prompt,
            "modules": expanded_modules,
        }).encode()

        req = Request(
            f"{_JSINFER_BASE}/v1/activations",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())

            activations = data.get("activations", {})
            for module_name, values in activations.items():
                arr = np.array(values, dtype=np.float32)
                safe_model = model_id.replace("/", "_")
                filename = (
                    f"activations_{safe_model}"
                    f"_p{prompt_idx}_{module_name}.npy"
                )
                filepath = output_dir / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(filepath), arr)
                saved_files.append(filename)

        except Exception:
            # Best-effort: skip failed activation fetches
            continue

    return saved_files
