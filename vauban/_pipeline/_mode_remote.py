# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Remote model probing early-mode runner.

Sends prompts to remote batch APIs (e.g. jsinfer) and collects
responses + optional activation tensors.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report

if TYPE_CHECKING:
    from vauban.types import RemoteConfig


def _load_dotenv(config_path: str | Path) -> None:
    """Auto-load .env from the config file's directory (best-effort)."""
    env_path = Path(config_path).parent / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except ImportError:
        # dotenv not installed — read manually
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            # Strip surrounding quotes — .env files commonly use
            # KEY="value" or KEY='value', and the quotes should not
            # become part of the environment variable value.
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key.strip(), value)


def _run_remote_mode(context: EarlyModeContext) -> None:
    """Run standalone [remote] mode and write its report."""
    config = context.config
    remote_cfg = config.remote
    if remote_cfg is None:
        msg = "[remote] section is required for remote mode"
        raise ValueError(msg)

    v = config.verbose

    # Auto-load .env from config directory
    _load_dotenv(context.config_path)

    # Read API key from environment
    api_key = os.environ.get(remote_cfg.api_key_env)
    if not api_key:
        msg = (
            f"Environment variable {remote_cfg.api_key_env!r} is not set"
            f" or empty — required for [remote] backend"
        )
        raise ValueError(msg)

    log(
        f"Remote probe — {remote_cfg.backend} backend,"
        f" {len(remote_cfg.models)} model(s),"
        f" {len(remote_cfg.prompts)} prompt(s)",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    from vauban.remote import run_remote_probe

    result = run_remote_probe(
        cfg=remote_cfg,
        api_key=api_key,
        output_dir=config.output_dir,
        verbose=v,
        t0=context.t0,
    )

    # Write main report
    report = ModeReport(filename="remote_report.json", payload=result)
    report_path = write_mode_report(config.output_dir, report)

    log(
        f"Remote report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    # Print summary table
    _print_summary(remote_cfg, result)

    # Compute metrics for experiment log
    n_errors = 0
    n_total = 0
    models_data = result.get("models")
    if isinstance(models_data, dict):
        for _mid, mentry in models_data.items():
            if isinstance(mentry, dict):
                entry = cast("dict[str, object]", mentry)
                responses = entry.get("responses")
                if isinstance(responses, list):
                    n_total += len(responses)
                    n_errors += sum(
                        1 for r in responses
                        if isinstance(r, dict) and "error" in r
                    )

    finish_mode_run(
        context,
        "remote",
        [str(report_path)],
        {
            "n_models": len(remote_cfg.models),
            "n_prompts": len(remote_cfg.prompts),
            "n_responses": n_total,
            "n_errors": n_errors,
        },
    )


def _print_summary(
    cfg: RemoteConfig,
    result: dict[str, object],
) -> None:
    """Print a summary table to stderr."""
    models_data = result.get("models")
    if not isinstance(models_data, dict):
        return
    models = cast("dict[str, object]", models_data)

    print(
        f"\n{'Model':<30} {'Responses':>10} {'Errors':>8} {'Activations':>12}",
        file=sys.stderr,
    )
    print("-" * 63, file=sys.stderr)

    for model_id in cfg.models:
        model_entry = models.get(model_id)
        if not isinstance(model_entry, dict):
            continue
        entry = cast("dict[str, object]", model_entry)
        responses = entry.get("responses")
        n_responses = len(responses) if isinstance(responses, list) else 0
        n_errors = 0
        if isinstance(responses, list):
            n_errors = sum(
                1 for r in responses
                if isinstance(r, dict) and "error" in r
            )
        act_files = entry.get("activation_files")
        n_acts = len(act_files) if isinstance(act_files, list) else 0
        act_str = str(n_acts) if n_acts > 0 else "-"
        print(
            f"{model_id:<30} {n_responses:>10} {n_errors:>8} {act_str:>12}",
            file=sys.stderr,
        )

    # Print response previews
    print(file=sys.stderr)
    for model_id in cfg.models:
        model_entry = models.get(model_id)
        if not isinstance(model_entry, dict):
            continue
        entry = cast("dict[str, object]", model_entry)
        responses = entry.get("responses")
        if not isinstance(responses, list):
            continue
        print(f"── {model_id} ──", file=sys.stderr)
        for r in responses:
            if not isinstance(r, dict):
                continue
            response_entry = cast("dict[str, object]", r)
            prompt_value = response_entry.get("prompt")
            prompt = prompt_value if isinstance(prompt_value, str) else "?"
            if "error" in response_entry:
                error = response_entry.get("error")
                print(f"  [{prompt}] ERROR: {error}", file=sys.stderr)
            else:
                response_text = response_entry.get("response")
                resp = response_text if isinstance(response_text, str) else ""
                preview = resp[:120].replace("\n", " ")
                print(f"  [{prompt}]", file=sys.stderr)
                print(f"    → {preview}", file=sys.stderr)
        print(file=sys.stderr)

    print(file=sys.stderr, flush=True)
