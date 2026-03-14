"""Remote model probing early-mode runner.

Sends prompts to remote batch APIs (e.g. jsinfer) and collects
responses + optional activation tensors.
"""

from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING, cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report

if TYPE_CHECKING:
    from vauban.types import RemoteConfig


def _run_remote_mode(context: EarlyModeContext) -> None:
    """Run standalone [remote] mode and write its report."""
    config = context.config
    remote_cfg = config.remote
    if remote_cfg is None:
        msg = "[remote] section is required for remote mode"
        raise ValueError(msg)

    v = config.verbose

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

    finish_mode_run(
        context,
        "remote",
        ["remote_report.json"],
        {
            "n_models": len(remote_cfg.models),
            "n_prompts": len(remote_cfg.prompts),
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
        f"\n{'Model':<30} {'Responses':>10} {'Activations':>12}",
        file=sys.stderr,
    )
    print("-" * 55, file=sys.stderr)

    for model_id in cfg.models:
        model_entry = models.get(model_id)
        if not isinstance(model_entry, dict):
            continue
        entry = cast("dict[str, object]", model_entry)
        responses = entry.get("responses")
        n_responses = len(responses) if isinstance(responses, list) else 0
        act_files = entry.get("activation_files")
        n_acts = len(act_files) if isinstance(act_files, list) else 0
        act_str = str(n_acts) if n_acts > 0 else "-"
        print(
            f"{model_id:<30} {n_responses:>10} {act_str:>12}",
            file=sys.stderr,
        )
    print(file=sys.stderr, flush=True)
