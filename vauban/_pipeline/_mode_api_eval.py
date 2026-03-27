# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Standalone API eval early-mode runner.

Evaluates pre-optimized adversarial tokens against remote API endpoints
without loading a local model.
"""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report

if TYPE_CHECKING:
    from vauban.types import TransferEvalResult


def _run_api_eval_mode(context: EarlyModeContext) -> None:
    """Run standalone [api_eval] mode and write its report."""
    config = context.config
    api_cfg = config.api_eval
    if api_cfg is None or api_cfg.token_text is None:
        msg = "[api_eval] with token_text is required for standalone api_eval mode"
        raise ValueError(msg)

    v = config.verbose

    if api_cfg.defense_proxy is not None:
        log(
            "WARNING: defense_proxy is set but standalone mode has no"
            " local model — skipping proxy, sending prompts directly",
            verbose=v,
            elapsed=time.monotonic() - context.t0,
        )

    log(
        f"Standalone API eval — {len(api_cfg.endpoints)} endpoint(s),"
        f" {len(api_cfg.prompts)} prompt(s)",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    from vauban.api_eval import evaluate_suffix_via_api

    results = evaluate_suffix_via_api(
        suffix_text=api_cfg.token_text,
        prompts=api_cfg.prompts,
        config=api_cfg,
        token_position=api_cfg.token_position,
    )

    # Build report payload
    report_payload = _build_report(api_cfg.token_text, api_cfg.prompts, results)
    report = ModeReport(filename="api_eval_report.json", payload=report_payload)
    report_path = write_mode_report(config.output_dir, report)

    log(
        f"API eval report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    # Print summary table to stderr
    _print_api_eval_summary(results, api_cfg.prompts)

    # Compute aggregate success rate for experiment log
    avg_asr = (
        sum(r.success_rate for r in results) / len(results)
        if results
        else 0.0
    )

    finish_mode_run(
        context,
        "api_eval",
        [str(report_path)],
        {"avg_success_rate": avg_asr, "n_endpoints": len(results)},
    )


def _build_report(
    token_text: str,
    prompts: list[str],
    results: list[TransferEvalResult],
) -> dict[str, object]:
    """Build the JSON report payload."""
    endpoints: dict[str, object] = {}
    for r in results:
        n = len(prompts)
        bypassed = int(r.success_rate * n)
        endpoints[r.model_id] = {
            "success_rate": r.success_rate,
            "n_prompts": n,
            "n_bypass": bypassed,
            "responses": r.eval_responses,
        }

    return {
        "token_text": token_text,
        "n_prompts": len(prompts),
        "endpoints": endpoints,
    }


def _print_api_eval_summary(
    results: list[TransferEvalResult],
    prompts: list[str],
) -> None:
    """Print a summary table to stderr."""
    n = len(prompts)
    print(
        f"\n{'Endpoint':<25} {'Bypass':>8} {'ASR':>6}",
        file=sys.stderr,
    )
    print("-" * 42, file=sys.stderr)
    for r in results:
        bypassed = int(r.success_rate * n)
        print(
            f"{r.model_id:<25} {bypassed:>3}/{n:<3} {r.success_rate:>5.0%}",
            file=sys.stderr,
        )
    print(file=sys.stderr, flush=True)
