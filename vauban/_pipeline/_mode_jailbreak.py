# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Jailbreak template evaluation early-mode runner."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import TYPE_CHECKING
from typing import cast as _cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report

if TYPE_CHECKING:
    from vauban.types import CausalLM, Tokenizer


def _run_jailbreak_mode(context: EarlyModeContext) -> None:
    """Run [jailbreak] early-return mode and write its report."""
    from vauban.defend import defend_content
    from vauban.jailbreak import apply_templates, filter_by_strategy, load_templates
    from vauban.measure import load_prompts
    from vauban.types import DefenseStackConfig, JailbreakStrategyResult

    config = context.config
    jcfg = config.jailbreak
    if jcfg is None:
        msg = "jailbreak config is required for jailbreak mode"
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    log(
        "Running jailbreak template evaluation",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    # Load templates
    templates = load_templates(jcfg.custom_templates_path)
    if jcfg.strategies:
        templates = filter_by_strategy(templates, jcfg.strategies)
    log(
        f"Loaded {len(templates)} templates"
        f" ({len({t.strategy for t in templates})} strategies)",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    # Load payloads
    if jcfg.payloads_from == "harmful":
        if context.harmful is None:
            msg = "harmful prompts required for jailbreak mode"
            raise ValueError(msg)
        payloads = context.harmful[:config.eval.num_prompts]
    else:
        payloads = load_prompts(jcfg.payloads_from)

    # Generate cross-product of templates x payloads
    expanded = apply_templates(templates, payloads)
    log(
        f"Generated {len(expanded)} jailbreak prompts"
        f" ({len(templates)} templates x {len(payloads)} payloads)",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    # Build a defense stack config — reuse [defend] if present, else minimal
    defend_cfg = config.defend or DefenseStackConfig(
        scan=config.scan,
        sic=config.sic,
        policy=config.policy,
        intent=config.intent,
    )

    # Get direction if available
    if context.direction_result is not None:
        direction_vec = context.direction_result.direction
        layer_idx = context.direction_result.layer_index
    else:
        direction_vec = None
        layer_idx = 0

    # Run each expanded prompt through the defense stack
    per_strategy_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "blocked": 0},
    )
    results_detail: list[dict[str, object]] = []

    for template, prompt in expanded:
        result = defend_content(
            model, tokenizer, prompt,
            direction_vec, defend_cfg, layer_idx,
        )
        per_strategy_counts[template.strategy]["total"] += 1
        if result.blocked:
            per_strategy_counts[template.strategy]["blocked"] += 1
        results_detail.append({
            "strategy": template.strategy,
            "template_name": template.name,
            "blocked": result.blocked,
            "layer_that_blocked": result.layer_that_blocked,
        })

    # Aggregate
    total_prompts = len(expanded)
    total_blocked = sum(c["blocked"] for c in per_strategy_counts.values())
    block_rate = total_blocked / total_prompts if total_prompts else 0.0

    per_strategy: list[JailbreakStrategyResult] = []
    for strategy, counts in sorted(per_strategy_counts.items()):
        sr = JailbreakStrategyResult(
            strategy=strategy,
            total_prompts=counts["total"],
            total_blocked=counts["blocked"],
            block_rate=(
                counts["blocked"] / counts["total"]
                if counts["total"] else 0.0
            ),
        )
        per_strategy.append(sr)

    report = {
        "total_prompts": total_prompts,
        "total_blocked": total_blocked,
        "block_rate": block_rate,
        "per_strategy": [
            {
                "strategy": s.strategy,
                "total_prompts": s.total_prompts,
                "total_blocked": s.total_blocked,
                "block_rate": s.block_rate,
            }
            for s in per_strategy
        ],
        "results": results_detail,
    }

    report_path = write_mode_report(
        config.output_dir,
        ModeReport("jailbreak_report.json", report),
    )
    log(
        f"Done — jailbreak report written to {report_path}"
        f" (block rate: {block_rate:.1%})",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    finish_mode_run(
        context,
        "jailbreak",
        ["jailbreak_report.json"],
        {"block_rate": block_rate},
    )
