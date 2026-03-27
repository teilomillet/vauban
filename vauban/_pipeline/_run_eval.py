# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Evaluation phase and final metric collection for the main runner."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from vauban._pipeline._context import log

if TYPE_CHECKING:
    from vauban._pipeline._run_state import RunState


def run_eval_phase(state: RunState) -> None:
    """Evaluate the modified model and collect final report metadata."""
    from vauban.evaluate import evaluate
    from vauban.measure import load_prompts

    config = state.config
    if config.eval.prompts_path is not None and state.modified_model is not None:
        log(
            "Evaluating modified model",
            verbose=state.verbose,
            elapsed=state.elapsed(),
        )
        eval_prompts = load_prompts(config.eval.prompts_path)
        result = evaluate(
            state.model,
            state.modified_model,
            state.tokenizer,
            eval_prompts,
            refusal_phrases=state.refusal_phrases,
            max_tokens=config.eval.max_tokens,
            refusal_mode=config.eval.refusal_mode,
        )
        state.eval_refusal_rate = result.refusal_rate_modified
        eval_report = {
            "refusal_rate_original": result.refusal_rate_original,
            "refusal_rate_modified": result.refusal_rate_modified,
            "perplexity_original": result.perplexity_original,
            "perplexity_modified": result.perplexity_modified,
            "kl_divergence": result.kl_divergence,
            "num_prompts": result.num_prompts,
        }
        (config.output_dir / "eval_report.json").write_text(
            json.dumps(eval_report, indent=2),
        )
        state.report_files.append("eval_report.json")
        state.metrics["refusal_rate_modified"] = result.refusal_rate_modified
