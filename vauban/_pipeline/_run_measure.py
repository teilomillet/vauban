"""Prompt loading, detection, and measure-phase orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from vauban._pipeline._context import log, write_experiment_log
from vauban._pipeline._helpers import load_refusal_phrases, write_measure_reports

if TYPE_CHECKING:
    from vauban._pipeline._run_state import RunState


def run_measure_phase(state: RunState) -> bool:
    """Run prompt loading, detection, measurement, and measure-only exit."""
    from vauban import _ops as ops
    from vauban._model_io import load_model
    from vauban._serializers import _detect_to_dict, _direction_transfer_to_dict
    from vauban.dataset import resolve_prompts
    from vauban.dequantize import dequantize_model, is_quantized
    from vauban.detect import detect
    from vauban.measure import (
        load_prompts,
        measure,
        measure_dbdi,
        measure_diff,
        measure_subspace,
        measure_subspace_bank,
    )
    from vauban.transfer import check_direction_transfer
    from vauban.types import DirectionResult, DirectionTransferResult

    config = state.config
    log("Loading prompts", verbose=state.verbose, elapsed=state.elapsed())
    state.harmful = resolve_prompts(config.harmful_path)
    state.harmless = resolve_prompts(config.harmless_path)

    if config.eval.refusal_phrases_path is not None:
        state.refusal_phrases = load_refusal_phrases(
            config.eval.refusal_phrases_path,
        )

    if config.detect is not None:
        log(
            "Running defense detection",
            verbose=state.verbose,
            elapsed=state.elapsed(),
        )
        detect_result = detect(
            state.model,
            state.tokenizer,
            state.harmful,
            state.harmless,
            config.detect,
        )
        report_path = config.output_dir / "detect_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(_detect_to_dict(detect_result), indent=2))

    clip_quantile = config.measure.clip_quantile
    log(
        f"Measuring refusal direction (mode={config.measure.mode})",
        verbose=state.verbose,
        elapsed=state.elapsed(),
    )
    if config.measure.mode == "subspace":
        state.subspace_result = measure_subspace(
            state.model,
            state.tokenizer,
            state.harmful,
            state.harmless,
            config.measure.top_k,
            clip_quantile,
        )
        if config.measure.bank:
            base_dir = Path(state.config_path).resolve().parent
            bank_entries: list[tuple[str, list[str], list[str]]] = []
            for entry in config.measure.bank:
                bank_harmful = (
                    state.harmful
                    if entry.harmful == "default"
                    else load_prompts(base_dir / entry.harmful)
                )
                bank_harmless = (
                    state.harmless
                    if entry.harmless == "default"
                    else load_prompts(base_dir / entry.harmless)
                )
                bank_entries.append((entry.name, bank_harmful, bank_harmless))
            log(
                f"Measuring subspace bank ({len(bank_entries)} entries)",
                verbose=state.verbose,
                elapsed=state.elapsed(),
            )
            bank_results = measure_subspace_bank(
                state.model,
                state.tokenizer,
                bank_entries,
                config.measure.top_k,
                clip_quantile,
            )
            bank_path = config.output_dir / "subspace_bank.safetensors"
            bank_path.parent.mkdir(parents=True, exist_ok=True)
            ops.save_safetensors(
                str(bank_path),
                {name: result.basis for name, result in bank_results.items()},
            )
            log(
                (
                    f"Subspace bank written to {bank_path}"
                    f" ({len(bank_results)} subspaces)"
                ),
                verbose=state.verbose,
                elapsed=state.elapsed(),
            )
    elif config.measure.mode == "diff":
        if config.measure.diff_model is None:
            msg = "measure.diff_model is required when mode='diff'"
            raise ValueError(msg)
        log(
            f"Loading base model for diff: {config.measure.diff_model}",
            verbose=state.verbose,
            elapsed=state.elapsed(),
        )
        base_model, _ = load_model(config.measure.diff_model)
        if is_quantized(base_model):
            log(
                "Dequantizing base model weights",
                verbose=state.verbose,
                elapsed=state.elapsed(),
            )
            dequantize_model(base_model)
        state.diff_result = measure_diff(
            base_model,
            state.model,
            top_k=config.measure.top_k,
            source_model_id=config.measure.diff_model,
            target_model_id=config.model_path,
        )
        state.measure_reports = write_measure_reports(
            config,
            direction_result=None,
            subspace_result=None,
            dbdi_result=None,
            diff_result=state.diff_result,
        )
        log(
            f"Diff report written to {config.output_dir / 'diff_report.json'}",
            verbose=state.verbose,
            elapsed=state.elapsed(),
        )
        state.direction_result = state.diff_result.best_direction()
    elif config.measure.mode == "dbdi":
        state.dbdi_result = measure_dbdi(
            state.model,
            state.tokenizer,
            state.harmful,
            state.harmless,
            clip_quantile,
        )
        if config.cut.dbdi_target in ("red", "both"):
            state.direction_result = DirectionResult(
                direction=state.dbdi_result.red,
                layer_index=state.dbdi_result.red_layer_index,
                cosine_scores=state.dbdi_result.red_cosine_scores,
                d_model=state.dbdi_result.d_model,
                model_path=state.dbdi_result.model_path,
                layer_types=state.dbdi_result.layer_types,
            )
            state.cosine_scores = state.dbdi_result.red_cosine_scores
        else:
            state.direction_result = DirectionResult(
                direction=state.dbdi_result.hdd,
                layer_index=state.dbdi_result.hdd_layer_index,
                cosine_scores=state.dbdi_result.hdd_cosine_scores,
                d_model=state.dbdi_result.d_model,
                model_path=state.dbdi_result.model_path,
                layer_types=state.dbdi_result.layer_types,
            )
            state.cosine_scores = state.dbdi_result.hdd_cosine_scores
    else:
        state.direction_result = measure(
            state.model,
            state.tokenizer,
            state.harmful,
            state.harmless,
            clip_quantile,
        )
        state.cosine_scores = state.direction_result.cosine_scores

    if config.measure.transfer_models and state.direction_result is not None:
        transfer_results: list[DirectionTransferResult] = []
        for transfer_model_id in config.measure.transfer_models:
            log(
                f"Testing direction transfer on {transfer_model_id}",
                verbose=state.verbose,
                elapsed=state.elapsed(),
            )
            transfer_model, _ = load_model(transfer_model_id)
            if is_quantized(transfer_model):
                dequantize_model(transfer_model)
            transfer_results.append(
                check_direction_transfer(
                    transfer_model,
                    state.tokenizer,
                    state.direction_result.direction,
                    state.harmful,
                    state.harmless,
                    transfer_model_id,
                    clip_quantile,
                ),
            )
        transfer_report_path = config.output_dir / "transfer_report.json"
        transfer_report_path.parent.mkdir(parents=True, exist_ok=True)
        transfer_report_path.write_text(
            json.dumps(
                [_direction_transfer_to_dict(result) for result in transfer_results],
                indent=2,
            ),
        )
        log(
            f"Transfer report written to {transfer_report_path}",
            verbose=state.verbose,
            elapsed=state.elapsed(),
        )
        state.transfer_reports = ["transfer_report.json"]
    else:
        state.transfer_reports = []

    if config.measure.measure_only:
        if not state.measure_reports:
            state.measure_reports = write_measure_reports(
                config,
                direction_result=state.direction_result,
                subspace_result=state.subspace_result,
                dbdi_result=state.dbdi_result,
                diff_result=state.diff_result,
            )
        log(
            (
                "Measure-only mode complete — output written to"
                f" {config.output_dir}"
            ),
            verbose=state.verbose,
            elapsed=state.elapsed(),
        )
        write_experiment_log(
            state.config_path,
            config,
            "measure",
            state.measure_reports + state.transfer_reports,
            {},
            state.elapsed(),
        )
        return True
    return False
