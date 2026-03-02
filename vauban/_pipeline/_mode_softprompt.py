"""Softprompt early-mode runner."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing import cast as _cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._helpers import write_arena_card
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report
from vauban._serializers import _softprompt_to_dict
from vauban.types import SoftPromptResult, TransferEvalResult

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import CausalLM, Tokenizer


def _run_softprompt_mode(context: EarlyModeContext) -> None:
    """Run [softprompt] early-return mode and write its report."""
    from vauban import _ops as ops
    from vauban._forward import force_eval
    from vauban._forward import get_transformer as _get_transformer
    from vauban._model_io import load_model
    from vauban.dequantize import dequantize_model, is_quantized
    from vauban.measure import load_prompts
    from vauban.softprompt import _project_to_tokens, softprompt_attack

    config = context.config
    if config.softprompt is None:
        msg = "softprompt config is required for softprompt mode"
        raise ValueError(msg)
    if context.harmful is None:
        msg = "harmful prompts are required for softprompt mode but were not loaded"
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    log(
        f"Running soft prompt attack (mode={config.softprompt.mode})",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    direction_vec: Array | None = (
        context.direction_result.direction
        if context.direction_result is not None
        else None
    )

    if (
        config.softprompt.loss_mode == "externality"
        and config.softprompt.externality_target is not None
    ):
        loaded = ops.load(config.softprompt.externality_target)
        if isinstance(loaded, dict):
            loaded_dict = _cast("dict[str, Array]", loaded)
            direction_vec = next(iter(loaded_dict.values()))
        elif isinstance(loaded, tuple):
            msg = "Expected array or dict from externality target, got tuple"
            raise TypeError(msg)
        else:
            direction_vec = loaded

    if config.eval.prompts_path is not None:
        sp_prompts: list[str] = load_prompts(config.eval.prompts_path)
    else:
        pool_size = (
            config.softprompt.prompt_pool_size
            if config.softprompt.prompt_pool_size is not None
            else config.eval.num_prompts
        )
        sp_prompts = context.harmful[:pool_size]

    ref_model: CausalLM | None = None
    if config.softprompt.ref_model is not None:
        ref_model, _ = load_model(config.softprompt.ref_model)
        if is_quantized(ref_model):
            dequantize_model(ref_model)

    transfer_models_loaded: list[tuple[str, CausalLM, Tokenizer]] | None = None
    if config.softprompt.transfer_models:
        transfer_models_loaded = []
        for transfer_model_id in config.softprompt.transfer_models:
            transfer_model, transfer_tokenizer = load_model(transfer_model_id)
            if is_quantized(transfer_model):
                dequantize_model(transfer_model)
            transfer_models_loaded.append(
                (transfer_model_id, transfer_model, transfer_tokenizer),
            )

    # Load SVF boundary if configured
    svf_boundary = None
    if config.softprompt.svf_boundary_path is not None:
        from vauban.svf import load_svf_boundary

        svf_boundary = load_svf_boundary(config.softprompt.svf_boundary_path)
        log(
            f"Loaded SVF boundary from {config.softprompt.svf_boundary_path}",
            verbose=v,
            elapsed=time.monotonic() - context.t0,
        )

    sp_result = softprompt_attack(
        model,
        tokenizer,
        sp_prompts,
        config.softprompt,
        direction_vec,
        ref_model,
        transfer_models=transfer_models_loaded,
        api_eval_config=config.api_eval,
        environment_config=config.environment,
        svf_boundary=svf_boundary,
    )

    if transfer_models_loaded and not config.softprompt.gan_rounds:
        from vauban.softprompt import _evaluate_attack

        if sp_result.token_ids is not None:
            transfer_token_ids = sp_result.token_ids
        elif sp_result.embeddings is not None:
            transfer_token_ids = _project_to_tokens(
                sp_result.embeddings,
                _get_transformer(model).embed_tokens.weight,
            )
        else:
            transfer_token_ids = []

        transfer_results: list[TransferEvalResult] = []
        for name, transfer_model, transfer_tokenizer in transfer_models_loaded:
            token_array = ops.array(transfer_token_ids)[None, :]
            embeds = _get_transformer(transfer_model).embed_tokens(token_array)
            force_eval(embeds)
            success_rate, eval_responses = _evaluate_attack(
                transfer_model,
                transfer_tokenizer,
                sp_prompts,
                embeds,
                config.softprompt,
            )
            transfer_results.append(
                TransferEvalResult(
                    model_id=name,
                    success_rate=success_rate,
                    eval_responses=eval_responses,
                ),
            )

        sp_result = SoftPromptResult(
            mode=sp_result.mode,
            success_rate=sp_result.success_rate,
            final_loss=sp_result.final_loss,
            loss_history=sp_result.loss_history,
            n_steps=sp_result.n_steps,
            n_tokens=sp_result.n_tokens,
            embeddings=sp_result.embeddings,
            token_ids=sp_result.token_ids,
            token_text=sp_result.token_text,
            eval_responses=sp_result.eval_responses,
            accessibility_score=sp_result.accessibility_score,
            per_prompt_losses=sp_result.per_prompt_losses,
            early_stopped=sp_result.early_stopped,
            transfer_results=transfer_results,
            defense_eval=sp_result.defense_eval,
            gan_history=sp_result.gan_history,
        )

    if config.api_eval and sp_result.token_text:
        from vauban.api_eval import evaluate_suffix_via_api

        api_results = evaluate_suffix_via_api(
            sp_result.token_text,
            sp_prompts,
            config.api_eval,
            config.softprompt.system_prompt if config.softprompt else None,
        )
        all_transfer = [*sp_result.transfer_results, *api_results]
        sp_result = SoftPromptResult(
            mode=sp_result.mode,
            success_rate=sp_result.success_rate,
            final_loss=sp_result.final_loss,
            loss_history=sp_result.loss_history,
            n_steps=sp_result.n_steps,
            n_tokens=sp_result.n_tokens,
            embeddings=sp_result.embeddings,
            token_ids=sp_result.token_ids,
            token_text=sp_result.token_text,
            eval_responses=sp_result.eval_responses,
            accessibility_score=sp_result.accessibility_score,
            per_prompt_losses=sp_result.per_prompt_losses,
            early_stopped=sp_result.early_stopped,
            transfer_results=all_transfer,
            defense_eval=sp_result.defense_eval,
            gan_history=sp_result.gan_history,
        )
        for api_result in api_results:
            log(
                f"API eval {api_result.model_id}: {api_result.success_rate:.2%}",
                verbose=v,
                elapsed=time.monotonic() - context.t0,
            )

    report_path = write_mode_report(
        config.output_dir,
        ModeReport("softprompt_report.json", _softprompt_to_dict(sp_result)),
    )

    report_files = ["softprompt_report.json"]
    if sp_result.token_text:
        arena_path = config.output_dir / "arena_card.txt"
        write_arena_card(arena_path, sp_result, sp_prompts)
        report_files.append("arena_card.txt")
        log(
            f"Arena card written to {arena_path}",
            verbose=v,
            elapsed=time.monotonic() - context.t0,
        )

    log(
        f"Done — softprompt report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    finish_mode_run(
        context,
        "softprompt",
        report_files,
        {"success_rate": sp_result.success_rate},
    )
