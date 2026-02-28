"""Parse the [softprompt] section of a TOML config."""

from vauban.config._types import TomlDict
from vauban.types import SoftPromptConfig


def _parse_softprompt(raw: TomlDict) -> SoftPromptConfig | None:
    """Parse the optional [softprompt] section into a SoftPromptConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("softprompt")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[softprompt] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    mode_raw = sec.get("mode", "continuous")  # type: ignore[arg-type]
    if not isinstance(mode_raw, str):
        msg = (
            f"[softprompt].mode must be a string,"
            f" got {type(mode_raw).__name__}"
        )
        raise TypeError(msg)
    valid_modes = ("continuous", "gcg", "egd")
    if mode_raw not in valid_modes:
        msg = (
            f"[softprompt].mode must be one of {valid_modes!r},"
            f" got {mode_raw!r}"
        )
        raise ValueError(msg)

    n_tokens_raw = sec.get("n_tokens", 16)  # type: ignore[arg-type]
    if not isinstance(n_tokens_raw, int):
        msg = (
            f"[softprompt].n_tokens must be an integer,"
            f" got {type(n_tokens_raw).__name__}"
        )
        raise TypeError(msg)
    if n_tokens_raw < 1:
        msg = f"[softprompt].n_tokens must be >= 1, got {n_tokens_raw}"
        raise ValueError(msg)

    n_steps_raw = sec.get("n_steps", 200)  # type: ignore[arg-type]
    if not isinstance(n_steps_raw, int):
        msg = (
            f"[softprompt].n_steps must be an integer,"
            f" got {type(n_steps_raw).__name__}"
        )
        raise TypeError(msg)
    if n_steps_raw < 1:
        msg = f"[softprompt].n_steps must be >= 1, got {n_steps_raw}"
        raise ValueError(msg)

    lr_raw = sec.get("learning_rate", 0.01)  # type: ignore[arg-type]
    if not isinstance(lr_raw, int | float):
        msg = (
            f"[softprompt].learning_rate must be a number,"
            f" got {type(lr_raw).__name__}"
        )
        raise TypeError(msg)
    if float(lr_raw) <= 0:
        msg = f"[softprompt].learning_rate must be > 0, got {lr_raw}"
        raise ValueError(msg)

    init_scale_raw = sec.get("init_scale", 0.1)  # type: ignore[arg-type]
    if not isinstance(init_scale_raw, int | float):
        msg = (
            f"[softprompt].init_scale must be a number,"
            f" got {type(init_scale_raw).__name__}"
        )
        raise TypeError(msg)

    batch_size_raw = sec.get("batch_size", 64)  # type: ignore[arg-type]
    if not isinstance(batch_size_raw, int):
        msg = (
            f"[softprompt].batch_size must be an integer,"
            f" got {type(batch_size_raw).__name__}"
        )
        raise TypeError(msg)
    if batch_size_raw < 1:
        msg = f"[softprompt].batch_size must be >= 1, got {batch_size_raw}"
        raise ValueError(msg)

    top_k_raw = sec.get("top_k", 256)  # type: ignore[arg-type]
    if not isinstance(top_k_raw, int):
        msg = (
            f"[softprompt].top_k must be an integer,"
            f" got {type(top_k_raw).__name__}"
        )
        raise TypeError(msg)
    if top_k_raw < 1:
        msg = f"[softprompt].top_k must be >= 1, got {top_k_raw}"
        raise ValueError(msg)

    dw_raw = sec.get("direction_weight", 0.0)  # type: ignore[arg-type]
    if not isinstance(dw_raw, int | float):
        msg = (
            f"[softprompt].direction_weight must be a number,"
            f" got {type(dw_raw).__name__}"
        )
        raise TypeError(msg)
    if float(dw_raw) < 0.0:
        msg = (
            f"[softprompt].direction_weight must be >= 0.0, got {dw_raw}"
        )
        raise ValueError(msg)

    target_raw = sec.get(  # type: ignore[arg-type]
        "target_prefixes", ["Sure", "Here"],
    )
    if not isinstance(target_raw, list):
        msg = (
            f"[softprompt].target_prefixes must be a list,"
            f" got {type(target_raw).__name__}"
        )
        raise TypeError(msg)
    target_prefixes: list[str] = [str(t) for t in target_raw]

    max_gen_raw = sec.get("max_gen_tokens", 100)  # type: ignore[arg-type]
    if not isinstance(max_gen_raw, int):
        msg = (
            f"[softprompt].max_gen_tokens must be an integer,"
            f" got {type(max_gen_raw).__name__}"
        )
        raise TypeError(msg)

    seed_raw = sec.get("seed")  # type: ignore[arg-type]
    seed: int | None = None
    if seed_raw is not None:
        if not isinstance(seed_raw, int):
            msg = (
                f"[softprompt].seed must be an integer,"
                f" got {type(seed_raw).__name__}"
            )
            raise TypeError(msg)
        seed = seed_raw

    embed_reg_raw = sec.get("embed_reg_weight", 0.0)  # type: ignore[arg-type]
    if not isinstance(embed_reg_raw, int | float):
        msg = (
            f"[softprompt].embed_reg_weight must be a number,"
            f" got {type(embed_reg_raw).__name__}"
        )
        raise TypeError(msg)
    if float(embed_reg_raw) < 0.0:
        msg = (
            f"[softprompt].embed_reg_weight must be >= 0.0,"
            f" got {embed_reg_raw}"
        )
        raise ValueError(msg)

    patience_raw = sec.get("patience", 0)  # type: ignore[arg-type]
    if not isinstance(patience_raw, int):
        msg = (
            f"[softprompt].patience must be an integer,"
            f" got {type(patience_raw).__name__}"
        )
        raise TypeError(msg)
    if patience_raw < 0:
        msg = f"[softprompt].patience must be >= 0, got {patience_raw}"
        raise ValueError(msg)

    lr_schedule_raw = sec.get("lr_schedule", "constant")  # type: ignore[arg-type]
    if not isinstance(lr_schedule_raw, str):
        msg = (
            f"[softprompt].lr_schedule must be a string,"
            f" got {type(lr_schedule_raw).__name__}"
        )
        raise TypeError(msg)
    valid_lr_schedules = ("constant", "cosine")
    if lr_schedule_raw not in valid_lr_schedules:
        msg = (
            f"[softprompt].lr_schedule must be one of"
            f" {valid_lr_schedules!r}, got {lr_schedule_raw!r}"
        )
        raise ValueError(msg)

    n_restarts_raw = sec.get("n_restarts", 1)  # type: ignore[arg-type]
    if not isinstance(n_restarts_raw, int):
        msg = (
            f"[softprompt].n_restarts must be an integer,"
            f" got {type(n_restarts_raw).__name__}"
        )
        raise TypeError(msg)
    if n_restarts_raw < 1:
        msg = f"[softprompt].n_restarts must be >= 1, got {n_restarts_raw}"
        raise ValueError(msg)

    prompt_strategy_raw = sec.get("prompt_strategy", "all")  # type: ignore[arg-type]
    if not isinstance(prompt_strategy_raw, str):
        msg = (
            f"[softprompt].prompt_strategy must be a string,"
            f" got {type(prompt_strategy_raw).__name__}"
        )
        raise TypeError(msg)
    valid_prompt_strategies = ("all", "cycle", "first", "worst_k", "sample")
    if prompt_strategy_raw not in valid_prompt_strategies:
        msg = (
            f"[softprompt].prompt_strategy must be one of"
            f" {valid_prompt_strategies!r}, got {prompt_strategy_raw!r}"
        )
        raise ValueError(msg)

    # -- direction_mode --
    direction_mode_raw = sec.get("direction_mode", "last")  # type: ignore[arg-type]
    if not isinstance(direction_mode_raw, str):
        msg = (
            f"[softprompt].direction_mode must be a string,"
            f" got {type(direction_mode_raw).__name__}"
        )
        raise TypeError(msg)
    valid_direction_modes = ("last", "raid", "all_positions")
    if direction_mode_raw not in valid_direction_modes:
        msg = (
            f"[softprompt].direction_mode must be one of"
            f" {valid_direction_modes!r}, got {direction_mode_raw!r}"
        )
        raise ValueError(msg)

    # -- direction_layers --
    direction_layers_raw = sec.get("direction_layers")  # type: ignore[arg-type]
    direction_layers: list[int] | None = None
    if direction_layers_raw is not None:
        if not isinstance(direction_layers_raw, list):
            msg = (
                f"[softprompt].direction_layers must be a list of ints,"
                f" got {type(direction_layers_raw).__name__}"
            )
            raise TypeError(msg)
        direction_layers = [
            int(x) for x in direction_layers_raw if isinstance(x, int | float)
        ]

    # -- loss_mode --
    loss_mode_raw = sec.get("loss_mode", "targeted")  # type: ignore[arg-type]
    if not isinstance(loss_mode_raw, str):
        msg = (
            f"[softprompt].loss_mode must be a string,"
            f" got {type(loss_mode_raw).__name__}"
        )
        raise TypeError(msg)
    valid_loss_modes = ("targeted", "untargeted", "defensive")
    if loss_mode_raw not in valid_loss_modes:
        msg = (
            f"[softprompt].loss_mode must be one of"
            f" {valid_loss_modes!r}, got {loss_mode_raw!r}"
        )
        raise ValueError(msg)

    # -- egd_temperature --
    egd_temp_raw = sec.get("egd_temperature", 1.0)  # type: ignore[arg-type]
    if not isinstance(egd_temp_raw, int | float):
        msg = (
            f"[softprompt].egd_temperature must be a number,"
            f" got {type(egd_temp_raw).__name__}"
        )
        raise TypeError(msg)
    if float(egd_temp_raw) <= 0:
        msg = (
            f"[softprompt].egd_temperature must be > 0, got {egd_temp_raw}"
        )
        raise ValueError(msg)

    # -- token_constraint --
    token_constraint_raw = sec.get("token_constraint")  # type: ignore[arg-type]
    token_constraint: str | list[str] | None = None
    valid_constraints = (
        "ascii", "alpha", "alphanumeric",
        "non_latin", "chinese", "non_alphabetic",
        "invisible", "zalgo", "emoji",
    )
    if token_constraint_raw is not None:
        if isinstance(token_constraint_raw, str):
            if token_constraint_raw not in valid_constraints:
                msg = (
                    f"[softprompt].token_constraint must be one of"
                    f" {valid_constraints!r},"
                    f" got {token_constraint_raw!r}"
                )
                raise ValueError(msg)
            token_constraint = token_constraint_raw
        elif isinstance(token_constraint_raw, list):
            for item in token_constraint_raw:
                if not isinstance(item, str):
                    msg = (
                        "[softprompt].token_constraint list"
                        " elements must be strings,"
                        f" got {type(item).__name__}"
                    )
                    raise TypeError(msg)
                if item not in valid_constraints:
                    msg = (
                        "[softprompt].token_constraint"
                        f" element {item!r} is not one of"
                        f" {valid_constraints!r}"
                    )
                    raise ValueError(msg)
            token_constraint = [str(x) for x in token_constraint_raw]
        else:
            msg = (
                "[softprompt].token_constraint must be a string"
                " or list of strings,"
                f" got {type(token_constraint_raw).__name__}"
            )
            raise TypeError(msg)

    # -- eos_loss_mode --
    eos_loss_mode_raw = sec.get("eos_loss_mode", "none")  # type: ignore[arg-type]
    if not isinstance(eos_loss_mode_raw, str):
        msg = (
            f"[softprompt].eos_loss_mode must be a string,"
            f" got {type(eos_loss_mode_raw).__name__}"
        )
        raise TypeError(msg)
    valid_eos_modes = ("none", "force", "suppress")
    if eos_loss_mode_raw not in valid_eos_modes:
        msg = (
            f"[softprompt].eos_loss_mode must be one of"
            f" {valid_eos_modes!r}, got {eos_loss_mode_raw!r}"
        )
        raise ValueError(msg)

    # -- eos_loss_weight --
    eos_loss_weight_raw = sec.get("eos_loss_weight", 0.0)  # type: ignore[arg-type]
    if not isinstance(eos_loss_weight_raw, int | float):
        msg = (
            f"[softprompt].eos_loss_weight must be a number,"
            f" got {type(eos_loss_weight_raw).__name__}"
        )
        raise TypeError(msg)
    if float(eos_loss_weight_raw) < 0.0:
        msg = (
            f"[softprompt].eos_loss_weight must be >= 0.0,"
            f" got {eos_loss_weight_raw}"
        )
        raise ValueError(msg)

    # -- kl_ref_weight --
    kl_ref_weight_raw = sec.get("kl_ref_weight", 0.0)  # type: ignore[arg-type]
    if not isinstance(kl_ref_weight_raw, int | float):
        msg = (
            f"[softprompt].kl_ref_weight must be a number,"
            f" got {type(kl_ref_weight_raw).__name__}"
        )
        raise TypeError(msg)
    if float(kl_ref_weight_raw) < 0.0:
        msg = (
            f"[softprompt].kl_ref_weight must be >= 0.0,"
            f" got {kl_ref_weight_raw}"
        )
        raise ValueError(msg)

    # -- worst_k --
    worst_k_raw = sec.get("worst_k", 5)  # type: ignore[arg-type]
    if not isinstance(worst_k_raw, int):
        msg = (
            f"[softprompt].worst_k must be an integer,"
            f" got {type(worst_k_raw).__name__}"
        )
        raise TypeError(msg)
    if worst_k_raw < 1:
        msg = f"[softprompt].worst_k must be >= 1, got {worst_k_raw}"
        raise ValueError(msg)

    # -- grad_accum_steps --
    grad_accum_steps_raw = sec.get("grad_accum_steps", 1)  # type: ignore[arg-type]
    if not isinstance(grad_accum_steps_raw, int):
        msg = (
            f"[softprompt].grad_accum_steps must be an integer,"
            f" got {type(grad_accum_steps_raw).__name__}"
        )
        raise TypeError(msg)
    if grad_accum_steps_raw < 1:
        msg = (
            f"[softprompt].grad_accum_steps must be >= 1,"
            f" got {grad_accum_steps_raw}"
        )
        raise ValueError(msg)

    # -- transfer_models --
    transfer_models_raw = sec.get(  # type: ignore[arg-type]
        "transfer_models", [],
    )
    if not isinstance(transfer_models_raw, list):
        msg = (
            f"[softprompt].transfer_models must be a list,"
            f" got {type(transfer_models_raw).__name__}"
        )
        raise TypeError(msg)
    transfer_models: list[str] = [str(m) for m in transfer_models_raw]

    # -- ref_model --
    ref_model_raw = sec.get("ref_model")  # type: ignore[arg-type]
    ref_model: str | None = None
    if ref_model_raw is not None:
        if not isinstance(ref_model_raw, str):
            msg = (
                f"[softprompt].ref_model must be a string,"
                f" got {type(ref_model_raw).__name__}"
            )
            raise TypeError(msg)
        ref_model = ref_model_raw

    # Validate: kl_ref_weight > 0 requires ref_model
    if float(kl_ref_weight_raw) > 0.0 and ref_model is None:
        msg = (
            "[softprompt].kl_ref_weight > 0 requires"
            " [softprompt].ref_model to be set"
        )
        raise ValueError(msg)

    # -- target_repeat_count --
    target_repeat_count_raw = sec.get(  # type: ignore[arg-type]
        "target_repeat_count", 0,
    )
    if not isinstance(target_repeat_count_raw, int):
        msg = (
            f"[softprompt].target_repeat_count must be an integer,"
            f" got {type(target_repeat_count_raw).__name__}"
        )
        raise TypeError(msg)
    if target_repeat_count_raw < 0:
        msg = (
            f"[softprompt].target_repeat_count must be >= 0,"
            f" got {target_repeat_count_raw}"
        )
        raise ValueError(msg)

    # -- system_prompt --
    system_prompt_raw = sec.get("system_prompt")  # type: ignore[arg-type]
    system_prompt: str | None = None
    if system_prompt_raw is not None:
        if not isinstance(system_prompt_raw, str):
            msg = (
                f"[softprompt].system_prompt must be a string,"
                f" got {type(system_prompt_raw).__name__}"
            )
            raise TypeError(msg)
        system_prompt = system_prompt_raw

    # -- defense_eval --
    defense_eval_raw = sec.get("defense_eval")  # type: ignore[arg-type]
    defense_eval: str | None = None
    if defense_eval_raw is not None:
        if not isinstance(defense_eval_raw, str):
            msg = (
                f"[softprompt].defense_eval must be a string,"
                f" got {type(defense_eval_raw).__name__}"
            )
            raise TypeError(msg)
        valid_defense_evals = ("sic", "cast", "both")
        if defense_eval_raw not in valid_defense_evals:
            msg = (
                f"[softprompt].defense_eval must be one of"
                f" {valid_defense_evals!r},"
                f" got {defense_eval_raw!r}"
            )
            raise ValueError(msg)
        defense_eval = defense_eval_raw

    # -- defense_eval_layer --
    defense_eval_layer_raw = sec.get(  # type: ignore[arg-type]
        "defense_eval_layer",
    )
    defense_eval_layer: int | None = None
    if defense_eval_layer_raw is not None:
        if not isinstance(defense_eval_layer_raw, int):
            msg = (
                "[softprompt].defense_eval_layer must be"
                f" an integer, got"
                f" {type(defense_eval_layer_raw).__name__}"
            )
            raise TypeError(msg)
        defense_eval_layer = defense_eval_layer_raw

    # -- defense_eval_alpha --
    defense_eval_alpha_raw = sec.get(  # type: ignore[arg-type]
        "defense_eval_alpha", 1.0,
    )
    if not isinstance(defense_eval_alpha_raw, int | float):
        msg = (
            "[softprompt].defense_eval_alpha must be"
            f" a number, got"
            f" {type(defense_eval_alpha_raw).__name__}"
        )
        raise TypeError(msg)

    # -- defense_eval_threshold --
    defense_eval_threshold_raw = sec.get(  # type: ignore[arg-type]
        "defense_eval_threshold", 0.0,
    )
    if not isinstance(defense_eval_threshold_raw, int | float):
        msg = (
            "[softprompt].defense_eval_threshold must be"
            f" a number, got"
            f" {type(defense_eval_threshold_raw).__name__}"
        )
        raise TypeError(msg)

    # -- defense_eval_sic_mode --
    sic_mode_raw = sec.get(  # type: ignore[arg-type]
        "defense_eval_sic_mode", "direction",
    )
    if not isinstance(sic_mode_raw, str):
        msg = (
            "[softprompt].defense_eval_sic_mode must be"
            f" a string, got {type(sic_mode_raw).__name__}"
        )
        raise TypeError(msg)
    valid_sic_modes = ("direction", "generation")
    if sic_mode_raw not in valid_sic_modes:
        msg = (
            "[softprompt].defense_eval_sic_mode must be"
            f" one of {valid_sic_modes!r},"
            f" got {sic_mode_raw!r}"
        )
        raise ValueError(msg)

    # -- defense_eval_sic_max_iterations --
    sic_max_iter_raw = sec.get(  # type: ignore[arg-type]
        "defense_eval_sic_max_iterations", 3,
    )
    if not isinstance(sic_max_iter_raw, int):
        msg = (
            "[softprompt].defense_eval_sic_max_iterations"
            " must be an integer,"
            f" got {type(sic_max_iter_raw).__name__}"
        )
        raise TypeError(msg)
    if sic_max_iter_raw < 1:
        msg = (
            "[softprompt].defense_eval_sic_max_iterations"
            f" must be >= 1, got {sic_max_iter_raw}"
        )
        raise ValueError(msg)

    # -- defense_eval_cast_layers --
    cast_layers_raw = sec.get(  # type: ignore[arg-type]
        "defense_eval_cast_layers",
    )
    defense_eval_cast_layers: list[int] | None = None
    if cast_layers_raw is not None:
        if not isinstance(cast_layers_raw, list):
            msg = (
                "[softprompt].defense_eval_cast_layers"
                " must be a list of ints,"
                f" got {type(cast_layers_raw).__name__}"
            )
            raise TypeError(msg)
        defense_eval_cast_layers = [
            int(x) for x in cast_layers_raw
            if isinstance(x, int | float)
        ]

    # -- gan_rounds --
    gan_rounds_raw = sec.get("gan_rounds", 0)  # type: ignore[arg-type]
    if not isinstance(gan_rounds_raw, int):
        msg = (
            f"[softprompt].gan_rounds must be an integer,"
            f" got {type(gan_rounds_raw).__name__}"
        )
        raise TypeError(msg)
    if gan_rounds_raw < 0:
        msg = (
            f"[softprompt].gan_rounds must be >= 0,"
            f" got {gan_rounds_raw}"
        )
        raise ValueError(msg)

    # -- gan_step_multiplier --
    gan_step_mult_raw = sec.get(  # type: ignore[arg-type]
        "gan_step_multiplier", 1.5,
    )
    if not isinstance(gan_step_mult_raw, int | float):
        msg = (
            "[softprompt].gan_step_multiplier must be"
            f" a number, got"
            f" {type(gan_step_mult_raw).__name__}"
        )
        raise TypeError(msg)
    if float(gan_step_mult_raw) <= 0:
        msg = (
            "[softprompt].gan_step_multiplier must be"
            f" > 0, got {gan_step_mult_raw}"
        )
        raise ValueError(msg)

    # -- gan_direction_escalation --
    gan_dir_esc_raw = sec.get(  # type: ignore[arg-type]
        "gan_direction_escalation", 0.25,
    )
    if not isinstance(gan_dir_esc_raw, int | float):
        msg = (
            "[softprompt].gan_direction_escalation must"
            f" be a number, got"
            f" {type(gan_dir_esc_raw).__name__}"
        )
        raise TypeError(msg)

    # -- gan_token_escalation --
    gan_tok_esc_raw = sec.get(  # type: ignore[arg-type]
        "gan_token_escalation", 4,
    )
    if not isinstance(gan_tok_esc_raw, int):
        msg = (
            "[softprompt].gan_token_escalation must be"
            f" an integer, got"
            f" {type(gan_tok_esc_raw).__name__}"
        )
        raise TypeError(msg)
    if gan_tok_esc_raw < 0:
        msg = (
            "[softprompt].gan_token_escalation must be"
            f" >= 0, got {gan_tok_esc_raw}"
        )
        raise ValueError(msg)

    # -- gan_defense_escalation --
    gan_def_esc_raw = sec.get(  # type: ignore[arg-type]
        "gan_defense_escalation", False,
    )
    if not isinstance(gan_def_esc_raw, bool):
        msg = (
            "[softprompt].gan_defense_escalation must be"
            f" a boolean, got"
            f" {type(gan_def_esc_raw).__name__}"
        )
        raise TypeError(msg)

    # -- gan_defense_alpha_multiplier --
    gan_def_alpha_raw = sec.get(  # type: ignore[arg-type]
        "gan_defense_alpha_multiplier", 1.5,
    )
    if not isinstance(gan_def_alpha_raw, int | float):
        msg = (
            "[softprompt].gan_defense_alpha_multiplier must be"
            f" a number, got"
            f" {type(gan_def_alpha_raw).__name__}"
        )
        raise TypeError(msg)
    if float(gan_def_alpha_raw) <= 0:
        msg = (
            "[softprompt].gan_defense_alpha_multiplier must be"
            f" > 0, got {gan_def_alpha_raw}"
        )
        raise ValueError(msg)

    # -- gan_defense_threshold_escalation --
    gan_def_thresh_raw = sec.get(  # type: ignore[arg-type]
        "gan_defense_threshold_escalation", 0.5,
    )
    if not isinstance(gan_def_thresh_raw, int | float):
        msg = (
            "[softprompt].gan_defense_threshold_escalation"
            f" must be a number, got"
            f" {type(gan_def_thresh_raw).__name__}"
        )
        raise TypeError(msg)
    if float(gan_def_thresh_raw) < 0:
        msg = (
            "[softprompt].gan_defense_threshold_escalation"
            f" must be >= 0, got {gan_def_thresh_raw}"
        )
        raise ValueError(msg)

    # -- gan_defense_sic_iteration_escalation --
    gan_def_sic_raw = sec.get(  # type: ignore[arg-type]
        "gan_defense_sic_iteration_escalation", 1,
    )
    if not isinstance(gan_def_sic_raw, int):
        msg = (
            "[softprompt].gan_defense_sic_iteration_escalation"
            f" must be an integer, got"
            f" {type(gan_def_sic_raw).__name__}"
        )
        raise TypeError(msg)
    if gan_def_sic_raw < 0:
        msg = (
            "[softprompt].gan_defense_sic_iteration_escalation"
            f" must be >= 0, got {gan_def_sic_raw}"
        )
        raise ValueError(msg)

    # -- gan_multiturn --
    gan_multiturn_raw = sec.get("gan_multiturn", False)  # type: ignore[arg-type]
    if not isinstance(gan_multiturn_raw, bool):
        msg = (
            "[softprompt].gan_multiturn must be"
            f" a boolean, got"
            f" {type(gan_multiturn_raw).__name__}"
        )
        raise TypeError(msg)

    # -- gan_multiturn_max_turns --
    gan_mt_max_raw = sec.get(  # type: ignore[arg-type]
        "gan_multiturn_max_turns", 10,
    )
    if not isinstance(gan_mt_max_raw, int):
        msg = (
            "[softprompt].gan_multiturn_max_turns must be"
            f" an integer, got"
            f" {type(gan_mt_max_raw).__name__}"
        )
        raise TypeError(msg)
    if gan_mt_max_raw < 1:
        msg = (
            "[softprompt].gan_multiturn_max_turns must be"
            f" >= 1, got {gan_mt_max_raw}"
        )
        raise ValueError(msg)

    # -- prompt_pool_size --
    prompt_pool_size_raw = sec.get("prompt_pool_size")  # type: ignore[arg-type]
    prompt_pool_size: int | None = None
    if prompt_pool_size_raw is not None:
        if not isinstance(prompt_pool_size_raw, int):
            msg = (
                "[softprompt].prompt_pool_size must be"
                f" an integer, got"
                f" {type(prompt_pool_size_raw).__name__}"
            )
            raise TypeError(msg)
        if prompt_pool_size_raw < 1:
            msg = (
                "[softprompt].prompt_pool_size must be"
                f" >= 1, got {prompt_pool_size_raw}"
            )
            raise ValueError(msg)
        prompt_pool_size = prompt_pool_size_raw

    # -- defense_aware_weight --
    defense_aware_weight_raw = sec.get(  # type: ignore[arg-type]
        "defense_aware_weight", 0.0,
    )
    if not isinstance(defense_aware_weight_raw, int | float):
        msg = (
            "[softprompt].defense_aware_weight must be"
            f" a number, got"
            f" {type(defense_aware_weight_raw).__name__}"
        )
        raise TypeError(msg)
    if float(defense_aware_weight_raw) < 0.0:
        msg = (
            "[softprompt].defense_aware_weight must be"
            f" >= 0.0, got {defense_aware_weight_raw}"
        )
        raise ValueError(msg)

    # -- transfer_loss_weight --
    transfer_loss_weight_raw = sec.get(  # type: ignore[arg-type]
        "transfer_loss_weight", 0.0,
    )
    if not isinstance(transfer_loss_weight_raw, int | float):
        msg = (
            "[softprompt].transfer_loss_weight must be"
            f" a number, got"
            f" {type(transfer_loss_weight_raw).__name__}"
        )
        raise TypeError(msg)
    if float(transfer_loss_weight_raw) < 0.0:
        msg = (
            "[softprompt].transfer_loss_weight must be"
            f" >= 0.0, got {transfer_loss_weight_raw}"
        )
        raise ValueError(msg)

    # -- transfer_rerank_count --
    transfer_rerank_count_raw = sec.get(  # type: ignore[arg-type]
        "transfer_rerank_count", 8,
    )
    if not isinstance(transfer_rerank_count_raw, int):
        msg = (
            "[softprompt].transfer_rerank_count must be"
            f" an integer, got"
            f" {type(transfer_rerank_count_raw).__name__}"
        )
        raise TypeError(msg)
    if transfer_rerank_count_raw < 1:
        msg = (
            "[softprompt].transfer_rerank_count must be"
            f" >= 1, got {transfer_rerank_count_raw}"
        )
        raise ValueError(msg)

    # -- defense_eval_alpha_tiers --
    alpha_tiers_raw = sec.get(  # type: ignore[arg-type]
        "defense_eval_alpha_tiers",
    )
    defense_eval_alpha_tiers: list[tuple[float, float]] | None = None
    if alpha_tiers_raw is not None:
        if not isinstance(alpha_tiers_raw, list):
            msg = (
                "[softprompt].defense_eval_alpha_tiers"
                " must be a list of [threshold, alpha] pairs,"
                f" got {type(alpha_tiers_raw).__name__}"
            )
            raise TypeError(msg)
        defense_eval_alpha_tiers = []
        for i, pair in enumerate(alpha_tiers_raw):
            if (
                not isinstance(pair, list)
                or len(pair) != 2
            ):
                msg = (
                    "[softprompt].defense_eval_alpha_tiers"
                    f"[{i}] must be a [threshold, alpha] pair"
                )
                raise ValueError(msg)
            if not isinstance(pair[0], int | float) or not isinstance(
                pair[1], int | float,
            ):
                msg = (
                    "[softprompt].defense_eval_alpha_tiers"
                    f"[{i}] values must be numbers"
                )
                raise TypeError(msg)
            defense_eval_alpha_tiers.append(
                (float(pair[0]), float(pair[1])),
            )

    # -- beam_width --
    beam_width_raw = sec.get("beam_width", 1)  # type: ignore[arg-type]
    if not isinstance(beam_width_raw, int):
        msg = (
            "[softprompt].beam_width must be"
            f" an integer, got"
            f" {type(beam_width_raw).__name__}"
        )
        raise TypeError(msg)
    if beam_width_raw < 1:
        msg = (
            "[softprompt].beam_width must be"
            f" >= 1, got {beam_width_raw}"
        )
        raise ValueError(msg)

    # -- init_tokens --
    init_tokens_raw = sec.get("init_tokens")  # type: ignore[arg-type]
    init_tokens: list[int] | None = None
    if init_tokens_raw is not None:
        if not isinstance(init_tokens_raw, list):
            msg = (
                "[softprompt].init_tokens must be a list"
                f" of ints, got"
                f" {type(init_tokens_raw).__name__}"
            )
            raise TypeError(msg)
        init_tokens = [int(x) for x in init_tokens_raw]

    return SoftPromptConfig(
        mode=mode_raw,
        n_tokens=n_tokens_raw,
        n_steps=n_steps_raw,
        learning_rate=float(lr_raw),
        init_scale=float(init_scale_raw),
        batch_size=batch_size_raw,
        top_k=top_k_raw,
        direction_weight=float(dw_raw),
        target_prefixes=target_prefixes,
        max_gen_tokens=max_gen_raw,
        seed=seed,
        embed_reg_weight=float(embed_reg_raw),
        patience=patience_raw,
        lr_schedule=lr_schedule_raw,
        n_restarts=n_restarts_raw,
        prompt_strategy=prompt_strategy_raw,
        direction_mode=direction_mode_raw,
        direction_layers=direction_layers,
        loss_mode=loss_mode_raw,
        egd_temperature=float(egd_temp_raw),
        token_constraint=token_constraint,
        eos_loss_mode=eos_loss_mode_raw,
        eos_loss_weight=float(eos_loss_weight_raw),
        kl_ref_weight=float(kl_ref_weight_raw),
        ref_model=ref_model,
        worst_k=worst_k_raw,
        grad_accum_steps=grad_accum_steps_raw,
        transfer_models=transfer_models,
        target_repeat_count=target_repeat_count_raw,
        system_prompt=system_prompt,
        defense_eval=defense_eval,
        defense_eval_layer=defense_eval_layer,
        defense_eval_alpha=float(defense_eval_alpha_raw),
        defense_eval_threshold=float(defense_eval_threshold_raw),
        defense_eval_sic_mode=sic_mode_raw,
        defense_eval_sic_max_iterations=sic_max_iter_raw,
        defense_eval_cast_layers=defense_eval_cast_layers,
        gan_rounds=gan_rounds_raw,
        gan_step_multiplier=float(gan_step_mult_raw),
        gan_direction_escalation=float(gan_dir_esc_raw),
        gan_token_escalation=gan_tok_esc_raw,
        gan_defense_escalation=gan_def_esc_raw,
        gan_defense_alpha_multiplier=float(gan_def_alpha_raw),
        gan_defense_threshold_escalation=float(gan_def_thresh_raw),
        gan_defense_sic_iteration_escalation=gan_def_sic_raw,
        gan_multiturn=gan_multiturn_raw,
        gan_multiturn_max_turns=gan_mt_max_raw,
        prompt_pool_size=prompt_pool_size,
        beam_width=beam_width_raw,
        defense_aware_weight=float(defense_aware_weight_raw),
        transfer_loss_weight=float(transfer_loss_weight_raw),
        transfer_rerank_count=transfer_rerank_count_raw,
        defense_eval_alpha_tiers=defense_eval_alpha_tiers,
        init_tokens=init_tokens,
    )
