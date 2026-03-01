"""Parse the [cast] section of a TOML config."""

from vauban.config._types import TomlDict
from vauban.types import AlphaTier, CastConfig


def _parse_cast(raw: TomlDict) -> CastConfig | None:
    """Parse the optional [cast] section into a CastConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("cast")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[cast] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    # -- prompts (required) --
    prompts_raw = sec.get("prompts")  # type: ignore[arg-type]
    if prompts_raw is None:
        msg = "[cast].prompts is required"
        raise ValueError(msg)
    if not isinstance(prompts_raw, list):
        msg = (
            f"[cast].prompts must be a list of strings,"
            f" got {type(prompts_raw).__name__}"
        )
        raise TypeError(msg)
    if len(prompts_raw) == 0:
        msg = "[cast].prompts must be non-empty"
        raise ValueError(msg)
    for i, item in enumerate(prompts_raw):
        if not isinstance(item, str):
            msg = (
                f"[cast].prompts[{i}] must be a string,"
                f" got {type(item).__name__}"
            )
            raise TypeError(msg)

    # -- layers (optional) --
    layers_raw = sec.get("layers")  # type: ignore[arg-type]
    layers: list[int] | None = None
    if layers_raw is not None:
        if not isinstance(layers_raw, list):
            msg = (
                f"[cast].layers must be a list of integers,"
                f" got {type(layers_raw).__name__}"
            )
            raise TypeError(msg)
        for i, item in enumerate(layers_raw):
            if not isinstance(item, int):
                msg = (
                    f"[cast].layers[{i}] must be an integer,"
                    f" got {type(item).__name__}"
                )
                raise TypeError(msg)
        layers = list(layers_raw)

    # -- alpha (optional, default 1.0) --
    alpha_raw = sec.get("alpha", 1.0)  # type: ignore[arg-type]
    if not isinstance(alpha_raw, int | float):
        msg = (
            f"[cast].alpha must be a number,"
            f" got {type(alpha_raw).__name__}"
        )
        raise TypeError(msg)

    # -- threshold (optional, default 0.0) --
    threshold_raw = sec.get("threshold", 0.0)  # type: ignore[arg-type]
    if not isinstance(threshold_raw, int | float):
        msg = (
            f"[cast].threshold must be a number,"
            f" got {type(threshold_raw).__name__}"
        )
        raise TypeError(msg)

    # -- max_tokens (optional, default 100) --
    max_tokens_raw = sec.get("max_tokens", 100)  # type: ignore[arg-type]
    if not isinstance(max_tokens_raw, int):
        msg = (
            f"[cast].max_tokens must be an integer,"
            f" got {type(max_tokens_raw).__name__}"
        )
        raise TypeError(msg)
    if max_tokens_raw < 1:
        msg = f"[cast].max_tokens must be >= 1, got {max_tokens_raw}"
        raise ValueError(msg)

    # -- condition_direction (optional) --
    condition_direction_raw = sec.get("condition_direction")  # type: ignore[arg-type]
    condition_direction: str | None = None
    if condition_direction_raw is not None:
        if not isinstance(condition_direction_raw, str):
            msg = (
                f"[cast].condition_direction must be a string,"
                f" got {type(condition_direction_raw).__name__}"
            )
            raise TypeError(msg)
        condition_direction = condition_direction_raw

    # -- alpha_tiers (optional) --
    alpha_tiers_raw = sec.get("alpha_tiers")  # type: ignore[arg-type]
    alpha_tiers: list[AlphaTier] | None = None
    if alpha_tiers_raw is not None:
        if not isinstance(alpha_tiers_raw, list):
            msg = (
                f"[cast].alpha_tiers must be a list of tables,"
                f" got {type(alpha_tiers_raw).__name__}"
            )
            raise TypeError(msg)
        alpha_tiers = []
        for i, tier_raw in enumerate(alpha_tiers_raw):
            if not isinstance(tier_raw, dict):
                msg = (
                    f"[cast].alpha_tiers[{i}] must be a table,"
                    f" got {type(tier_raw).__name__}"
                )
                raise TypeError(msg)
            t_threshold = tier_raw.get("threshold")
            t_alpha = tier_raw.get("alpha")
            if t_threshold is None or t_alpha is None:
                msg = (
                    f"[cast].alpha_tiers[{i}] must have 'threshold'"
                    f" and 'alpha' keys"
                )
                raise ValueError(msg)
            if not isinstance(t_threshold, int | float):
                msg = (
                    f"[cast].alpha_tiers[{i}].threshold must be a number,"
                    f" got {type(t_threshold).__name__}"
                )
                raise TypeError(msg)
            if not isinstance(t_alpha, int | float):
                msg = (
                    f"[cast].alpha_tiers[{i}].alpha must be a number,"
                    f" got {type(t_alpha).__name__}"
                )
                raise TypeError(msg)
            alpha_tiers.append(
                AlphaTier(threshold=float(t_threshold), alpha=float(t_alpha)),
            )

        # Validate tiers are sorted by ascending threshold
        for i in range(1, len(alpha_tiers)):
            if alpha_tiers[i].threshold < alpha_tiers[i - 1].threshold:
                msg = (
                    "[cast].alpha_tiers must be sorted by ascending threshold"
                )
                raise ValueError(msg)

    # -- direction_source (optional, default "linear") --
    direction_source_raw = sec.get("direction_source", "linear")  # type: ignore[arg-type]
    if not isinstance(direction_source_raw, str):
        msg = (
            f"[cast].direction_source must be a string,"
            f" got {type(direction_source_raw).__name__}"
        )
        raise TypeError(msg)
    if direction_source_raw not in ("linear", "svf"):
        msg = (
            f"[cast].direction_source must be 'linear' or 'svf',"
            f" got {direction_source_raw!r}"
        )
        raise ValueError(msg)

    # -- svf_boundary_path (optional, required when direction_source="svf") --
    svf_path_raw = sec.get("svf_boundary_path")  # type: ignore[arg-type]
    svf_boundary_path: str | None = None
    if svf_path_raw is not None:
        if not isinstance(svf_path_raw, str):
            msg = (
                f"[cast].svf_boundary_path must be a string,"
                f" got {type(svf_path_raw).__name__}"
            )
            raise TypeError(msg)
        svf_boundary_path = svf_path_raw

    if direction_source_raw == "svf" and svf_boundary_path is None:
        msg = (
            "[cast].svf_boundary_path is required"
            " when direction_source = 'svf'"
        )
        raise ValueError(msg)

    # -- bank_path (optional) --
    bank_path_raw = sec.get("bank_path")  # type: ignore[arg-type]
    bank_path: str | None = None
    if bank_path_raw is not None:
        if not isinstance(bank_path_raw, str):
            msg = (
                f"[cast].bank_path must be a string,"
                f" got {type(bank_path_raw).__name__}"
            )
            raise TypeError(msg)
        bank_path = bank_path_raw

    # -- composition (optional, dict of name -> float) --
    comp_raw = sec.get("composition")  # type: ignore[arg-type]
    composition: dict[str, float] = {}
    if comp_raw is not None:
        if not isinstance(comp_raw, dict):
            msg = (
                f"[cast].composition must be a table,"
                f" got {type(comp_raw).__name__}"
            )
            raise TypeError(msg)
        for k, v in comp_raw.items():
            if not isinstance(v, int | float):
                msg = (
                    f"[cast].composition.{k} must be a number,"
                    f" got {type(v).__name__}"
                )
                raise TypeError(msg)
            composition[str(k)] = float(v)

    # -- externality_monitor (optional, bool) --
    ext_monitor_raw = sec.get(  # type: ignore[arg-type]
        "externality_monitor", False,
    )
    if not isinstance(ext_monitor_raw, bool):
        msg = (
            "[cast].externality_monitor must be a boolean,"
            f" got {type(ext_monitor_raw).__name__}"
        )
        raise TypeError(msg)

    # -- displacement_threshold (optional, float) --
    disp_thresh_raw = sec.get(  # type: ignore[arg-type]
        "displacement_threshold", 0.0,
    )
    if not isinstance(disp_thresh_raw, int | float):
        msg = (
            "[cast].displacement_threshold must be a number,"
            f" got {type(disp_thresh_raw).__name__}"
        )
        raise TypeError(msg)

    # -- baseline_activations_path (optional, string) --
    baseline_path_raw = sec.get(  # type: ignore[arg-type]
        "baseline_activations_path", None,
    )
    baseline_activations_path: str | None = None
    if baseline_path_raw is not None:
        if not isinstance(baseline_path_raw, str):
            msg = (
                "[cast].baseline_activations_path must be"
                f" a string, got {type(baseline_path_raw).__name__}"
            )
            raise TypeError(msg)
        baseline_activations_path = baseline_path_raw

    return CastConfig(
        prompts=list(prompts_raw),
        layers=layers,
        alpha=float(alpha_raw),
        threshold=float(threshold_raw),
        max_tokens=max_tokens_raw,
        condition_direction_path=condition_direction,
        alpha_tiers=alpha_tiers,
        direction_source=direction_source_raw,
        svf_boundary_path=svf_boundary_path,
        bank_path=bank_path,
        composition=composition,
        externality_monitor=ext_monitor_raw,
        displacement_threshold=float(disp_thresh_raw),
        baseline_activations_path=baseline_activations_path,
    )
