"""Parse the [optimize] section of a TOML config."""

from vauban.config._types import TomlDict
from vauban.types import OptimizeConfig


def _parse_optimize(raw: TomlDict) -> OptimizeConfig | None:
    """Parse the optional [optimize] section into an OptimizeConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("optimize")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[optimize] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    n_trials_raw = sec.get("n_trials", 50)  # type: ignore[arg-type]
    if not isinstance(n_trials_raw, int):
        msg = (
            f"[optimize].n_trials must be an integer,"
            f" got {type(n_trials_raw).__name__}"
        )
        raise TypeError(msg)
    if n_trials_raw < 1:
        msg = f"[optimize].n_trials must be >= 1, got {n_trials_raw}"
        raise ValueError(msg)

    alpha_min_raw = sec.get("alpha_min", 0.1)  # type: ignore[arg-type]
    if not isinstance(alpha_min_raw, int | float):
        msg = (
            f"[optimize].alpha_min must be a number,"
            f" got {type(alpha_min_raw).__name__}"
        )
        raise TypeError(msg)

    alpha_max_raw = sec.get("alpha_max", 5.0)  # type: ignore[arg-type]
    if not isinstance(alpha_max_raw, int | float):
        msg = (
            f"[optimize].alpha_max must be a number,"
            f" got {type(alpha_max_raw).__name__}"
        )
        raise TypeError(msg)

    if float(alpha_min_raw) >= float(alpha_max_raw):
        msg = (
            f"[optimize].alpha_min ({alpha_min_raw}) must be"
            f" < alpha_max ({alpha_max_raw})"
        )
        raise ValueError(msg)

    sparsity_min_raw = sec.get("sparsity_min", 0.0)  # type: ignore[arg-type]
    if not isinstance(sparsity_min_raw, int | float):
        msg = (
            f"[optimize].sparsity_min must be a number,"
            f" got {type(sparsity_min_raw).__name__}"
        )
        raise TypeError(msg)

    sparsity_max_raw = sec.get("sparsity_max", 0.9)  # type: ignore[arg-type]
    if not isinstance(sparsity_max_raw, int | float):
        msg = (
            f"[optimize].sparsity_max must be a number,"
            f" got {type(sparsity_max_raw).__name__}"
        )
        raise TypeError(msg)

    search_norm_raw = sec.get("search_norm_preserve", True)  # type: ignore[arg-type]
    if not isinstance(search_norm_raw, bool):
        msg = (
            f"[optimize].search_norm_preserve must be a boolean,"
            f" got {type(search_norm_raw).__name__}"
        )
        raise TypeError(msg)

    strategies_raw = sec.get(  # type: ignore[arg-type]
        "search_strategies", ["all", "above_median", "top_k"],
    )
    if not isinstance(strategies_raw, list):
        msg = (
            f"[optimize].search_strategies must be a list,"
            f" got {type(strategies_raw).__name__}"
        )
        raise TypeError(msg)
    valid_strategies = {"all", "above_median", "top_k"}
    strategies: list[str] = [str(s) for s in strategies_raw]
    for s in strategies:
        if s not in valid_strategies:
            msg = (
                f"[optimize].search_strategies contains invalid"
                f" strategy {s!r}, must be one of {valid_strategies!r}"
            )
            raise ValueError(msg)

    layer_top_k_min_raw = sec.get("layer_top_k_min", 3)  # type: ignore[arg-type]
    if not isinstance(layer_top_k_min_raw, int):
        msg = (
            f"[optimize].layer_top_k_min must be an integer,"
            f" got {type(layer_top_k_min_raw).__name__}"
        )
        raise TypeError(msg)

    layer_top_k_max_raw = sec.get("layer_top_k_max")  # type: ignore[arg-type]
    layer_top_k_max: int | None = None
    if layer_top_k_max_raw is not None:
        if not isinstance(layer_top_k_max_raw, int):
            msg = (
                f"[optimize].layer_top_k_max must be an integer,"
                f" got {type(layer_top_k_max_raw).__name__}"
            )
            raise TypeError(msg)
        layer_top_k_max = layer_top_k_max_raw

    max_tokens_raw = sec.get("max_tokens", 100)  # type: ignore[arg-type]
    if not isinstance(max_tokens_raw, int):
        msg = (
            f"[optimize].max_tokens must be an integer,"
            f" got {type(max_tokens_raw).__name__}"
        )
        raise TypeError(msg)

    seed_raw = sec.get("seed")  # type: ignore[arg-type]
    seed: int | None = None
    if seed_raw is not None:
        if not isinstance(seed_raw, int):
            msg = (
                f"[optimize].seed must be an integer,"
                f" got {type(seed_raw).__name__}"
            )
            raise TypeError(msg)
        seed = seed_raw

    timeout_raw = sec.get("timeout")  # type: ignore[arg-type]
    timeout: float | None = None
    if timeout_raw is not None:
        if not isinstance(timeout_raw, int | float):
            msg = (
                f"[optimize].timeout must be a number,"
                f" got {type(timeout_raw).__name__}"
            )
            raise TypeError(msg)
        timeout = float(timeout_raw)

    return OptimizeConfig(
        n_trials=n_trials_raw,
        alpha_min=float(alpha_min_raw),
        alpha_max=float(alpha_max_raw),
        sparsity_min=float(sparsity_min_raw),
        sparsity_max=float(sparsity_max_raw),
        search_norm_preserve=search_norm_raw,
        search_strategies=strategies,
        layer_top_k_min=layer_top_k_min_raw,
        layer_top_k_max=layer_top_k_max,
        max_tokens=max_tokens_raw,
        seed=seed,
        timeout=timeout,
    )
