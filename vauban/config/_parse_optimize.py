"""Parse the [optimize] section of a TOML config."""

from vauban.config._parse_helpers import SectionReader, require_toml_table
from vauban.config._types import TomlDict
from vauban.types import OptimizeConfig


def _parse_optimize(raw: TomlDict) -> OptimizeConfig | None:
    """Parse the optional [optimize] section into an OptimizeConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("optimize")
    if sec is None:
        return None
    reader = SectionReader("[optimize]", require_toml_table("[optimize]", sec))

    n_trials = reader.integer("n_trials", default=50)
    if n_trials < 1:
        msg = f"[optimize].n_trials must be >= 1, got {n_trials}"
        raise ValueError(msg)

    alpha_min = reader.number("alpha_min", default=0.1)
    alpha_max = reader.number("alpha_max", default=5.0)
    if alpha_min >= alpha_max:
        msg = (
            f"[optimize].alpha_min ({alpha_min}) must be"
            f" < alpha_max ({alpha_max})"
        )
        raise ValueError(msg)

    sparsity_min = reader.number("sparsity_min", default=0.0)
    sparsity_max = reader.number("sparsity_max", default=0.9)

    search_norm_preserve = reader.boolean(
        "search_norm_preserve", default=True,
    )

    strategies = reader.string_list(
        "search_strategies",
        default=["all", "above_median", "top_k"],
    )
    valid_strategies = {"all", "above_median", "top_k"}
    for s in strategies:
        if s not in valid_strategies:
            msg = (
                f"[optimize].search_strategies contains invalid"
                f" strategy {s!r}, must be one of {valid_strategies!r}"
            )
            raise ValueError(msg)

    layer_top_k_min = reader.integer("layer_top_k_min", default=3)
    layer_top_k_max = reader.optional_integer("layer_top_k_max")
    max_tokens = reader.integer("max_tokens", default=100)
    seed = reader.optional_integer("seed")
    timeout = reader.optional_number("timeout")

    return OptimizeConfig(
        n_trials=n_trials,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        sparsity_min=sparsity_min,
        sparsity_max=sparsity_max,
        search_norm_preserve=search_norm_preserve,
        search_strategies=strategies,
        layer_top_k_min=layer_top_k_min,
        layer_top_k_max=layer_top_k_max,
        max_tokens=max_tokens,
        seed=seed,
        timeout=timeout,
    )
