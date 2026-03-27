"""Parse the [flywheel] section of a TOML config."""

from vauban.config._parse_helpers import SectionReader, require_toml_table
from vauban.config._types import TomlDict
from vauban.types import FlywheelConfig

_VALID_POSITIONS = ("infix", "prefix", "suffix")
_VALID_SIC_MODES = ("direction", "generation", "svf")


def _parse_flywheel(raw: TomlDict) -> FlywheelConfig | None:
    """Parse the optional [flywheel] section into a FlywheelConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("flywheel")
    if sec is None:
        return None
    reader = SectionReader("[flywheel]", require_toml_table("[flywheel]", sec))

    # -- cycle counts --
    n_cycles = reader.integer("n_cycles", default=10)
    if n_cycles < 1:
        msg = f"[flywheel].n_cycles must be >= 1, got {n_cycles}"
        raise ValueError(msg)

    worlds_per_cycle = reader.integer("worlds_per_cycle", default=50)
    if worlds_per_cycle < 1:
        msg = f"[flywheel].worlds_per_cycle must be >= 1, got {worlds_per_cycle}"
        raise ValueError(msg)

    payloads_per_world = reader.integer("payloads_per_world", default=5)
    if payloads_per_world < 1:
        msg = (
            f"[flywheel].payloads_per_world must be >= 1,"
            f" got {payloads_per_world}"
        )
        raise ValueError(msg)

    # -- skeletons --
    skeletons = reader.string_list(
        "skeletons",
        default=["email", "doc", "code", "calendar", "search"],
    )
    if not skeletons:
        msg = "[flywheel].skeletons must be non-empty"
        raise ValueError(msg)

    # -- model expansion --
    model_expand = reader.boolean("model_expand", default=True)
    expand_temperature = reader.number("expand_temperature", default=0.7)
    expand_max_tokens = reader.integer("expand_max_tokens", default=200)

    # -- difficulty_range --
    difficulty_range_raw = reader.int_list(
        "difficulty_range", default=[1, 5],
    )
    if len(difficulty_range_raw) != 2:
        msg = (
            f"[flywheel].difficulty_range must have exactly 2 elements,"
            f" got {len(difficulty_range_raw)}"
        )
        raise ValueError(msg)
    d_min, d_max = difficulty_range_raw
    if d_min > d_max:
        msg = (
            f"[flywheel].difficulty_range[0] must be <= difficulty_range[1],"
            f" got [{d_min}, {d_max}]"
        )
        raise ValueError(msg)
    if d_min < 1 or d_max > 5:
        msg = (
            f"[flywheel].difficulty_range values must be in [1, 5],"
            f" got [{d_min}, {d_max}]"
        )
        raise ValueError(msg)
    difficulty_range: tuple[int, int] = (d_min, d_max)

    # -- payload library --
    payload_library_path = reader.optional_string("payload_library_path")

    # -- positions --
    positions = reader.string_list("positions", default=["infix"])
    if not positions:
        msg = "[flywheel].positions must be non-empty"
        raise ValueError(msg)
    for pos in positions:
        if pos not in _VALID_POSITIONS:
            msg = (
                f"[flywheel].positions: invalid position {pos!r},"
                f" must be one of {_VALID_POSITIONS}"
            )
            raise ValueError(msg)

    # -- GCG warm-start --
    warmstart_gcg = reader.boolean("warmstart_gcg", default=False)
    gcg_steps = reader.integer("gcg_steps", default=50)
    gcg_n_tokens = reader.integer("gcg_n_tokens", default=16)

    # -- CAST defense --
    cast_alpha = reader.number("cast_alpha", default=2.0)
    cast_threshold = reader.number("cast_threshold", default=0.0)
    cast_layers = reader.optional_int_list("cast_layers")

    # -- SIC defense --
    sic_threshold = reader.number("sic_threshold", default=0.5)
    sic_iterations = reader.integer("sic_iterations", default=3)
    if sic_iterations < 1:
        msg = f"[flywheel].sic_iterations must be >= 1, got {sic_iterations}"
        raise ValueError(msg)
    sic_mode = reader.literal(
        "sic_mode", _VALID_SIC_MODES, default="direction",
    )

    # -- hardening --
    harden = reader.boolean("harden", default=True)
    adaptation_rate = reader.number("adaptation_rate", default=0.1)
    if adaptation_rate <= 0.0 or adaptation_rate > 1.0:
        msg = (
            f"[flywheel].adaptation_rate must be in (0.0, 1.0],"
            f" got {adaptation_rate}"
        )
        raise ValueError(msg)

    utility_floor = reader.number("utility_floor", default=0.90)
    if utility_floor < 0.0 or utility_floor > 1.0:
        msg = (
            f"[flywheel].utility_floor must be in [0.0, 1.0],"
            f" got {utility_floor}"
        )
        raise ValueError(msg)

    validate_previous = reader.boolean("validate_previous", default=True)

    # -- convergence --
    convergence_window = reader.integer("convergence_window", default=3)
    if convergence_window < 2:
        msg = (
            f"[flywheel].convergence_window must be >= 2,"
            f" got {convergence_window}"
        )
        raise ValueError(msg)

    convergence_threshold = reader.number(
        "convergence_threshold", default=0.01,
    )

    # -- misc --
    seed = reader.optional_integer("seed")
    max_turns = reader.integer("max_turns", default=6)
    max_gen_tokens = reader.integer("max_gen_tokens", default=200)

    return FlywheelConfig(
        n_cycles=n_cycles,
        worlds_per_cycle=worlds_per_cycle,
        payloads_per_world=payloads_per_world,
        skeletons=skeletons,
        model_expand=model_expand,
        expand_temperature=expand_temperature,
        expand_max_tokens=expand_max_tokens,
        difficulty_range=difficulty_range,
        payload_library_path=payload_library_path,
        positions=positions,
        warmstart_gcg=warmstart_gcg,
        gcg_steps=gcg_steps,
        gcg_n_tokens=gcg_n_tokens,
        cast_alpha=cast_alpha,
        cast_threshold=cast_threshold,
        cast_layers=cast_layers,
        sic_threshold=sic_threshold,
        sic_iterations=sic_iterations,
        sic_mode=sic_mode,
        harden=harden,
        adaptation_rate=adaptation_rate,
        utility_floor=utility_floor,
        validate_previous=validate_previous,
        convergence_window=convergence_window,
        convergence_threshold=convergence_threshold,
        seed=seed,
        max_turns=max_turns,
        max_gen_tokens=max_gen_tokens,
    )
