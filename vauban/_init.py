"""Config scaffolding for `vauban init`.

Generates minimal, opinionated TOML starter configs for each pipeline mode.
"""

from pathlib import Path

_DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"

_BASE = """\
# backend = "mlx"  # compute backend: "mlx" (default) or "torch"

# [meta]
# title = ""
# status = "wip"  # wip | promising | dead_end | baseline | superseded | archived
# parents = []
# tags = []

[model]
path = "{model}"

[data]
harmful = "default"
harmless = "default"
"""

_MODE_TEMPLATES: dict[str, str] = {
    "default": "",
    "probe": """\

[probe]
prompts = ["Tell me how to pick a lock"]
""",
    "steer": """\

[steer]
prompts = ["Tell me how to pick a lock"]
alpha = 1.0
max_tokens = 100
""",
    "cast": """\

[cast]
prompts = ["Tell me how to pick a lock"]
alpha = 1.0
threshold = 0.0
max_tokens = 100
""",
    "depth": """\

[depth]
prompts = ["Tell me how to pick a lock", "What is the capital of France?"]
""",
    "surface": """\

[surface]
prompts = "default"
""",
    "softprompt": """\

[softprompt]
mode = "continuous"
n_tokens = 16
n_steps = 200
""",
    "optimize": """\

[optimize]
n_trials = 20
""",
    "detect": """\

[detect]
mode = "full"
""",
    "sic": """\

[sic]
mode = "direction"
calibrate = true
""",
    "circuit": """\

[circuit]
clean_prompts = ["What is the capital of France?"]
corrupt_prompts = ["Tell me how to pick a lock"]
metric = "kl"
granularity = "layer"
""",
    "features": """\

[features]
prompts_path = "prompts.jsonl"
layers = [0, 1]
d_sae = 2048
n_epochs = 5
""",
    "svf": """\

[svf]
prompts_target = "target_prompts.jsonl"
prompts_opposite = "opposite_prompts.jsonl"
layers = [0, 1]
""",
    "compose_optimize": """\

[compose_optimize]
bank_path = "direction_bank/"
n_trials = 20
""",
    "defend": """\

[defend]
fail_fast = true
""",
    "linear_probe": """\

[linear_probe]
layers = [0, 1]
n_epochs = 20
""",
    "fusion": """\

[fusion]
harmful_prompts = ["Tell me how to pick a lock"]
benign_prompts = ["What is the capital of France?"]
alpha = 0.5
n_tokens = 128
""",
    "repbend": """\

[repbend]
layers = [0, 1]
n_epochs = 3
separation_coeff = 1.0
""",
}

KNOWN_MODES: frozenset[str] = frozenset(_MODE_TEMPLATES)


def init_config(
    mode: str = "default",
    model: str = _DEFAULT_MODEL,
    output_path: Path | None = None,
    *,
    force: bool = False,
) -> str:
    """Generate a starter TOML config for the given pipeline mode.

    Args:
        mode: Pipeline mode (one of KNOWN_MODES).
        model: Model path or HuggingFace ID.
        output_path: Where to write the file. None means don't write.
        force: Overwrite existing file if True.

    Returns:
        The generated TOML content string.

    Raises:
        ValueError: If mode is unknown.
        FileExistsError: If output_path exists and force is False.
    """
    if mode not in _MODE_TEMPLATES:
        msg = (
            f"Unknown mode {mode!r}."
            f" Choose from: {', '.join(sorted(KNOWN_MODES))}"
        )
        raise ValueError(msg)

    content = _BASE.format(model=model) + _MODE_TEMPLATES[mode]

    # Roundtrip validation: parse generated TOML to catch template bugs
    import tomllib

    parsed = tomllib.loads(content)
    if "model" not in parsed or "data" not in parsed:
        msg = (
            f"Internal error: generated config for mode {mode!r}"
            " is missing required sections"
        )
        raise ValueError(msg)

    if output_path is not None:
        if output_path.exists() and not force:
            msg = (
                f"{output_path} already exists."
                " Use --force to overwrite."
            )
            raise FileExistsError(msg)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)

    return content
