"""Config scaffolding for `vauban init`.

Generates minimal, opinionated TOML starter configs for each pipeline mode.
"""

from pathlib import Path

_DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"

_BASE = """\
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
