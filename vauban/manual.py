"""Runtime-generated manual for vauban's TOML-first interface."""

import ast
import inspect
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Manual metadata for one TOML key."""

    key: str
    description: str
    constraints: str | None = None
    attr: str | None = None
    required: bool | None = None
    default_override: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SectionSpec:
    """Manual metadata for one TOML section."""

    name: str
    description: str
    required: bool = False
    early_return: bool = False
    table: bool = True
    config_class: str | None = None
    fields: tuple[FieldSpec, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AutoField:
    """Auto-discovered field metadata from a config dataclass."""

    type_name: str
    default_repr: str | None
    required: bool


@dataclass(frozen=True, slots=True)
class ManualField:
    """Rendered field documentation."""

    key: str
    type_name: str
    required: bool
    default_repr: str | None
    description: str
    constraints: str | None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ManualSection:
    """Rendered section documentation."""

    name: str
    description: str
    required: bool
    early_return: bool
    table: bool
    fields: tuple[ManualField, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PipelineModeDoc:
    """Manual metadata for one pipeline mode."""

    mode: str
    trigger: str
    output: str
    early_return: bool


EARLY_RETURN_PRECEDENCE: tuple[str, ...] = (
    "depth",
    "probe",
    "steer",
    "sic",
    "optimize",
    "softprompt",
)

_PIPELINE_MODES: tuple[PipelineModeDoc, ...] = (
    PipelineModeDoc(
        mode="default",
        trigger="No early-return section is present.",
        output="Modified model directory in [output].dir.",
        early_return=False,
    ),
    PipelineModeDoc(
        mode="depth",
        trigger="[depth] section present.",
        output="depth_report.json (+ optional depth_direction.npy).",
        early_return=True,
    ),
    PipelineModeDoc(
        mode="probe",
        trigger="[probe] section present.",
        output="probe_report.json.",
        early_return=True,
    ),
    PipelineModeDoc(
        mode="steer",
        trigger="[steer] section present.",
        output="steer_report.json.",
        early_return=True,
    ),
    PipelineModeDoc(
        mode="sic",
        trigger="[sic] section present.",
        output="sic_report.json.",
        early_return=True,
    ),
    PipelineModeDoc(
        mode="optimize",
        trigger="[optimize] section present.",
        output="optimize_report.json.",
        early_return=True,
    ),
    PipelineModeDoc(
        mode="softprompt",
        trigger="[softprompt] section present.",
        output="softprompt_report.json.",
        early_return=True,
    ),
)

_FORMAT_NOTES: tuple[str, ...] = (
    "Prompt JSONL ([data] + [eval]): one object per line with key 'prompt'.",
    "Surface JSONL ([surface].prompts): keys 'prompt', 'label', and 'category'.",
    "Refusal phrases file: plain text, one phrase per line ('#' comments allowed).",
    "All relative paths resolve from the directory of the loaded TOML file.",
)

_QUICKSTART_NOTES: tuple[str, ...] = (
    "1. Create run.toml with the minimal config shown below.",
    "2. Validate first: vauban --validate run.toml",
    "3. Run: vauban run.toml",
    "4. Inspect outputs in [output].dir (default: output/).",
    "5. If vauban is installed through uv project deps, use: uv run vauban ...",
)

_MINIMAL_CONFIG_EXAMPLE: tuple[str, ...] = (
    "[model]",
    'path = "mlx-community/Llama-3.2-3B-Instruct-4bit"',
    "",
    "[data]",
    'harmful = "default"',
    'harmless = "default"',
)

_VALIDATE_NOTES: tuple[str, ...] = (
    "Use validate before running heavy experiments.",
    "Warnings are tagged [LOW], [MEDIUM], or [HIGH].",
    "Most warnings include a 'fix:' hint for direct remediation.",
    (
        "Validate checks JSONL schema, missing files, mode conflicts,"
        " skipped sections, and key-level typos within each section."
    ),
)

_PLAYBOOK_NOTES: tuple[str, ...] = (
    "1. Scaffold a config: vauban init --mode default --output run.toml",
    "2. Validate and iterate until warnings are understood/fixed.",
    "3. Run one experiment per TOML file for reproducibility.",
    "4. Compare two runs with: vauban diff out_a out_b",
    "5. Keep an experiment log (config path + output dir + key metrics).",
)

_QUICK_NOTES: tuple[str, ...] = (
    "Use vauban.quick in Python REPL/Jupyter for rapid experiments.",
    "This is complementary to TOML runs; use TOML for reproducible reports.",
)

_EXAMPLE_NOTES: tuple[str, ...] = (
    "Scaffold a starter config:",
    "  vauban init --mode default --output run.toml",
    "Validate before expensive runs:",
    "  vauban --validate run.toml",
    "Run default pipeline:",
    "  vauban run.toml",
    "Compare two experiment outputs:",
    "  vauban diff runs/baseline runs/experiment_a",
    "Open manual for one topic:",
    "  vauban man softprompt",
)

_PRINT_NOTES: tuple[str, ...] = (
    "Full manual to text file:",
    "  vauban man > VAUBAN_MANUAL.txt",
    "Focused topic to text file:",
    "  vauban man cut > VAUBAN_CUT.txt",
    "Read in pager (Unix):",
    "  vauban man | less",
    "Print from Unix shell:",
    "  lpr VAUBAN_MANUAL.txt",
)

_QUICK_EXAMPLE: tuple[str, ...] = (
    "from vauban import quick",
    "",
    'model, tokenizer = quick.load("mlx-community/Llama-3.2-3B-Instruct-4bit")',
    "direction = quick.measure_direction(model, tokenizer)",
    "probe = quick.probe_prompt(model, tokenizer, \"Explain lockpicking\", direction)",
    (
        "steered = quick.steer_prompt("
        "model, tokenizer, \"Explain lockpicking\", direction, alpha=1.0)"
    ),
    "surface = quick.scan(model, tokenizer, direction)",
    'print(quick.compare("output_a", "output_b"))',
)

_SECTION_SPECS: tuple[SectionSpec, ...] = (
    SectionSpec(
        name="model",
        description="Model loading configuration.",
        required=True,
        fields=(
            FieldSpec(
                key="path",
                description="Model identifier or local model directory.",
                constraints="required string; must be loadable by mlx_lm.load().",
                required=True,
            ),
        ),
    ),
    SectionSpec(
        name="data",
        description="Prompt sources for harmful/harmless measurement sets.",
        required=True,
        fields=(
            FieldSpec(
                key="harmful",
                description="Source for harmful prompts.",
                constraints=(
                    'required; one of: "default", local path string,'
                    ' "hf:<repo_id>", or [data.harmful] HF table.'
                ),
                required=True,
            ),
            FieldSpec(
                key="harmless",
                description="Source for harmless prompts.",
                constraints=(
                    'required; one of: "default", local path string,'
                    ' "hf:<repo_id>", or [data.harmless] HF table.'
                ),
                required=True,
            ),
            FieldSpec(
                key="borderline",
                description="Optional borderline prompts for false-refusal handling.",
                constraints=(
                    "string or HF dataset reference; required when"
                    " [cut].false_refusal_ortho = true."
                ),
            ),
        ),
        notes=(
            "HF table form accepts keys: hf, split, column, config, limit.",
        ),
    ),
    SectionSpec(
        name="measure",
        description="Behavioral direction extraction settings.",
        config_class="MeasureConfig",
        fields=(
            FieldSpec(
                key="mode",
                description="Measurement algorithm.",
                constraints="one of: direction, subspace, dbdi.",
            ),
            FieldSpec(
                key="top_k",
                description="Number of directions to keep in subspace workflows.",
                constraints="integer.",
            ),
            FieldSpec(
                key="clip_quantile",
                description="Winsorization quantile for activation clipping.",
                constraints="number in [0.0, 0.5).",
            ),
        ),
    ),
    SectionSpec(
        name="cut",
        description="Weight-space surgery controls.",
        config_class="CutConfig",
        fields=(
            FieldSpec(
                key="alpha",
                description="Global cut strength multiplier.",
                constraints="number.",
            ),
            FieldSpec(
                key="layers",
                description="Explicit target layer list or auto mode.",
                constraints='either "auto" or list of integers.',
            ),
            FieldSpec(
                key="norm_preserve",
                description="Preserve row norms after projection removal.",
                constraints="boolean.",
            ),
            FieldSpec(
                key="biprojected",
                description="Apply biprojected direction removal.",
                constraints="boolean.",
            ),
            FieldSpec(
                key="layer_strategy",
                description="Automatic layer selection strategy.",
                constraints="one of: all, above_median, top_k.",
            ),
            FieldSpec(
                key="layer_top_k",
                description="Layer count when layer_strategy = top_k.",
                constraints="integer.",
            ),
            FieldSpec(
                key="layer_weights",
                description="Per-layer alpha multipliers.",
                constraints="list of numbers; length should match selected layers.",
            ),
            FieldSpec(
                key="sparsity",
                description="Fraction of direction components to zero before cutting.",
                constraints="number in [0.0, 1.0).",
            ),
            FieldSpec(
                key="dbdi_target",
                description="Which DBDI component to cut when measure.mode = dbdi.",
                constraints="one of: red, hdd, both.",
            ),
            FieldSpec(
                key="false_refusal_ortho",
                description=(
                    "Orthogonalize against a borderline false-refusal direction."
                ),
                constraints="boolean.",
                notes=(
                    "Requires [data].borderline to be configured.",
                ),
            ),
            FieldSpec(
                key="layer_type_filter",
                description="Optional architectural layer-type filter.",
                constraints="one of: global, sliding, or null.",
            ),
        ),
    ),
    SectionSpec(
        name="eval",
        description="Post-cut quality and refusal evaluation.",
        config_class="EvalConfig",
        fields=(
            FieldSpec(
                key="prompts",
                attr="prompts_path",
                description="Path to evaluation prompts JSONL.",
                constraints="string path; if omitted, harmful fallback is used.",
            ),
            FieldSpec(
                key="max_tokens",
                description="Generation cap for refusal-rate evaluation.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="num_prompts",
                description="Fallback count when eval prompts file is omitted.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="refusal_phrases",
                attr="refusal_phrases_path",
                description="Custom refusal phrase file.",
                constraints="string path; one phrase per line.",
            ),
        ),
    ),
    SectionSpec(
        name="surface",
        description="Before/after refusal surface mapping.",
        config_class="SurfaceConfig",
        fields=(
            FieldSpec(
                key="prompts",
                attr="prompts_path",
                description="Surface prompt source.",
                constraints='string path or "default".',
            ),
            FieldSpec(
                key="generate",
                description="Generate text while scanning (vs projections only).",
                constraints="boolean.",
            ),
            FieldSpec(
                key="max_tokens",
                description="Generation cap per surface prompt.",
                constraints="integer.",
            ),
            FieldSpec(
                key="progress",
                description="Show scan progress logs.",
                constraints="boolean.",
            ),
        ),
    ),
    SectionSpec(
        name="detect",
        description="Defense-hardening detection that runs before cutting.",
        config_class="DetectConfig",
        fields=(
            FieldSpec(
                key="mode",
                description="Detection depth.",
                constraints="one of: fast, probe, full.",
            ),
            FieldSpec(
                key="top_k",
                description="Subspace dimensionality for probe/full diagnostics.",
                constraints="integer.",
            ),
            FieldSpec(
                key="clip_quantile",
                description="Optional activation clipping quantile for detection.",
                constraints="number.",
            ),
            FieldSpec(
                key="alpha",
                description="Test cut strength in full mode.",
                constraints="number.",
            ),
            FieldSpec(
                key="max_tokens",
                description="Generation cap in full mode.",
                constraints="integer.",
            ),
        ),
        notes=(
            (
                "Detection does not run when [depth] is active, because depth"
                " returns early."
            ),
        ),
    ),
    SectionSpec(
        name="optimize",
        description="Optuna multi-objective search over cut parameters.",
        early_return=True,
        config_class="OptimizeConfig",
        fields=(
            FieldSpec(
                key="n_trials",
                description="Number of Optuna trials.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="alpha_min",
                description="Minimum alpha sampled by the optimizer.",
                constraints="number; must be < alpha_max.",
            ),
            FieldSpec(
                key="alpha_max",
                description="Maximum alpha sampled by the optimizer.",
                constraints="number; must be > alpha_min.",
            ),
            FieldSpec(
                key="sparsity_min",
                description="Minimum direction sparsity sampled by the optimizer.",
                constraints="number.",
            ),
            FieldSpec(
                key="sparsity_max",
                description="Maximum direction sparsity sampled by the optimizer.",
                constraints="number.",
            ),
            FieldSpec(
                key="search_norm_preserve",
                description="Search both norm-preserving and standard cuts.",
                constraints="boolean.",
            ),
            FieldSpec(
                key="search_strategies",
                description="Layer strategies included in the search.",
                constraints="list of: all, above_median, top_k.",
            ),
            FieldSpec(
                key="layer_top_k_min",
                description="Minimum top-k layer count sampled.",
                constraints="integer.",
            ),
            FieldSpec(
                key="layer_top_k_max",
                description="Maximum top-k layer count sampled.",
                constraints="integer or null.",
            ),
            FieldSpec(
                key="max_tokens",
                description="Generation cap used while scoring each trial.",
                constraints="integer.",
            ),
            FieldSpec(
                key="seed",
                description="Optional optimizer seed.",
                constraints="integer or null.",
            ),
            FieldSpec(
                key="timeout",
                description="Optional wall-clock timeout in seconds.",
                constraints="number or null.",
            ),
        ),
    ),
    SectionSpec(
        name="softprompt",
        description="Continuous/discrete soft prompt attack configuration.",
        early_return=True,
        config_class="SoftPromptConfig",
        fields=(
            FieldSpec(
                key="mode",
                description="Soft prompt optimization algorithm.",
                constraints="one of: continuous, gcg, egd.",
            ),
            FieldSpec(
                key="n_tokens",
                description="Learnable prompt length.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="n_steps",
                description="Optimization step count.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="learning_rate",
                description="Learning rate for continuous mode.",
                constraints="number > 0.",
            ),
            FieldSpec(
                key="init_scale",
                description="Initial embedding scale.",
                constraints="number.",
            ),
            FieldSpec(
                key="batch_size",
                description="Candidate batch size for token search.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="top_k",
                description="Top-k token candidates per position.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="direction_weight",
                description="Weight for direction-aware regularization.",
                constraints="number >= 0.",
            ),
            FieldSpec(
                key="target_prefixes",
                description="Target prefixes used by targeted losses.",
                constraints="list of strings.",
            ),
            FieldSpec(
                key="max_gen_tokens",
                description="Generation cap for attack evaluation.",
                constraints="integer.",
            ),
            FieldSpec(
                key="seed",
                description="Optional random seed.",
                constraints="integer or null.",
            ),
            FieldSpec(
                key="embed_reg_weight",
                description="Embedding norm regularization strength.",
                constraints="number >= 0.",
            ),
            FieldSpec(
                key="patience",
                description="Early stopping patience (0 disables).",
                constraints="integer >= 0.",
            ),
            FieldSpec(
                key="lr_schedule",
                description="Learning-rate schedule.",
                constraints="one of: constant, cosine.",
            ),
            FieldSpec(
                key="n_restarts",
                description="Number of random restarts.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="prompt_strategy",
                description="Prompt sampling strategy during optimization.",
                constraints="one of: all, cycle, first, worst_k.",
            ),
            FieldSpec(
                key="direction_mode",
                description="How direction loss is applied in token positions.",
                constraints="one of: last, raid, all_positions.",
            ),
            FieldSpec(
                key="direction_layers",
                description="Specific layers for direction loss.",
                constraints="list of integers or null.",
            ),
            FieldSpec(
                key="loss_mode",
                description="Primary loss objective.",
                constraints="one of: targeted, untargeted, defensive.",
            ),
            FieldSpec(
                key="egd_temperature",
                description="Temperature for EGD simplex sharpening.",
                constraints="number > 0.",
            ),
            FieldSpec(
                key="token_constraint",
                description="Token constraint set for candidate tokens.",
                constraints="one of: ascii, alpha, alphanumeric, or null.",
            ),
            FieldSpec(
                key="eos_loss_mode",
                description="EOS auxiliary loss behavior.",
                constraints="one of: none, force, suppress.",
            ),
            FieldSpec(
                key="eos_loss_weight",
                description="Weight for EOS auxiliary loss.",
                constraints="number >= 0.",
            ),
            FieldSpec(
                key="kl_ref_weight",
                description="KL collision regularization weight.",
                constraints="number >= 0.",
                notes=(
                    "If > 0, [softprompt].ref_model must be set.",
                ),
            ),
            FieldSpec(
                key="ref_model",
                description="Reference model for KL collision loss.",
                constraints="string or null.",
            ),
            FieldSpec(
                key="worst_k",
                description="Prompt count used by worst_k strategy.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="grad_accum_steps",
                description="Gradient accumulation steps.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="transfer_models",
                description="Model IDs used for transfer evaluation.",
                constraints="list of strings.",
            ),
        ),
    ),
    SectionSpec(
        name="sic",
        description="SIC iterative sanitization settings.",
        early_return=True,
        config_class="SICConfig",
        fields=(
            FieldSpec(
                key="mode",
                description="Sanitization scoring mode.",
                constraints="one of: direction, generation.",
            ),
            FieldSpec(
                key="threshold",
                description="Detection threshold.",
                constraints="number.",
            ),
            FieldSpec(
                key="max_iterations",
                description="Maximum sanitize iterations per prompt.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="max_tokens",
                description="Generation cap for generation mode scoring.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="target_layer",
                description="Layer override for direction projection.",
                constraints="integer or null.",
            ),
            FieldSpec(
                key="sanitize_system_prompt",
                description="System prompt used for rewrite sanitization.",
                constraints="string.",
            ),
            FieldSpec(
                key="max_sanitize_tokens",
                description="Token cap for sanitize rewrites.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="block_on_failure",
                description="Block prompts that remain unsafe after retries.",
                constraints="boolean.",
            ),
            FieldSpec(
                key="calibrate",
                description="Auto-calibrate threshold from prompt samples.",
                constraints="boolean.",
            ),
            FieldSpec(
                key="calibrate_prompts",
                description="Prompt pool used when calibration is enabled.",
                constraints="one of: harmless, harmful.",
            ),
        ),
    ),
    SectionSpec(
        name="depth",
        description="Deep-thinking token analysis settings.",
        early_return=True,
        config_class="DepthConfig",
        fields=(
            FieldSpec(
                key="prompts",
                description="Inline prompts for depth profiling.",
                constraints="required non-empty list of strings.",
                required=True,
            ),
            FieldSpec(
                key="settling_threshold",
                description="JSD settling threshold.",
                constraints="number in (0.0, 1.0].",
            ),
            FieldSpec(
                key="deep_fraction",
                description=(
                    "Layer-fraction threshold for deep-thinking classification."
                ),
                constraints="number in (0.0, 1.0].",
            ),
            FieldSpec(
                key="top_k_logits",
                description="Top-k logit approximation size for JSD.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="max_tokens",
                description="Generated tokens for dynamic depth mode.",
                constraints="integer >= 0.",
            ),
            FieldSpec(
                key="extract_direction",
                description="Extract a depth direction in addition to depth profiles.",
                constraints="boolean.",
                notes=(
                    "Requires at least 2 effective prompts"
                    " (direction_prompts if set, else prompts).",
                ),
            ),
            FieldSpec(
                key="direction_prompts",
                description="Optional prompt subset for direction extraction.",
                constraints="list of strings or null.",
            ),
            FieldSpec(
                key="clip_quantile",
                description="Winsorization quantile during direction extraction.",
                constraints="number in [0.0, 0.5).",
            ),
        ),
    ),
    SectionSpec(
        name="probe",
        description="Per-layer projection inspection.",
        early_return=True,
        config_class="ProbeConfig",
        fields=(
            FieldSpec(
                key="prompts",
                description="Inline prompts to probe.",
                constraints="required non-empty list of strings.",
                required=True,
            ),
        ),
    ),
    SectionSpec(
        name="steer",
        description="Runtime activation steering for text generation.",
        early_return=True,
        config_class="SteerConfig",
        fields=(
            FieldSpec(
                key="prompts",
                description="Inline prompts used for steered generation.",
                constraints="required non-empty list of strings.",
                required=True,
            ),
            FieldSpec(
                key="layers",
                description="Layer subset to steer.",
                constraints="list of integers or null (null means all layers).",
            ),
            FieldSpec(
                key="alpha",
                description="Steering strength.",
                constraints="number.",
            ),
            FieldSpec(
                key="max_tokens",
                description="Generation cap per prompt.",
                constraints="integer >= 1.",
            ),
        ),
    ),
    SectionSpec(
        name="output",
        description="Filesystem output location.",
        fields=(
            FieldSpec(
                key="dir",
                description="Directory where models/reports are written.",
                constraints="string path.",
                default_override='"output"',
            ),
        ),
    ),
    SectionSpec(
        name="verbose",
        description="Top-level logging control.",
        table=False,
        fields=(
            FieldSpec(
                key="verbose",
                description="Enable stderr progress logs.",
                constraints="boolean.",
                default_override="true",
            ),
        ),
    ),
)

_TOPIC_ALIASES: dict[str, str] = {
    "start": "quickstart",
    "getting-started": "quickstart",
    "getting_started": "quickstart",
    "command": "commands",
    "cmd": "commands",
    "checks": "validate",
    "validation": "validate",
    "lint": "validate",
    "playbooks": "playbook",
    "experiment": "playbook",
    "experiments": "playbook",
    "recipe": "playbook",
    "recipes": "playbook",
    "repl": "quick",
    "python": "quick",
    "quick-api": "quick",
    "quick_api": "quick",
    "example": "examples",
    "demo": "examples",
    "demos": "examples",
    "printing": "print",
    "share": "print",
    "sharing": "print",
    "pdf": "print",
    "mode": "modes",
    "pipeline": "modes",
    "pipelines": "modes",
    "format": "formats",
    "file": "formats",
    "files": "formats",
}


def manual_topics() -> list[str]:
    """Return supported manual topics."""
    return [
        "all",
        "quickstart",
        "commands",
        "validate",
        "playbook",
        "quick",
        "examples",
        "print",
        "modes",
        "formats",
        *[spec.name for spec in _SECTION_SPECS],
    ]


@lru_cache(maxsize=1)
def _known_init_modes() -> tuple[str, ...]:
    from vauban._init import KNOWN_MODES

    return tuple(sorted(KNOWN_MODES))


@lru_cache(maxsize=1)
def _known_diff_reports() -> tuple[str, ...]:
    from vauban._diff import known_report_filenames

    return known_report_filenames()


@lru_cache(maxsize=1)
def _known_quick_functions() -> tuple[str, ...]:
    import vauban.quick as quick

    names: list[str] = []
    for name in dir(quick):
        if name.startswith("_"):
            continue
        value = getattr(quick, name)
        if inspect.isfunction(value) and value.__module__ == quick.__name__:
            names.append(name)
    return tuple(sorted(names))


def render_manual(topic: str | None = None) -> str:
    """Render a grep-friendly text manual."""
    normalized = _normalize_topic(topic)
    sections = _build_sections()
    selected = _select_sections(sections, normalized)
    include_quickstart = normalized in (None, "all", "quickstart")
    include_commands = normalized in (None, "all", "commands")
    include_validate = normalized in (None, "all", "validate")
    include_playbook = normalized in (None, "all", "playbook")
    include_quick = normalized in (None, "all", "quick")
    include_examples = normalized in (None, "all", "examples")
    include_print = normalized in (None, "all", "print")
    include_modes = normalized in (None, "all", "modes")
    include_formats = normalized in (None, "all", "formats")

    lines: list[str] = []
    lines.append("VAUBAN(1)")
    lines.append("")
    lines.append("NAME")
    lines.append("    vauban - TOML-first model behavior surgery toolkit")
    lines.append("")
    lines.append("SYNOPSIS")
    lines.append("    vauban <config.toml>")
    lines.append("    vauban --validate <config.toml>")
    lines.append(
        "    vauban init [--mode MODE] [--model PATH]"
        " [--output FILE] [--force]",
    )
    lines.append(
        "    vauban diff [--format text|markdown]"
        " [--threshold FLOAT] <dir_a> <dir_b>",
    )
    lines.append("    vauban man [topic]")
    lines.append("")
    lines.append("DESCRIPTION")
    lines.append(
        "    Generated from vauban config dataclasses + parser constraints at runtime.",
    )
    lines.append("    Defaults and types auto-refresh when config dataclasses change.")
    lines.append("")
    lines.append("TOPICS")
    lines.append(f"    {', '.join(manual_topics())}")
    lines.append("")

    if include_quickstart:
        lines.append("QUICKSTART")
        for note in _QUICKSTART_NOTES:
            lines.append(f"    {note}")
        lines.append("    Minimal run.toml:")
        for line in _MINIMAL_CONFIG_EXAMPLE:
            lines.append(f"    {line}")
        lines.append("")

    if include_commands:
        lines.append("COMMANDS")
        lines.append("    vauban <config.toml>")
        lines.append("      Run the configured pipeline from a TOML file.")
        lines.append("    vauban --validate <config.toml>")
        lines.append(
            "      Validate config + prompt files without loading model weights.",
        )
        lines.append(
            "    vauban init [--mode MODE] [--model PATH]"
            " [--output FILE] [--force]",
        )
        lines.append("      Generate a starter config file.")
        lines.append(f"      known modes: {', '.join(_known_init_modes())}")
        lines.append(
            "    vauban diff [--format text|markdown]"
            " [--threshold FLOAT] <dir_a> <dir_b>",
        )
        lines.append("      Compare report metrics from two output directories.")
        lines.append(
            "      --format: output format (default: text).",
        )
        lines.append(
            "      --threshold: exit code 1 if any |delta| exceeds value.",
        )
        lines.append(f"      report files: {', '.join(_known_diff_reports())}")
        lines.append("    vauban man [topic]")
        lines.append("      Show this manual or one focused topic.")
        lines.append("")

    if include_validate:
        lines.append("VALIDATE WORKFLOW")
        for note in _VALIDATE_NOTES:
            lines.append(f"    - {note}")
        lines.append("    Example:")
        lines.append("      vauban --validate run.toml")
        lines.append("")

    if include_playbook:
        lines.append("EXPERIMENT PLAYBOOK")
        for note in _PLAYBOOK_NOTES:
            lines.append(f"    {note}")
        lines.append("    Common loop: init -> validate -> run -> diff -> tune.")
        lines.append("")

    if include_quick:
        lines.append("PYTHON QUICK API")
        for note in _QUICK_NOTES:
            lines.append(f"    - {note}")
        lines.append(f"    helpers: {', '.join(_known_quick_functions())}")
        lines.append("    Minimal interactive flow:")
        for line in _QUICK_EXAMPLE:
            lines.append(f"    {line}")
        lines.append("")

    if include_examples:
        lines.append("EXAMPLES")
        for note in _EXAMPLE_NOTES:
            lines.append(f"    {note}")
        lines.append("")

    if include_print:
        lines.append("PRINTING AND SHARING")
        for note in _PRINT_NOTES:
            lines.append(f"    {note}")
        lines.append("")

    if include_modes:
        lines.append("PIPELINE MODES")
        lines.append(
            "    Early-return precedence:"
            f" {' > '.join(EARLY_RETURN_PRECEDENCE)}",
        )
        for mode in _PIPELINE_MODES:
            suffix = " (early return)" if mode.early_return else ""
            lines.append(f"    {mode.mode}{suffix}")
            lines.append(f"      trigger: {mode.trigger}")
            lines.append(f"      output:  {mode.output}")
        lines.append("")

    if include_formats:
        lines.append("DATA FORMATS")
        for note in _FORMAT_NOTES:
            lines.append(f"    - {note}")
        lines.append("")

    if selected:
        lines.append("CONFIG SECTIONS")
        for section in selected:
            lines.append("")
            lines.append(_format_section_header(section))
            lines.append(f"  description: {section.description}")
            lines.append(f"  required: {'yes' if section.required else 'no'}")
            if section.early_return:
                lines.append("  early_return: yes")
            for note in section.notes:
                lines.append(f"  note: {note}")
            lines.append("  fields:")
            for field in section.fields:
                lines.append(f"    - {_field_path(section, field)}")
                lines.append(f"      type: {field.type_name}")
                lines.append(f"      required: {'yes' if field.required else 'no'}")
                if field.default_repr is not None:
                    lines.append(f"      default: {field.default_repr}")
                elif field.required:
                    lines.append("      default: (required)")
                else:
                    lines.append("      default: null")
                if field.constraints is not None:
                    lines.append(f"      constraints: {field.constraints}")
                lines.append(f"      description: {field.description}")
                for note in field.notes:
                    lines.append(f"      note: {note}")

    return "\n".join(lines).rstrip() + "\n"


def _normalize_topic(topic: str | None) -> str | None:
    if topic is None:
        return None
    normalized = topic.strip().lower().strip("[]")
    normalized = _TOPIC_ALIASES.get(normalized, normalized)
    valid_topics = set(manual_topics())
    if normalized not in valid_topics:
        options = ", ".join(manual_topics())
        msg = (
            f"Unknown manual topic {topic!r}. "
            f"Available topics: {options}"
        )
        raise ValueError(msg)
    return normalized


def _build_sections() -> tuple[ManualSection, ...]:
    return tuple(_build_section(spec) for spec in _SECTION_SPECS)


def _build_section(spec: SectionSpec) -> ManualSection:
    auto_fields = _auto_field_map(spec.config_class)
    used_attrs: set[str] = set()
    rendered_fields: list[ManualField] = []

    for field_spec in spec.fields:
        attr = field_spec.attr if field_spec.attr is not None else field_spec.key
        used_attrs.add(attr)
        auto = auto_fields.get(attr)
        type_name = auto.type_name if auto is not None else "object"
        required = (
            field_spec.required
            if field_spec.required is not None
            else (auto.required if auto is not None else False)
        )
        default_repr = field_spec.default_override
        if default_repr is None and auto is not None:
            default_repr = auto.default_repr
        rendered_fields.append(
            ManualField(
                key=field_spec.key,
                type_name=type_name,
                required=required,
                default_repr=default_repr,
                description=field_spec.description,
                constraints=field_spec.constraints,
                notes=field_spec.notes,
            ),
        )

    for attr, auto in sorted(auto_fields.items()):
        if attr in used_attrs:
            continue
        rendered_fields.append(
            ManualField(
                key=attr,
                type_name=auto.type_name,
                required=auto.required,
                default_repr=auto.default_repr,
                description=(
                    "Auto-discovered field with no explicit manual entry yet."
                ),
                constraints=None,
            ),
        )

    return ManualSection(
        name=spec.name,
        description=spec.description,
        required=spec.required,
        early_return=spec.early_return,
        table=spec.table,
        fields=tuple(rendered_fields),
        notes=spec.notes,
    )


def _auto_field_map(config_class: str | None) -> dict[str, AutoField]:
    if config_class is None:
        return {}
    all_fields = _parsed_config_fields()
    if config_class not in all_fields:
        msg = f"Config class {config_class!r} not found in types.py"
        raise ValueError(msg)
    return all_fields[config_class]


@lru_cache(maxsize=1)
def _parsed_config_fields() -> dict[str, dict[str, AutoField]]:
    types_path = Path(__file__).with_name("types.py")
    module = ast.parse(types_path.read_text())
    class_names = {
        spec.config_class
        for spec in _SECTION_SPECS
        if spec.config_class is not None
    }

    parsed: dict[str, dict[str, AutoField]] = {}
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name in class_names:
            parsed[node.name] = _parse_class_fields(node)
    return parsed


def _parse_class_fields(class_node: ast.ClassDef) -> dict[str, AutoField]:
    parsed: dict[str, AutoField] = {}
    for node in class_node.body:
        if not isinstance(node, ast.AnnAssign):
            continue
        if not isinstance(node.target, ast.Name):
            continue

        field_name = node.target.id
        type_name = _format_type_expr(node.annotation)
        default_repr, required = _default_from_assignment(node.value)
        parsed[field_name] = AutoField(
            type_name=type_name,
            default_repr=default_repr,
            required=required,
        )
    return parsed


def _default_from_assignment(
    value: ast.expr | None,
) -> tuple[str | None, bool]:
    if value is None:
        return None, True

    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "field"
    ):
        return _default_from_field_call(value)

    return _default_from_expr(value), False


def _default_from_field_call(call: ast.Call) -> tuple[str | None, bool]:
    for keyword in call.keywords:
        if keyword.arg == "default":
            return _default_from_expr(keyword.value), False
        if keyword.arg == "default_factory":
            return _default_from_factory_expr(keyword.value), False
    return None, True


def _default_from_factory_expr(expr: ast.expr) -> str:
    if isinstance(expr, ast.Name):
        if expr.id == "list":
            return "[]"
        if expr.id == "dict":
            return "{}"

    if isinstance(expr, ast.Lambda):
        return _default_from_expr(expr.body)

    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
        if expr.func.id == "list":
            return "[]"
        if expr.func.id == "dict":
            return "{}"

    return ast.unparse(expr)


def _default_from_expr(expr: ast.expr) -> str:
    try:
        literal = ast.literal_eval(expr)
    except (SyntaxError, TypeError, ValueError):
        return ast.unparse(expr)
    return _format_default(literal)


def _format_type_expr(annotation: ast.expr) -> str:
    text = ast.unparse(annotation)
    return text.replace("NoneType", "None")


def _format_default(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, Path):
        return json.dumps(str(value))
    if isinstance(value, list):
        inner = ", ".join(_format_default(item) for item in value)
        return f"[{inner}]"
    if isinstance(value, tuple):
        inner = ", ".join(_format_default(item) for item in value)
        return f"[{inner}]"
    if isinstance(value, dict):
        pairs = ", ".join(
            f"{json.dumps(str(key))}: {_format_default(item)}"
            for key, item in value.items()
        )
        return f"{{{pairs}}}"
    return json.dumps(str(value))


def _select_sections(
    sections: tuple[ManualSection, ...],
    topic: str | None,
) -> tuple[ManualSection, ...]:
    if topic is None or topic == "all":
        return sections
    if topic in {
        "quickstart",
        "commands",
        "validate",
        "playbook",
        "quick",
        "examples",
        "print",
        "modes",
        "formats",
    }:
        return ()
    for section in sections:
        if section.name == topic:
            return (section,)
    return ()


def _format_section_header(section: ManualSection) -> str:
    if section.table:
        return f"SECTION [{section.name}]"
    return f"SECTION {section.name} (top-level key)"


def _field_path(section: ManualSection, field: ManualField) -> str:
    if section.table:
        return f"[{section.name}].{field.key}"
    return field.key


__all__ = [
    "EARLY_RETURN_PRECEDENCE",
    "manual_topics",
    "render_manual",
]
