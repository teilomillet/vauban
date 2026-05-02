# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Entry point for `python -m vauban` and `vauban` console script.

Usage:
    vauban <config.toml>
    vauban --validate <config.toml>
    vauban schema [--output FILE]
    vauban init [--mode MODE] [--model PATH] [--scenario NAME] [--output FILE] [--force]
    vauban diff [--format text|markdown] [--threshold FLOAT] <dir_a> <dir_b>
    vauban verify-bundle [--base-dir DIR] [--secret-env ENV]
                         [--require-signature] <ai_act_integrity.json>
    vauban tree [directory] [--format text|mermaid] [--status STATUS] [--tag TAG]
    vauban man [topic]
"""

import difflib
import json
import sys
from pathlib import Path

USAGE = """\
Usage: vauban [--validate] <config.toml>
       vauban schema [--output FILE]
       vauban init [--mode MODE] [--model PATH] [--scenario NAME]
                   [--output FILE] [--force]
       vauban diff [--format text|markdown] [--threshold FLOAT] <dir_a> <dir_b>
       vauban verify-bundle [--base-dir DIR] [--secret-env ENV]
                            [--require-signature] <ai_act_integrity.json>
       vauban tree [directory] [--format text|mermaid] [--status STATUS] [--tag TAG]
       vauban man [topic]

Run the full measure -> cut -> evaluate pipeline from a TOML config file.
All configuration lives in the TOML file.

Commands:
  schema          Print the current JSON Schema for Vauban TOML configs.
  init            Generate a starter TOML config file.
  diff            Compare JSON reports from two output directories.
                  Use --threshold as a CI gate (exit 1 on large deltas).
  verify-bundle   Verify an AI Act readiness integrity manifest and signatures.
  tree            Render the experiment lineage tree from TOML configs.
  man             Manual and workflow guides.
                  Start with 'vauban man workflows' to pick a goal.
                  Run 'vauban man <topic>' for config section details.

Options:
  --validate    Check config for errors without loading the model.
  -h, --help    Show this message and exit.
"""

_INIT_USAGE = (
    "Usage: vauban init [--mode MODE] [--model PATH] [--scenario NAME]"
    " [--output FILE] [--force]\n"
    "\n"
    "Notes:\n"
    "  ai_act mode also scaffolds draft evidence templates in ./evidence/\n"
    "  --scenario implies softprompt mode when --mode is omitted\n"
)

_DIFF_USAGE = (
    "Usage: vauban diff [--format text|markdown]"
    " [--threshold FLOAT] <dir_a> <dir_b>\n"
)

_SCHEMA_USAGE = "Usage: vauban schema [--output FILE]\n"
_VERIFY_BUNDLE_USAGE = (
    "Usage: vauban verify-bundle [--base-dir DIR] [--secret-env ENV]"
    " [--require-signature] [--format text|json] <ai_act_integrity.json>\n"
)
_DIFF_HELP = (
    "Usage: vauban diff [--format text|markdown]"
    " [--threshold FLOAT] <dir_a> <dir_b>\n"
    "\n"
    "Options:\n"
    "  --format text|markdown   Output format (default: text).\n"
    "  --threshold FLOAT        CI gate:"
    " exit 1 if any absolute metric delta exceeds threshold.\n"
)


def _command_suggestion(token: str) -> str | None:
    """Return a suggested command for typo'd subcommands."""
    candidates = (
        "man",
        "init",
        "diff",
        "schema",
        "tree",
        "validate",
        "verify-bundle",
    )
    aliases = {"validate": "--validate"}
    matches = difflib.get_close_matches(token, candidates, n=1, cutoff=0.6)
    if not matches:
        return None
    return aliases.get(matches[0], matches[0])


def _format_mode_list() -> str:
    """Format mode names with descriptions for help output."""
    from vauban._init import KNOWN_MODES, MODE_DESCRIPTIONS

    lines: list[str] = []
    for mode in sorted(KNOWN_MODES):
        desc = MODE_DESCRIPTIONS.get(mode, "")
        lines.append(f"  {mode:<20s} {desc}")
    return "Modes:\n" + "\n".join(lines) + "\n"


def _format_scenario_list() -> str:
    """Format built-in environment benchmark scenario names for help output."""
    from vauban.environment import list_scenarios

    names = ", ".join(list_scenarios())
    return f"Scenarios:\n  {names}\n"


def _run_init(args: list[str]) -> None:
    """Handle `vauban init` subcommand."""
    from vauban._init import init_config

    if len(args) == 1 and args[0] in ("--help", "-h"):
        sys.stdout.write(_INIT_USAGE)
        sys.stdout.write(_format_mode_list())
        sys.stdout.write(_format_scenario_list())
        return

    mode = "default"
    model = "Qwen/Qwen2.5-1.5B-Instruct"
    scenario: str | None = None
    output = "run.toml"
    force = False

    i = 0
    while i < len(args):
        if args[i] in ("--help", "-h"):
            sys.stdout.write(_INIT_USAGE)
            sys.stdout.write(_format_mode_list())
            sys.stdout.write(_format_scenario_list())
            return
        if args[i] == "--mode" and i + 1 < len(args):
            mode = args[i + 1]
            i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        elif args[i] == "--scenario" and i + 1 < len(args):
            scenario = args[i + 1]
            i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            output = args[i + 1]
            i += 2
        elif args[i] == "--force":
            force = True
            i += 1
        else:
            sys.stderr.write(
                f"Error: unexpected argument {args[i]!r}\n\n",
            )
            sys.stderr.write(_INIT_USAGE)
            sys.stderr.write(_format_mode_list())
            sys.stderr.write(_format_scenario_list())
            raise SystemExit(1)

    output_path = Path(output)
    try:
        init_config(
            mode,
            model,
            output_path,
            force=force,
            scenario=scenario,
        )
    except (ValueError, FileExistsError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        raise SystemExit(1) from exc

    if scenario is not None:
        label = f"softprompt + {scenario} scenario"
    else:
        label = mode if mode != "default" else "measure → cut → export"
    sys.stderr.write(f"Created {output_path} ({label} mode)\n")
    if mode in ("ai_act", "public_sector_readiness"):
        sys.stderr.write(
            (
                "Scaffolded draft AI Act evidence templates in"
                f" {output_path.parent / 'evidence'}\n"
            ),
        )
        sys.stderr.write(
            "Replace draft fields before treating those documents as evidence.\n",
        )
    sys.stderr.write(f"Next: vauban --validate {output_path}\n")


def _run_diff(args: list[str]) -> None:
    """Handle ``vauban diff`` with optional --format and --threshold flags."""
    if len(args) == 1 and args[0] in ("--help", "-h"):
        sys.stdout.write(_DIFF_HELP)
        raise SystemExit(0)

    fmt = "text"
    threshold: float | None = None
    positional: list[str] = []

    i = 0
    while i < len(args):
        if args[i] in ("--help", "-h"):
            sys.stdout.write(_DIFF_HELP)
            raise SystemExit(0)
        if args[i] == "--format" and i + 1 < len(args):
            fmt = args[i + 1]
            if fmt not in ("text", "markdown"):
                sys.stderr.write(
                    f"Error: --format must be 'text' or 'markdown',"
                    f" got {fmt!r}\n",
                )
                raise SystemExit(1)
            i += 2
        elif args[i] == "--threshold" and i + 1 < len(args):
            try:
                threshold = float(args[i + 1])
            except ValueError as exc:
                sys.stderr.write(
                    f"Error: --threshold must be a number,"
                    f" got {args[i + 1]!r}\n",
                )
                raise SystemExit(1) from exc
            i += 2
        elif args[i].startswith("--"):
            sys.stderr.write(
                f"Error: unexpected flag {args[i]!r}\n\n",
            )
            sys.stderr.write(_DIFF_USAGE)
            raise SystemExit(1)
        else:
            positional.append(args[i])
            i += 1

    if len(positional) != 2:
        sys.stderr.write(
            f"Error: expected 2 directory paths, got {len(positional)}\n\n",
        )
        sys.stderr.write(_DIFF_USAGE)
        raise SystemExit(1)

    from vauban._diff import diff_reports, format_diff, format_diff_markdown

    dir_a = Path(positional[0])
    dir_b = Path(positional[1])

    try:
        reports = diff_reports(dir_a, dir_b)
    except FileNotFoundError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        raise SystemExit(1) from exc

    if fmt == "markdown":
        sys.stdout.write(format_diff_markdown(dir_a, dir_b, reports))
    else:
        sys.stdout.write(format_diff(dir_a, dir_b, reports))

    # Threshold exit code
    if threshold is not None:
        for report in reports:
            for m in report.metrics:
                if abs(m.delta) > threshold:
                    raise SystemExit(1)
    raise SystemExit(0)


def _run_schema(args: list[str]) -> None:
    """Handle ``vauban schema`` with optional output path."""
    if len(args) == 1 and args[0] in ("--help", "-h"):
        sys.stdout.write(_SCHEMA_USAGE)
        raise SystemExit(0)

    output_path: Path | None = None
    i = 0
    while i < len(args):
        if args[i] in ("--help", "-h"):
            sys.stdout.write(_SCHEMA_USAGE)
            raise SystemExit(0)
        if args[i] == "--output" and i + 1 < len(args):
            output_path = Path(args[i + 1])
            i += 2
            continue
        sys.stderr.write(f"Error: unexpected argument {args[i]!r}\n\n")
        sys.stderr.write(_SCHEMA_USAGE)
        raise SystemExit(1)

    from vauban.config import generate_config_schema, write_config_schema

    if output_path is None:
        sys.stdout.write(json.dumps(generate_config_schema(), indent=2) + "\n")
        raise SystemExit(0)

    write_config_schema(output_path)
    sys.stderr.write(f"Wrote schema to {output_path}\n")
    raise SystemExit(0)


def _run_verify_bundle(args: list[str]) -> None:
    """Handle ``vauban verify-bundle`` for AI Act integrity manifests."""
    if len(args) == 1 and args[0] in ("--help", "-h"):
        sys.stdout.write(_VERIFY_BUNDLE_USAGE)
        raise SystemExit(0)

    base_dir: Path | None = None
    secret_env: str | None = None
    require_signature = False
    fmt = "text"
    positional: list[str] = []

    i = 0
    while i < len(args):
        if args[i] in ("--help", "-h"):
            sys.stdout.write(_VERIFY_BUNDLE_USAGE)
            raise SystemExit(0)
        if args[i] == "--base-dir" and i + 1 < len(args):
            base_dir = Path(args[i + 1])
            i += 2
            continue
        if args[i] == "--secret-env" and i + 1 < len(args):
            secret_env = args[i + 1]
            i += 2
            continue
        if args[i] == "--require-signature":
            require_signature = True
            i += 1
            continue
        if args[i] == "--format" and i + 1 < len(args):
            fmt = args[i + 1]
            if fmt not in ("text", "json"):
                sys.stderr.write(
                    f"Error: --format must be 'text' or 'json', got {fmt!r}\n",
                )
                raise SystemExit(1)
            i += 2
            continue
        if args[i].startswith("--"):
            sys.stderr.write(f"Error: unexpected flag {args[i]!r}\n\n")
            sys.stderr.write(_VERIFY_BUNDLE_USAGE)
            raise SystemExit(1)
        positional.append(args[i])
        i += 1

    if len(positional) != 1:
        sys.stderr.write(
            f"Error: expected 1 integrity manifest path, got {len(positional)}\n\n",
        )
        sys.stderr.write(_VERIFY_BUNDLE_USAGE)
        raise SystemExit(1)

    from vauban.integrity import (
        format_integrity_verification,
        verify_ai_act_integrity,
    )

    try:
        result = verify_ai_act_integrity(
            positional[0],
            base_dir=base_dir,
            secret_env=secret_env,
            require_signature=require_signature,
        )
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        raise SystemExit(1) from exc

    if fmt == "json":
        sys.stdout.write(json.dumps(result.to_dict(), indent=2) + "\n")
    else:
        sys.stdout.write(format_integrity_verification(result))
    raise SystemExit(0 if result.passed else 1)


def _run_tree(args: list[str]) -> None:
    """Handle ``vauban tree`` by delegating to the tree viewer CLI."""
    from vauban.tree import main as tree_main

    try:
        tree_main(args)
    except SystemExit:
        raise
    raise SystemExit(0)


def _set_backend_from_config(path: str) -> None:
    """Read backend from TOML and set VAUBAN_BACKEND env var.

    Called before ``import vauban`` so the env var is visible at import time.
    Errors here are intentionally swallowed — they will surface later in
    ``load_config()``.
    """
    import os
    import tomllib

    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        backend = raw.get("backend")
        if isinstance(backend, str):
            os.environ["VAUBAN_BACKEND"] = backend
        elif "VAUBAN_BACKEND" not in os.environ:
            os.environ["VAUBAN_BACKEND"] = "torch"
    except Exception:
        pass  # Fall through — config errors will surface in load_config()


def main() -> None:
    """Parse sys.argv and delegate to vauban.run() or vauban.validate()."""
    args = sys.argv[1:]

    if not args or args[0] in ("--help", "-h"):
        sys.stdout.write(USAGE)
        raise SystemExit(0 if args else 1)

    if args[0] == "--man":
        sys.stderr.write(
            "Error: '--man' is no longer supported. "
            "Use 'vauban man [topic]'.\n\n",
        )
        sys.stderr.write(USAGE)
        raise SystemExit(1)

    if args[0] == "man":
        topic: str | None = None
        if len(args) > 2:
            sys.stderr.write(
                f"Error: expected at most one manual topic, got {len(args) - 1}\n\n",
            )
            sys.stderr.write(USAGE)
            raise SystemExit(1)
        if len(args) == 2:
            topic = args[1]
        from vauban.manual import render_manual

        try:
            sys.stdout.write(render_manual(topic))
        except ValueError as exc:
            sys.stderr.write(f"Error: {exc}\n")
            raise SystemExit(1) from exc
        raise SystemExit(0)

    if args[0] == "init":
        _run_init(args[1:])
        raise SystemExit(0)

    if args[0] == "diff":
        _run_diff(args[1:])
        raise SystemExit(0)

    if args[0] == "schema":
        _run_schema(args[1:])
        raise SystemExit(0)

    if args[0] == "verify-bundle":
        _run_verify_bundle(args[1:])
        raise SystemExit(0)

    if args[0] == "tree":
        _run_tree(args[1:])
        raise SystemExit(0)

    known_prefixes = {"--validate", "--help", "-h"}
    if args[0] not in known_prefixes:
        arg0_path = Path(args[0])
        suggestion = _command_suggestion(args[0])
        if suggestion is not None and not arg0_path.exists():
            sys.stderr.write(
                f"Error: unknown command {args[0]!r}. "
                f"Did you mean {suggestion!r}?\n\n",
            )
            sys.stderr.write(USAGE)
            raise SystemExit(1)

    validate_mode = False
    if args[0] == "--validate":
        validate_mode = True
        args = args[1:]

    if len(args) != 1:
        sys.stderr.write(
            f"Error: expected 1 config path, got {len(args)}\n\n",
        )
        sys.stderr.write(USAGE)
        raise SystemExit(1)

    config_path = args[0]

    # Peek at backend before importing vauban (sets env var for import-time dispatch)
    _set_backend_from_config(config_path)

    from vauban._pipeline._run import run as run_pipeline
    from vauban.config._validation import validate_config

    try:
        if validate_mode:
            warnings = validate_config(config_path)
            raise SystemExit(1 if warnings else 0)
        run_pipeline(config_path)
    except SystemExit:
        raise
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
