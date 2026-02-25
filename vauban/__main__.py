"""Entry point for `python -m vauban` and `vauban` console script.

Usage:
    vauban <config.toml>
    vauban --validate <config.toml>
    vauban init [--mode MODE] [--model PATH] [--output FILE] [--force]
    vauban diff [--format text|markdown] [--threshold FLOAT] <dir_a> <dir_b>
    vauban man [topic]
"""

import difflib
import sys
from pathlib import Path

USAGE = """\
Usage: vauban [--validate] <config.toml>
       vauban init [--mode MODE] [--model PATH] [--output FILE] [--force]
       vauban diff [--format text|markdown] [--threshold FLOAT] <dir_a> <dir_b>
       vauban man [topic]

Run the full measure -> cut -> evaluate pipeline from a TOML config file.
All configuration lives in the TOML file.

Commands:
  init            Generate a starter TOML config file.
  diff            Compare JSON reports from two output directories.
                  Use --threshold as a CI gate (exit 1 on large deltas).
  man             Show built-in manual (topics: quickstart, commands,
                  validate, playbook, quick, examples, print, modes, formats,
                  model, data, measure, cut, eval, surface, detect, optimize,
                  softprompt, sic, depth, probe, steer, output, verbose).

Options:
  --validate    Check config for errors without loading the model.
  -h, --help    Show this message and exit.
"""

_INIT_USAGE = (
    "Usage: vauban init [--mode MODE] [--model PATH]"
    " [--output FILE] [--force]\n"
)

_DIFF_USAGE = (
    "Usage: vauban diff [--format text|markdown]"
    " [--threshold FLOAT] <dir_a> <dir_b>\n"
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
    candidates = ("man", "init", "diff", "validate")
    aliases = {"validate": "--validate"}
    matches = difflib.get_close_matches(token, candidates, n=1, cutoff=0.6)
    if not matches:
        return None
    return aliases.get(matches[0], matches[0])


def _run_init(args: list[str]) -> None:
    """Handle `vauban init` subcommand."""
    from vauban._init import KNOWN_MODES, init_config

    if len(args) == 1 and args[0] in ("--help", "-h"):
        sys.stdout.write(_INIT_USAGE)
        sys.stdout.write(f"Modes: {', '.join(sorted(KNOWN_MODES))}\n")
        return

    mode = "default"
    model = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    output = "run.toml"
    force = False

    i = 0
    while i < len(args):
        if args[i] in ("--help", "-h"):
            sys.stdout.write(_INIT_USAGE)
            sys.stdout.write(f"Modes: {', '.join(sorted(KNOWN_MODES))}\n")
            return
        if args[i] == "--mode" and i + 1 < len(args):
            mode = args[i + 1]
            i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            model = args[i + 1]
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
            sys.stderr.write(f"Modes: {', '.join(sorted(KNOWN_MODES))}\n")
            raise SystemExit(1)

    output_path = Path(output)
    try:
        init_config(mode, model, output_path, force=force)
    except (ValueError, FileExistsError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        raise SystemExit(1) from exc

    label = mode if mode != "default" else "measure → cut → export"
    sys.stderr.write(f"Created {output_path} ({label} mode)\n")
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

    import vauban

    try:
        if validate_mode:
            warnings = vauban.validate(config_path)
            raise SystemExit(1 if warnings else 0)
        vauban.run(config_path)
    except SystemExit:
        raise
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
