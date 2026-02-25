"""Entry point for `python -m vauban` and `vauban` console script.

Usage:
    vauban <config.toml>
    vauban --validate <config.toml>
    vauban init [--mode MODE] [--model PATH] [--output FILE] [--force]
    vauban diff <dir_a> <dir_b>
    vauban man [topic]
"""

import sys

USAGE = """\
Usage: vauban [--validate] <config.toml>
       vauban init [--mode MODE] [--model PATH] [--output FILE] [--force]
       vauban diff <dir_a> <dir_b>
       vauban man [topic]

Run the full measure -> cut -> evaluate pipeline from a TOML config file.
All configuration lives in the TOML file.

Commands:
  init            Generate a starter TOML config file.
  diff            Compare JSON reports from two output directories.
  man             Show built-in manual (topics: quickstart, modes, formats,
                  model, data, measure, cut, eval, surface, detect, optimize,
                  softprompt, sic, depth, probe, steer, output, verbose).

Options:
  --validate    Check config for errors without loading the model.
  -h, --help    Show this message and exit.
"""


def _run_init(args: list[str]) -> None:
    """Handle `vauban init` subcommand."""
    from pathlib import Path

    from vauban._init import KNOWN_MODES, init_config

    mode = "default"
    model = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    output = "run.toml"
    force = False

    i = 0
    while i < len(args):
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
            sys.stderr.write(
                f"Usage: vauban init [--mode MODE] [--model PATH]"
                f" [--output FILE] [--force]\n"
                f"Modes: {', '.join(sorted(KNOWN_MODES))}\n",
            )
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
    """Handle `vauban diff <dir_a> <dir_b>` subcommand."""
    from pathlib import Path

    if len(args) != 2:
        sys.stderr.write(
            f"Error: expected 2 directory paths, got {len(args)}\n\n",
        )
        sys.stderr.write("Usage: vauban diff <dir_a> <dir_b>\n")
        raise SystemExit(1)

    from vauban._diff import diff_reports, format_diff

    dir_a = Path(args[0])
    dir_b = Path(args[1])

    try:
        reports = diff_reports(dir_a, dir_b)
    except FileNotFoundError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        raise SystemExit(1) from exc

    sys.stdout.write(format_diff(dir_a, dir_b, reports))


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
