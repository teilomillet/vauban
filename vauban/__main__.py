"""Entry point for `python -m vauban` and `vauban` console script.

Usage:
    vauban <config.toml>
    vauban --validate <config.toml>
    vauban man [topic]
"""

import sys

USAGE = """\
Usage: vauban [--validate] <config.toml>
       vauban man [topic]

Run the full measure -> cut -> evaluate pipeline from a TOML config file.
All configuration lives in the TOML file.

Options:
  --validate    Check config for errors without loading the model.
  man           Show built-in manual (topics: quickstart, modes, formats,
                model, data, measure, cut, eval, surface, detect, optimize,
                softprompt, sic, depth, probe, steer, output, verbose).
  -h, --help    Show this message and exit.
"""


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
