"""Entry point for `python -m vauban` and `vauban` console script.

Usage: vauban <config.toml>
"""

import sys

USAGE = """\
Usage: vauban <config.toml>

Run the full measure → cut → evaluate pipeline from a TOML config file.
All configuration lives in the TOML file — no flags, no subcommands.
"""


def main() -> None:
    """Parse sys.argv for a single TOML path and delegate to vauban.run()."""
    args = sys.argv[1:]

    if not args or args[0] in ("--help", "-h"):
        sys.stdout.write(USAGE)
        raise SystemExit(0 if args else 1)

    if len(args) > 1:
        sys.stderr.write(f"Error: expected 1 argument, got {len(args)}\n\n")
        sys.stderr.write(USAGE)
        raise SystemExit(1)

    config_path = args[0]

    import vauban

    try:
        vauban.run(config_path)
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
