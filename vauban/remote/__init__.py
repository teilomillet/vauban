"""Remote model probing via pluggable inference backends.

Backends are registered in ``_registry`` and resolved by name from
the ``[remote].backend`` config field.

Usage::

    from vauban.remote import run_remote_probe

    result = run_remote_probe(cfg=remote_cfg, api_key=key, output_dir=out)
"""

from vauban.remote._probe import run_remote_probe

__all__ = ["run_remote_probe"]
