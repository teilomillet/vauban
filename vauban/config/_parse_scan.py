# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Parse the [scan] section of a TOML config."""

from typing import cast

from vauban.config._types import TomlDict
from vauban.types import ScanConfig


def _parse_scan(raw: TomlDict) -> ScanConfig | None:
    """Parse the optional [scan] section into a ScanConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("scan")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[scan] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    scan_dict = cast("dict[str, object]", sec)

    # -- target_layer --
    target_layer_raw = scan_dict.get("target_layer")
    target_layer: int | None = None
    if target_layer_raw is not None:
        if not isinstance(target_layer_raw, int):
            msg = (
                f"[scan].target_layer must be an integer,"
                f" got {type(target_layer_raw).__name__}"
            )
            raise TypeError(msg)
        target_layer = target_layer_raw

    # -- span_threshold --
    span_threshold_raw = scan_dict.get("span_threshold", 0.5)
    if not isinstance(span_threshold_raw, int | float):
        msg = (
            f"[scan].span_threshold must be a number,"
            f" got {type(span_threshold_raw).__name__}"
        )
        raise TypeError(msg)

    # -- threshold --
    threshold_raw = scan_dict.get("threshold", 0.0)
    if not isinstance(threshold_raw, int | float):
        msg = (
            f"[scan].threshold must be a number,"
            f" got {type(threshold_raw).__name__}"
        )
        raise TypeError(msg)

    # -- calibrate --
    calibrate_raw = scan_dict.get("calibrate", False)
    if not isinstance(calibrate_raw, bool):
        msg = (
            f"[scan].calibrate must be a boolean,"
            f" got {type(calibrate_raw).__name__}"
        )
        raise TypeError(msg)

    return ScanConfig(
        target_layer=target_layer,
        span_threshold=float(span_threshold_raw),
        threshold=float(threshold_raw),
        calibrate=calibrate_raw,
    )
