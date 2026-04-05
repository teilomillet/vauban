# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Parse the [objective] section of a TOML config."""

from typing import Literal, cast

from vauban._objective import (
    FLYWHEEL_OBJECTIVE_METRICS,
    OBJECTIVE_ACCESS_MODES,
    OBJECTIVE_AGGREGATES,
    OBJECTIVE_COMPARISONS,
)
from vauban.config._parse_helpers import SectionReader, require_toml_table
from vauban.config._types import TomlDict
from vauban.types import ObjectiveConfig, ObjectiveMetricSpec


def _parse_objective(raw: TomlDict) -> ObjectiveConfig | None:
    """Parse the optional [objective] section into an ObjectiveConfig."""
    sec = raw.get("objective")
    if sec is None:
        return None
    reader = SectionReader("[objective]", require_toml_table("[objective]", sec))

    name = reader.string("name")
    if not name:
        msg = "[objective].name must be non-empty"
        raise ValueError(msg)

    deployment = reader.string("deployment", default="")
    summary = reader.string("summary", default="")
    access = reader.literal(
        "access",
        cast(
            'tuple[Literal["weights", "api", "hybrid", "system"], ...]',
            OBJECTIVE_ACCESS_MODES,
        ),
        default="system",
    )
    preserve = reader.optional_string_list("preserve") or []
    prevent = reader.optional_string_list("prevent") or []

    safety = _parse_metric_group(
        reader.data.get("safety"),
        "[objective].safety",
        default_comparison="at_most",
    )
    utility = _parse_metric_group(
        reader.data.get("utility"),
        "[objective].utility",
        default_comparison="at_least",
    )

    if not safety and not utility:
        msg = (
            "[objective] must define at least one [[objective.safety]] "
            "or [[objective.utility]] threshold"
        )
        raise ValueError(msg)

    return ObjectiveConfig(
        name=name,
        deployment=deployment,
        summary=summary,
        access=access,
        preserve=preserve,
        prevent=prevent,
        safety=safety,
        utility=utility,
    )


def _parse_metric_group(
    raw: object,
    field_name: str,
    *,
    default_comparison: Literal["at_least", "at_most"],
) -> list[ObjectiveMetricSpec]:
    """Parse one objective metric group from an array of tables."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        msg = f"{field_name} must be an array of tables, got {type(raw).__name__}"
        raise TypeError(msg)

    specs: list[ObjectiveMetricSpec] = []
    table_prefix = field_name.replace("[", "").replace("]", "")
    for index, entry in enumerate(raw):
        table_name = f"[[{table_prefix}]][{index}]"
        table = require_toml_table(table_name, entry)
        reader = SectionReader(table_name, table)

        metric = reader.string("metric")
        if metric not in FLYWHEEL_OBJECTIVE_METRICS:
            msg = (
                f"{reader.section}.metric must be one of "
                f"{sorted(FLYWHEEL_OBJECTIVE_METRICS)!r}, got {metric!r}"
            )
            raise ValueError(msg)

        threshold = reader.number("threshold")
        comparison = reader.literal(
            "comparison",
            cast(
                'tuple[Literal["at_least", "at_most"], ...]',
                OBJECTIVE_COMPARISONS,
            ),
            default=default_comparison,
        )
        aggregate = reader.literal(
            "aggregate",
            cast(
                'tuple[Literal["final", "mean", "min", "max"], ...]',
                OBJECTIVE_AGGREGATES,
            ),
            default="final",
        )
        label = reader.string("label", default="")
        description = reader.string("description", default="")

        specs.append(ObjectiveMetricSpec(
            metric=metric,
            threshold=threshold,
            comparison=comparison,
            aggregate=aggregate,
            label=label,
            description=description,
        ))
    return specs
