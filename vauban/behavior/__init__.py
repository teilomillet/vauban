# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Behavior-change report primitives for model behavior auditing."""

from vauban.behavior._markdown import render_behavior_report_markdown
from vauban.behavior._primitives import (
    ActivationFinding,
    BehaviorChatMessage,
    BehaviorExample,
    BehaviorMetric,
    BehaviorMetricDelta,
    BehaviorMetricSpec,
    BehaviorPrompt,
    BehaviorReport,
    BehaviorSuite,
    BehaviorSuiteRef,
    ChatRole,
    ExampleRedaction,
    ExpectedBehavior,
    FindingSeverity,
    JsonScalar,
    JsonValue,
    MetricPolarity,
    MetricQuality,
    ModelRole,
    ReportModelRef,
    ReproducibilityInfo,
    compare_behavior_metrics,
)

__all__ = [
    "ActivationFinding",
    "BehaviorChatMessage",
    "BehaviorExample",
    "BehaviorMetric",
    "BehaviorMetricDelta",
    "BehaviorMetricSpec",
    "BehaviorPrompt",
    "BehaviorReport",
    "BehaviorSuite",
    "BehaviorSuiteRef",
    "ChatRole",
    "ExampleRedaction",
    "ExpectedBehavior",
    "FindingSeverity",
    "JsonScalar",
    "JsonValue",
    "MetricPolarity",
    "MetricQuality",
    "ModelRole",
    "ReportModelRef",
    "ReproducibilityInfo",
    "compare_behavior_metrics",
    "render_behavior_report_markdown",
]
