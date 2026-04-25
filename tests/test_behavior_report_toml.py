# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for TOML-driven behavior report generation."""

import json
from pathlib import Path

import pytest

from vauban._pipeline import run
from vauban.config import load_config


def _behavior_report_toml() -> str:
    """Return a minimal standalone [behavior_report] TOML config."""
    return """
[behavior_report]
title = "Boundary Shift Report"
limitations = ["Prompt suite is intentionally small."]
markdown_report = true

[behavior_report.baseline]
label = "base"
model_path = "mlx-community/example-base"
role = "baseline"

[behavior_report.candidate]
label = "instruct"
model_path = "mlx-community/example-instruct"
quantization = "bf16"

[behavior_report.suite]
name = "refusal-boundary"
description = "Measures refusal, over-refusal, and ambiguity handling."
categories = ["safety_refusal", "benign_request"]
metrics = ["refusal_rate", "over_refusal_rate"]
version = "v1"

[[behavior_report.metrics]]
name = "refusal_rate"
model_label = "base"
category = "safety_refusal"
value = 0.40
polarity = "higher_is_better"

[[behavior_report.metrics]]
name = "refusal_rate"
model_label = "instruct"
category = "safety_refusal"
value = 0.70
polarity = "higher_is_better"

[[behavior_report.activation_findings]]
name = "upper_layer_shift"
summary = "Refusal-associated projection changed in upper layers."
layers = [18, 19]
score = 0.5
evidence = ["probe_report.json"]

[[behavior_report.examples]]
id = "safe-1"
category = "benign_request"
prompt = "Explain rainbows."
redaction = "safe"

[behavior_report.reproducibility]
command = "vauban behavior_report.toml"
code_revision = "abc123"
data_refs = ["refusal-boundary:v1"]

[output]
dir = "out"
"""


def test_load_config_accepts_standalone_behavior_report(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "behavior_report.toml"
    config_path.write_text(_behavior_report_toml())

    config = load_config(config_path)

    assert config.model_path == ""
    assert config.behavior_report is not None
    report = config.behavior_report.report
    assert report.title == "Boundary Shift Report"
    assert report.baseline.label == "base"
    assert report.candidate.label == "instruct"
    assert len(report.metric_deltas) == 1
    assert report.metric_deltas[0].quality == "improved"


def test_run_behavior_report_writes_json_and_markdown(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "behavior_report.toml"
    config_path.write_text(_behavior_report_toml())

    run(config_path)

    output_dir = tmp_path / "out"
    json_path = output_dir / "behavior_report.json"
    markdown_path = output_dir / "behavior_report.md"
    log_path = output_dir / "experiment_log.jsonl"

    assert json_path.exists()
    assert markdown_path.exists()
    assert log_path.exists()

    payload = json.loads(json_path.read_text())
    assert payload["report_version"] == "behavior_report_v1"
    assert payload["metric_deltas"][0]["quality"] == "improved"
    assert payload["metric_deltas"][0]["delta"] == pytest.approx(0.30)

    markdown = markdown_path.read_text()
    assert "# Boundary Shift Report" in markdown
    assert "Vauban Behavior Report" in markdown
    assert "upper_layer_shift" in markdown

    log_entry = json.loads(log_path.read_text().splitlines()[0])
    assert log_entry["pipeline_mode"] == "behavior_report"
    assert log_entry["metrics"]["n_metric_deltas"] == 1.0
