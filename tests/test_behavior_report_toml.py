# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for TOML-driven behavior report generation."""

import json
from pathlib import Path

import pytest

from vauban._init import init_config
from vauban._pipeline import run
from vauban.config import load_config


def _behavior_report_toml() -> str:
    """Return a minimal standalone [behavior_report] TOML config."""
    return """
[behavior_report]
title = "Boundary Shift Report"
target_change = "base -> instruct"
findings = [
  "Refusal behavior changed.",
  "Activation diagnostics suggest an upper-layer shift.",
]
recommendation = "Run additional benign-request regression testing before shipping."
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

[behavior_report.transformation]
kind = "fine_tune"
summary = "Base model compared with its instruction-tuned variant."
before = "base"
after = "instruct"
method = "instruction_tuning"

[behavior_report.access]
level = "base_and_transformed"
claim_strength = "model_change_audit"
available_evidence = ["paired_outputs", "activation_diagnostics"]
missing_evidence = ["training_data"]
notes = ["Internal claims require local activation access."]

[[behavior_report.evidence]]
id = "surface_report"
kind = "run_report"
path_or_url = "surface_report.json"
description = "Behavioral metric output from the boundary suite."

[[behavior_report.evidence]]
id = "probe_report"
kind = "activation"
path_or_url = "probe_report.json"
description = "Layer-wise activation projection diagnostics."

[[behavior_report.claims]]
id = "claim-refusal-shift"
statement = "Instruction tuning changed refusal behavior."
strength = "model_change_audit"
access_level = "base_and_transformed"
status = "extended"
evidence = ["surface_report", "probe_report"]
limitations = ["Small behavior suite."]

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

[[behavior_report.reproduction_targets]]
id = "arditi-2024-refusal-direction"
title = "Refusal in Language Models Is Mediated by a Single Direction"
source_url = "https://arxiv.org/abs/2406.11717"
original_claim = "Refusal behavior can be mediated by a one-dimensional direction."
planned_extension = "Separate safety refusal from benign over-refusal in the report."
status = "planned"

[[behavior_report.reproduction_results]]
target_id = "arditi-2024-refusal-direction"
status = "partially_replicated"
summary = "Small-suite run recovered positive separation."
replicated_claims = ["Positive direction separation."]
extensions = ["Recorded access-aware limitations."]
evidence = ["surface_report", "probe_report"]
limitations = ["Single model."]

[[behavior_report.intervention_results]]
id = "steer-refusal-direction-negative"
kind = "activation_steering"
summary = "Negative steering reduced the refusal-style metric in a controlled probe."
target = "refusal_direction_lite"
effect = "decreased"
polarity = "negative"
layers = [18, 19]
strength = -1.0
baseline_condition = "unsteered"
intervention_condition = "negative_alpha"
behavior_metric = "refusal_style_rate"
activation_metric = "mean_projection"
evidence = ["probe_report"]
limitations = ["Single controlled probe."]

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
    assert report.target_change == "base -> instruct"
    assert report.findings == (
        "Refusal behavior changed.",
        "Activation diagnostics suggest an upper-layer shift.",
    )
    assert (
        report.recommendation
        == "Run additional benign-request regression testing before shipping."
    )
    assert report.baseline.label == "base"
    assert report.candidate.label == "instruct"
    assert report.transformation is not None
    assert report.transformation.kind == "fine_tune"
    assert report.access is not None
    assert report.access.claim_strength == "model_change_audit"
    assert len(report.evidence) == 2
    assert report.claims[0].evidence == ("surface_report", "probe_report")
    assert report.reproduction_targets[0].source_url == (
        "https://arxiv.org/abs/2406.11717"
    )
    assert report.reproduction_results[0].target_id == (
        "arditi-2024-refusal-direction"
    )
    assert report.intervention_results[0].intervention_id == (
        "steer-refusal-direction-negative"
    )
    assert report.intervention_results[0].effect == "decreased"
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
    assert payload["target_change"] == "base -> instruct"
    assert payload["findings"] == [
        "Refusal behavior changed.",
        "Activation diagnostics suggest an upper-layer shift.",
    ]
    assert payload["transformation"]["kind"] == "fine_tune"
    assert payload["access"]["level"] == "base_and_transformed"
    assert payload["claims"][0]["status"] == "extended"
    assert payload["evidence"][1]["kind"] == "activation"
    assert payload["reproduction_targets"][0]["id"] == (
        "arditi-2024-refusal-direction"
    )
    assert payload["reproduction_results"][0]["status"] == "partially_replicated"
    assert payload["intervention_results"][0]["id"] == (
        "steer-refusal-direction-negative"
    )
    assert payload["intervention_results"][0]["kind"] == "activation_steering"
    assert payload["intervention_results"][0]["effect"] == "decreased"
    assert (
        payload["recommendation"]
        == "Run additional benign-request regression testing before shipping."
    )
    assert payload["metric_deltas"][0]["quality"] == "improved"
    assert payload["metric_deltas"][0]["delta"] == pytest.approx(0.30)

    markdown = markdown_path.read_text()
    assert "# Boundary Shift Report" in markdown
    assert "Model Behavior Change Report" in markdown
    assert "## Target Change" in markdown
    assert "base -> instruct" in markdown
    assert "## Transformation" in markdown
    assert "fine_tune" in markdown
    assert "## Access And Claim Strength" in markdown
    assert "base_and_transformed" in markdown
    assert "## Claims" in markdown
    assert "claim-refusal-shift" in markdown
    assert "## Reproduction Targets" in markdown
    assert "arditi-2024-refusal-direction" in markdown
    assert "## Reproduction Results" in markdown
    assert "Positive direction separation." in markdown
    assert "## Intervention Results" in markdown
    assert "steer-refusal-direction-negative" in markdown
    assert "negative_alpha" in markdown
    assert "## Findings" in markdown
    assert "Refusal behavior changed." in markdown
    assert "upper_layer_shift" in markdown
    assert "## Recommendation" in markdown

    log_entry = json.loads(log_path.read_text().splitlines()[0])
    assert log_entry["pipeline_mode"] == "behavior_report"
    assert log_entry["metrics"]["n_metric_deltas"] == 1.0
    assert log_entry["metrics"]["n_claims"] == 1.0
    assert log_entry["metrics"]["n_reproduction_targets"] == 1.0
    assert log_entry["metrics"]["n_reproduction_results"] == 1.0
    assert log_entry["metrics"]["n_intervention_results"] == 1.0


def test_init_behavior_report_scaffold_loads(tmp_path: Path) -> None:
    config_path = tmp_path / "behavior_report.toml"

    content = init_config(
        mode="behavior_report",
        output_path=config_path,
        force=True,
    )
    config = load_config(config_path)

    assert "[behavior_report]" in content
    assert "[model]" not in content
    assert config.model_path == ""
    assert config.behavior_report is not None
    assert config.behavior_report.report.access is not None
    assert config.behavior_report.report.access.level == "paired_outputs"
