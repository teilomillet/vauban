# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Focused coverage tests for AI Act Article 50 and coverage-guard helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import vauban.ai_act as ai_act_module
from vauban.types import AIActConfig

if TYPE_CHECKING:
    from pathlib import Path


def _write_notice(path: Path, lines: list[str]) -> None:
    """Write a newline-terminated notice file for helper tests."""
    path.write_text("\n".join(lines) + "\n")


def _article50_config(
    notice: Path | None,
    *,
    deploys_deepfake: bool = False,
    deepfake_creative_context: bool = False,
    publishes_text: bool = False,
    public_interest_review: bool = False,
    public_interest_responsibility: bool = False,
) -> AIActConfig:
    """Build a minimal AI Act config for Article 50 helper coverage."""
    return AIActConfig(
        company_name="Example Energy",
        system_name="Article 50 Helper Case",
        intended_purpose="Exercises the Article 50 coverage branches.",
        transparency_notice=notice,
        deploys_deepfake_or_synthetic_media=deploys_deepfake,
        deepfake_creative_satirical_artistic_or_fictional_context=(
            deepfake_creative_context
        ),
        publishes_text_on_matters_of_public_interest=publishes_text,
        public_interest_text_human_review_or_editorial_control=(
            public_interest_review
        ),
        public_interest_text_editorial_responsibility=(
            public_interest_responsibility
        ),
    )


class TestArticle50Helpers:
    """Test the remaining Article 50 helper branches directly."""

    def test_deepfake_disclosure_covers_pass_unknown_and_fail_paths(
        self,
        tmp_path: Path,
    ) -> None:
        notice = tmp_path / "transparency.md"
        _write_notice(
            notice,
            [
                "This automated system notice explains a deepfake clip.",
                "The content is synthetic and manipulated.",
            ],
        )
        manifest = [{"evidence_id": "file.transparency_notice", "exists": True}]

        pass_entry = ai_act_module._evaluate_article50_deepfake(
            _article50_config(
                notice,
                deploys_deepfake=True,
                deepfake_creative_context=True,
            ),
            manifest,
        )
        assert str(pass_entry["status"]) == "pass"
        assert "limited disclosure path" in str(pass_entry["rationale"])

        _write_notice(
            notice,
            [
                "This automated system notice explains the content.",
                "The publication is described for readers.",
            ],
        )
        unknown_entry = ai_act_module._evaluate_article50_deepfake(
            _article50_config(notice, deploys_deepfake=True),
            manifest,
        )
        assert str(unknown_entry["status"]) == "unknown"
        assert "synthetic_or_manipulated_content" in str(
            unknown_entry["missing_markers"],
        )

        fail_entry = ai_act_module._evaluate_article50_deepfake(
            _article50_config(None, deploys_deepfake=True),
            [],
        )
        assert str(fail_entry["status"]) == "fail"
        assert "transparency_notice" in str(fail_entry["owner_action"])

    @pytest.mark.parametrize(
        (
            "notice_lines",
            "review",
            "responsibility",
            "manifest_exists",
            "expected_status",
        ),
        [
            (
                [
                    "This automated system notice informs the public.",
                    "The publication is for public interest reporting.",
                ],
                False,
                False,
                True,
                "pass",
            ),
            (
                [
                    "Human review is required.",
                    "The editor has editorial responsibility.",
                ],
                True,
                True,
                True,
                "not_applicable",
            ),
            (
                [
                    "This automated system notice is published.",
                ],
                False,
                False,
                True,
                "unknown",
            ),
            (
                [
                    "This automated system notice is published.",
                ],
                True,
                True,
                True,
                "unknown",
            ),
            (
                [
                    "This automated system notice is published.",
                ],
                False,
                False,
                False,
                "fail",
            ),
            (
                [
                    "This automated system notice is published.",
                ],
                True,
                True,
                False,
                "unknown",
            ),
        ],
    )
    def test_public_interest_disclosure_covers_exception_and_missing_paths(
        self,
        tmp_path: Path,
        notice_lines: list[str],
        review: bool,
        responsibility: bool,
        manifest_exists: bool,
        expected_status: str,
    ) -> None:
        notice = tmp_path / "transparency.md"
        _write_notice(notice, notice_lines)
        manifest = (
            [{"evidence_id": "file.transparency_notice", "exists": True}]
            if manifest_exists
            else []
        )
        entry = ai_act_module._evaluate_article50_public_interest_text(
            _article50_config(
                notice if manifest_exists else None,
                publishes_text=True,
                public_interest_review=review,
                public_interest_responsibility=responsibility,
            ),
            manifest,
        )
        assert str(entry["status"]) == expected_status
        if expected_status == "pass":
            assert "public-interest AI-generated text" in str(entry["rationale"])
        elif expected_status == "not_applicable":
            assert "exception" in str(entry["rationale"])
        elif review and responsibility and manifest_exists:
            assert "human review/editorial control" in str(entry["rationale"])
        elif review and responsibility:
            assert "editorial responsibility" in str(entry["rationale"])
        elif manifest_exists:
            assert "public-interest text disclosure" in str(entry["rationale"])
        else:
            assert "no disclosure evidence was provided" in str(
                entry["rationale"],
            )

    def test_ensure_coverage_complete_accepts_terminal_coverage(self) -> None:
        controls = [{"control_id": "a"}, {"control_id": "b"}]
        coverage = [
            {"control_id": "b", "status": "pass"},
            {"control_id": "a", "status": "unknown"},
        ]

        ai_act_module._ensure_coverage_complete(controls, coverage)

    @pytest.mark.parametrize(
        ("controls", "coverage", "message"),
        [
            (
                [{"control_id": "a"}],
                [],
                "coverage ledger is incomplete",
            ),
            (
                [{"control_id": "a"}],
                [{"control_id": "a", "status": "maybe"}],
                "non-terminal coverage status",
            ),
        ],
    )
    def test_ensure_coverage_complete_rejects_invalid_coverage(
        self,
        controls: list[dict[str, object]],
        coverage: list[dict[str, object]],
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            ai_act_module._ensure_coverage_complete(controls, coverage)
