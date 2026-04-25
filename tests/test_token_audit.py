# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for token_audit parsing, aggregation, and mode execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from tests.conftest import make_early_mode_context
from vauban._pipeline._mode_token_audit import _run_token_audit_mode
from vauban.config._parse_token_audit import _parse_token_audit
from vauban.token_audit import run_token_audit
from vauban.types import TokenAuditConfig

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(slots=True)
class _FakeWeight:
    shape: tuple[int, int]


@dataclass(slots=True)
class _FakeEmbedding:
    weight: _FakeWeight


@dataclass(slots=True)
class _FakeTransformer:
    embed_tokens: _FakeEmbedding


@dataclass(slots=True)
class _FakeModel:
    model: _FakeTransformer


class _AuditTokenizer:
    """Tokenizer stub exposing varied decoded surfaces for audit coverage."""

    def __init__(self) -> None:
        self.chat_template = "{{ messages }}"
        self.all_special_ids = [0, 1, 2]
        self._decode_map: dict[int, str] = {
            0: "",
            1: "\u200b",
            2: "🙂",
            3: "你好",
            4: "<|assistant|>",
            5: " lead",
            6: "trail ",
            7: "\ufffd",
            8: "dup",
            9: "dup",
        }

    def encode(self, text: str) -> list[int]:
        del text
        return [0]

    def decode(self, token_ids: list[int]) -> str:
        token_id = token_ids[0]
        if token_id == 10:
            msg = "decode failed"
            raise ValueError(msg)
        return self._decode_map[token_id]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
    ) -> str | list[int]:
        del messages
        return [0] if tokenize else self.chat_template


def _fake_model(vocab_size: int) -> _FakeModel:
    return _FakeModel(
        model=_FakeTransformer(
            embed_tokens=_FakeEmbedding(weight=_FakeWeight((vocab_size, 4))),
        ),
    )


class TestParseTokenAudit:
    """Parser coverage for the [token_audit] section."""

    def test_parse_reads_values(self) -> None:
        cfg = _parse_token_audit(
            {
                "token_audit": {
                    "max_token_id": 255,
                    "include_duplicate_surface_scan": False,
                },
            },
        )
        assert cfg == TokenAuditConfig(
            max_token_id=255,
            include_duplicate_surface_scan=False,
        )

    def test_parse_rejects_negative_max_token_id(self) -> None:
        with pytest.raises(ValueError, match="max_token_id must be >= 0"):
            _parse_token_audit({"token_audit": {"max_token_id": -1}})


class TestRunTokenAudit:
    """Unit tests for aggregate tokenizer-surface auditing."""

    def test_run_token_audit_redacts_and_counts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "vauban.token_audit.get_transformer",
            lambda model: model.model,
        )
        result = run_token_audit(
            _fake_model(11),
            _AuditTokenizer(),
            TokenAuditConfig(),
            model_path="mlx-community/gemma-4-e2b-it-bf16",
        )

        assert result.vocab_size == 11
        assert result.scanned_token_count == 11
        assert result.declared_special_token_count == 3
        assert result.chat_template_declared is True
        assert result.category_counts["empty_decode"] == 1
        assert result.category_counts["contains_non_printing"] == 1
        assert result.category_counts["invisible_only"] == 1
        assert result.category_counts["emoji_only"] == 1
        assert result.category_counts["non_latin_only"] == 1
        assert result.category_counts["template_like"] == 1
        assert result.category_counts["leading_space"] == 1
        assert result.category_counts["trailing_space"] == 1
        assert result.category_counts["replacement_char"] == 1
        assert result.category_counts["decode_error"] == 1
        assert result.duplicate_surface_count == 1
        assert result.duplicate_token_count == 2
        assert any("redacted" in note.lower() for note in result.notes)
        assert all("dup" not in finding for finding in result.findings)

    def test_run_token_audit_respects_scan_limit_and_duplicate_toggle(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "vauban.token_audit.get_transformer",
            lambda model: model.model,
        )
        result = run_token_audit(
            _fake_model(11),
            _AuditTokenizer(),
            TokenAuditConfig(
                max_token_id=4,
                include_duplicate_surface_scan=False,
            ),
            model_path="mlx-community/gemma-4-e2b-it-bf16",
        )

        assert result.scanned_token_count == 5
        assert result.duplicate_surface_count is None
        assert result.duplicate_token_count is None
        assert any("first 5 token ids" in note.lower() for note in result.notes)


class TestTokenAuditMode:
    """End-to-end mode coverage for standalone token_audit execution."""

    def test_run_token_audit_mode_writes_report(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "vauban.token_audit.get_transformer",
            lambda model: model.model,
        )
        monkeypatch.setattr(
            "vauban._model_io.load_model",
            lambda model_path: (_fake_model(11), _AuditTokenizer()),
        )
        ctx = make_early_mode_context(
            tmp_path,
            model_path="mlx-community/gemma-4-e2b-it-bf16",
            token_audit=TokenAuditConfig(),
        )

        _run_token_audit_mode(ctx)

        report_path = tmp_path / "token_audit_report.json"
        assert report_path.exists()
        payload = json.loads(report_path.read_text())
        assert payload["model_path"] == "mlx-community/gemma-4-e2b-it-bf16"
        assert payload["vocab_size"] == 11
        assert payload["duplicate_surface_count"] == 1
        assert payload["chat_template_declared"] is True
        assert (tmp_path / "experiment_log.jsonl").exists()

    def test_run_token_audit_mode_requires_model_path(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(
            tmp_path,
            model_path="",
            token_audit=TokenAuditConfig(),
        )
        with pytest.raises(
            ValueError,
            match=r"\[token_audit\] mode requires \[model\]\.path",
        ):
            _run_token_audit_mode(ctx)
