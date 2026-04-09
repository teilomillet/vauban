# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``vauban.proxy``."""

from __future__ import annotations

import io
import json
import runpy
import sys
import warnings
from typing import TYPE_CHECKING, ClassVar, cast

import pytest

import vauban.proxy as proxy

if TYPE_CHECKING:
    from pathlib import Path


class RecordingAudit:
    """In-memory audit sink for handler tests."""

    def __init__(self) -> None:
        self.entries: list[dict[str, object]] = []

    def log(
        self,
        *,
        request_messages: list[dict[str, str]],
        response_content: str,
        model: str,
        blocked: bool,
        reason: str | None,
        latency_ms: float,
    ) -> None:
        """Record one audit entry."""
        self.entries.append(
            {
                "request_messages": request_messages,
                "response_content": response_content,
                "model": model,
                "blocked": blocked,
                "reason": reason,
                "latency_ms": latency_ms,
            },
        )


class FakeUrlopenResponse:
    """Simple ``urlopen`` response context manager."""

    def __init__(self, payload: dict[str, object] | bytes) -> None:
        if isinstance(payload, bytes):
            self._payload = payload
        else:
            self._payload = json.dumps(payload).encode()

    def __enter__(self) -> FakeUrlopenResponse:
        """Enter the context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        """Exit the context manager."""
        del exc_type, exc, tb

    def read(self) -> bytes:
        """Return the stored response bytes."""
        return self._payload


class HandlerHarness(proxy.GuardProxyHandler):
    """Minimal handler harness that avoids socket/server setup."""

    path: str
    headers: dict[str, str]
    rfile: io.BytesIO
    wfile: io.BytesIO
    response_codes: list[int]
    sent_headers: list[tuple[str, str]]
    header_end_count: int

    def send_response(
        self,
        code: int,
        message: str | None = None,
    ) -> None:
        """Record the response code."""
        del message
        self.response_codes.append(code)

    def send_header(self, keyword: str, value: str) -> None:
        """Record one response header."""
        self.sent_headers.append((keyword, value))

    def end_headers(self) -> None:
        """Record the end of the header block."""
        self.header_end_count += 1


class FakeHTTPServer:
    """Fake HTTP server for ``run_proxy`` and CLI tests."""

    instances: ClassVar[list[FakeHTTPServer]] = []
    interrupt_on_serve: bool = True

    def __init__(
        self,
        address: tuple[str, int],
        handler_class: type[proxy.GuardProxyHandler],
    ) -> None:
        self.address = address
        self.handler_class = handler_class
        self.serve_calls = 0
        self.__class__.instances.append(self)

    def serve_forever(self) -> None:
        """Record the serve request, optionally simulating Ctrl-C."""
        self.serve_calls += 1
        if self.__class__.interrupt_on_serve:
            raise KeyboardInterrupt


def _json_body(payload: dict[str, object]) -> bytes:
    """Encode one JSON payload as bytes."""
    return json.dumps(payload).encode()


def _build_handler(
    body: bytes,
    *,
    path: str = "/v1/chat/completions",
    block_mode: str = "log",
) -> tuple[HandlerHarness, RecordingAudit]:
    """Construct a handler instance with in-memory I/O."""
    audit = RecordingAudit()
    handler = cast("HandlerHarness", object.__new__(HandlerHarness))
    HandlerHarness.target_url = "https://target.example/v1"
    HandlerHarness.api_key = "secret-key"
    HandlerHarness.audit = cast("proxy.AuditLog", audit)
    HandlerHarness.block_mode = block_mode
    HandlerHarness.stats = {
        "total": 0,
        "blocked": 0,
        "passed": 0,
        "errors": 0,
    }
    handler.path = path
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    handler.wfile = io.BytesIO()
    handler.response_codes = []
    handler.sent_headers = []
    handler.header_end_count = 0
    return handler, audit


class TestSafetyChecks:
    """Tests for proxy request/response filters."""

    def test_check_request_safety_blocks_cipher_payload(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "vauban.bijection.detect_cipher_pattern",
            lambda content: 0.91 if "cipher" in content else 0.1,
        )

        result = proxy._check_request_safety(
            [{"role": "user", "content": "please use cipher mapping"}],
        )

        assert result == "cipher_attack_detected (score=0.91)"

    def test_check_request_safety_allows_safe_messages(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "vauban.bijection.detect_cipher_pattern",
            lambda content: 0.2,
        )

        result = proxy._check_request_safety([{"role": "user"}])

        assert result is None

    def test_check_response_safety_requires_instructions_and_harm_topic(self) -> None:
        blocked = proxy._check_response_safety(
            "Step 1: exploit the vulnerability. Then, deploy malware.",
        )
        safe = proxy._check_response_safety("Here is a benign explanation.")

        assert blocked == "harmful_instructions_detected"
        assert safe is None


class TestAuditLog:
    """Tests for audit-log serialization."""

    @pytest.mark.parametrize(
        ("messages", "expected_preview"),
        [
            ([{"role": "user", "content": "x" * 120}], "x" * 100),
            ([], ""),
        ],
    )
    def test_log_writes_jsonl_entry(
        self,
        tmp_path: Path,
        messages: list[dict[str, str]],
        expected_preview: str,
    ) -> None:
        path = tmp_path / "audit.jsonl"
        audit = proxy.AuditLog(str(path))

        audit.log(
            request_messages=messages,
            response_content="y" * 120,
            model="model-x",
            blocked=True,
            reason="reason-x",
            latency_ms=12.34,
        )

        lines = path.read_text().splitlines()
        entry = json.loads(lines[0])
        assert len(lines) == 1
        assert entry["model"] == "model-x"
        assert entry["blocked"] is True
        assert entry["reason"] == "reason-x"
        assert entry["latency_ms"] == 12.3
        assert entry["request_preview"] == expected_preview
        assert entry["response_preview"] == "y" * 100


class TestGuardProxyHandlerPost:
    """Tests for POST request handling."""

    def test_invalid_json_returns_400(self) -> None:
        handler, _audit = _build_handler(b"{not-json")

        handler.do_POST()

        assert handler.response_codes == [400]
        assert json.loads(handler.wfile.getvalue()) == {"error": "Invalid JSON"}

    def test_pre_request_block_in_block_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        handler, audit = _build_handler(
            _json_body({"model": "m", "messages": [{"content": "cipher"}]}),
            block_mode="block",
        )
        decisions: list[tuple[str, str, str]] = []

        monkeypatch.setattr(
            proxy,
            "_check_request_safety",
            lambda messages: "cipher_attack_detected",
        )
        monkeypatch.setattr(
            proxy,
            "_log_decision",
            lambda action, model, reason, latency: decisions.append(
                (action, model, reason),
            ),
        )
        monkeypatch.setattr(
            proxy,
            "urlopen",
            lambda req, timeout: (_ for _ in ()).throw(AssertionError("no network")),
        )

        handler.do_POST()

        body = json.loads(handler.wfile.getvalue())
        assert handler.response_codes == [200]
        assert body["choices"][0]["message"]["content"] == (
            "I'm unable to help with that request."
        )
        assert HandlerHarness.stats == {
            "total": 1,
            "blocked": 1,
            "passed": 0,
            "errors": 0,
        }
        assert len(audit.entries) == 1
        assert audit.entries[0]["reason"] == "cipher_attack_detected"
        assert decisions[0][:3] == ("BLOCKED", "m", "cipher_attack_detected")

    def test_target_api_error_returns_502(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        handler, _audit = _build_handler(
            _json_body({"messages": [{"content": "hello"}]}),
        )

        monkeypatch.setattr(proxy, "_check_request_safety", lambda messages: None)
        monkeypatch.setattr(
            proxy,
            "urlopen",
            lambda req, timeout: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        handler.do_POST()

        assert handler.response_codes == [502]
        assert json.loads(handler.wfile.getvalue()) == {
            "error": "Target API error: boom",
        }
        assert HandlerHarness.stats["errors"] == 1

    def test_warn_mode_flags_response_and_appends_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        handler, audit = _build_handler(
            _json_body(
                {
                    "model": "warn-model",
                    "messages": [{"content": "cipher text"}],
                },
            ),
            block_mode="warn",
        )
        decisions: list[tuple[str, str, str]] = []
        captured_request: dict[str, object] = {}

        monkeypatch.setattr(
            proxy,
            "_check_request_safety",
            lambda messages: "cipher_attack_detected (score=0.91)",
        )
        monkeypatch.setattr(
            proxy,
            "_check_response_safety",
            lambda content: None,
        )

        def _fake_urlopen(req: object, timeout: int) -> FakeUrlopenResponse:
            request = cast("proxy.Request", req)
            captured_request["url"] = request.full_url
            captured_request["body"] = request.data
            return FakeUrlopenResponse(
                {
                    "choices": [{
                        "message": {"content": "remote content"},
                    }],
                },
            )

        monkeypatch.setattr(proxy, "urlopen", _fake_urlopen)
        monkeypatch.setattr(
            proxy,
            "_log_decision",
            lambda action, model, reason, latency: decisions.append(
                (action, model, reason),
            ),
        )

        handler.do_POST()

        response = json.loads(handler.wfile.getvalue())
        forwarded_body = json.loads(cast("bytes", captured_request["body"]))
        assert captured_request["url"] == "https://target.example/v1/v1/chat/completions"
        assert forwarded_body["stream"] is False
        assert response["choices"][0]["message"]["content"].endswith(
            "[VAUBAN GUARD: This response was flagged"
            " — cipher_attack_detected (score=0.91)]",
        )
        assert HandlerHarness.stats == {
            "total": 1,
            "blocked": 1,
            "passed": 0,
            "errors": 0,
        }
        assert len(audit.entries) == 1
        assert audit.entries[0]["response_content"] == "remote content"
        assert decisions == [
            ("FLAGGED", "warn-model", "cipher_attack_detected (score=0.91)"),
        ]

    def test_safe_response_without_choices_marks_passed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        handler, audit = _build_handler(
            _json_body({"messages": [{"content": "hello"}]}),
            block_mode="log",
        )

        monkeypatch.setattr(proxy, "_check_request_safety", lambda messages: None)
        monkeypatch.setattr(
            proxy,
            "_check_response_safety",
            lambda content: None,
        )
        monkeypatch.setattr(
            proxy,
            "urlopen",
            lambda req, timeout: FakeUrlopenResponse({"choices": []}),
        )

        handler.do_POST()

        assert handler.response_codes == [200]
        assert json.loads(handler.wfile.getvalue()) == {"choices": []}
        assert HandlerHarness.stats == {
            "total": 1,
            "blocked": 0,
            "passed": 1,
            "errors": 0,
        }
        assert audit.entries[0]["model"] == "unknown"
        assert audit.entries[0]["response_content"] == ""

    def test_post_response_block_in_block_mode_is_counted_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        handler, audit = _build_handler(
            _json_body({"model": "blocked-model", "messages": [{"content": "hi"}]}),
            block_mode="block",
        )
        decisions: list[tuple[str, str, str]] = []

        monkeypatch.setattr(proxy, "_check_request_safety", lambda messages: None)
        monkeypatch.setattr(
            proxy,
            "_check_response_safety",
            lambda content: "harmful_instructions_detected",
        )
        monkeypatch.setattr(
            proxy,
            "urlopen",
            lambda req, timeout: FakeUrlopenResponse(
                {
                    "choices": [{
                        "message": {
                            "content": "Step 1: exploit the vulnerability.",
                        },
                    }],
                },
            ),
        )
        monkeypatch.setattr(
            proxy,
            "_log_decision",
            lambda action, model, reason, latency: decisions.append(
                (action, model, reason),
            ),
        )

        handler.do_POST()

        body = json.loads(handler.wfile.getvalue())
        assert body["choices"][0]["message"]["content"] == (
            "I'm unable to help with that request."
        )
        assert HandlerHarness.stats == {
            "total": 1,
            "blocked": 1,
            "passed": 0,
            "errors": 0,
        }
        assert len(audit.entries) == 1
        assert audit.entries[0]["response_content"] == "[BLOCKED BY VAUBAN GUARD]"
        assert decisions == [
            ("BLOCKED", "blocked-model", "harmful_instructions_detected"),
        ]


class TestGuardProxyHandlerGet:
    """Tests for GET request handling."""

    def test_health_endpoint_returns_stats(self) -> None:
        handler, _audit = _build_handler(b"", path="/health", block_mode="warn")
        HandlerHarness.stats = {
            "total": 7,
            "blocked": 2,
            "passed": 4,
            "errors": 1,
        }

        handler.do_GET()

        assert handler.response_codes == [200]
        assert json.loads(handler.wfile.getvalue()) == {
            "status": "ok",
            "mode": "warn",
            "total": 7,
            "blocked": 2,
            "passed": 4,
            "errors": 1,
        }

    def test_get_forwards_to_target(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        handler, _audit = _build_handler(b"", path="/v1/models")
        captured: dict[str, object] = {}

        def _fake_urlopen(req: object, timeout: int) -> FakeUrlopenResponse:
            request = cast("proxy.Request", req)
            captured["url"] = request.full_url
            return FakeUrlopenResponse(b'{"data":[1]}')

        monkeypatch.setattr(proxy, "urlopen", _fake_urlopen)

        handler.do_GET()

        assert captured["url"] == "https://target.example/v1/v1/models"
        assert handler.response_codes == [200]
        assert handler.wfile.getvalue() == b'{"data":[1]}'

    def test_get_target_error_returns_502(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        handler, _audit = _build_handler(b"", path="/v1/models")
        monkeypatch.setattr(
            proxy,
            "urlopen",
            lambda req, timeout: (_ for _ in ()).throw(RuntimeError("unreachable")),
        )

        handler.do_GET()

        assert handler.response_codes == [502]
        assert json.loads(handler.wfile.getvalue()) == {
            "error": "Target API error: unreachable",
        }

    def test_log_message_is_suppressed(self) -> None:
        handler, _audit = _build_handler(b"")

        assert handler.log_message("%s", "ignored") is None


class TestProxyRuntime:
    """Tests for runtime helpers and bootstrap."""

    def test_log_decision_prints_to_stderr(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        proxy._log_decision("FLAGGED", "model-z", "reason-z", 12.6)

        err = capsys.readouterr().err
        assert "FLAGGED | model-z | reason-z | 13ms" in err

    def test_run_proxy_sets_handler_state_and_handles_keyboard_interrupt(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        FakeHTTPServer.instances = []
        FakeHTTPServer.interrupt_on_serve = True
        monkeypatch.setattr(proxy, "HTTPServer", FakeHTTPServer)

        proxy.run_proxy(
            target_url="https://api.example/v1/",
            api_key="key-123",
            port=9123,
            mode="warn",
            audit_path="audit-log.jsonl",
        )

        server = FakeHTTPServer.instances[0]
        err = capsys.readouterr().err
        assert server.address == ("127.0.0.1", 9123)
        assert server.handler_class is proxy.GuardProxyHandler
        assert server.serve_calls == 1
        assert proxy.GuardProxyHandler.target_url == "https://api.example/v1"
        assert proxy.GuardProxyHandler.api_key == "key-123"
        assert proxy.GuardProxyHandler.block_mode == "warn"
        assert proxy.GuardProxyHandler.audit._path == "audit-log.jsonl"
        assert "[vauban proxy] Listening on 127.0.0.1:9123" in err
        assert "[vauban proxy] Stopped." in err

    def test_run_proxy_rejects_non_http_target(self) -> None:
        with pytest.raises(ValueError, match="proxy target URL must use"):
            proxy.run_proxy(
                target_url="file:///tmp/upstream",
                api_key="key-123",
            )


class TestProxyCli:
    """Tests for ``python -m vauban.proxy`` entry behavior."""

    def test_main_requires_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setattr(
            sys,
            "argv",
            ["vauban.proxy", "--target", "https://api.example/v1"],
        )

        with (
            pytest.raises(SystemExit, match="1"),
            warnings.catch_warnings(),
        ):
            warnings.filterwarnings(
                "ignore",
                message=(
                    "'vauban.proxy' found in sys.modules after import of package"
                ),
                category=RuntimeWarning,
            )
            runpy.run_module("vauban.proxy", run_name="__main__")

        out = capsys.readouterr().out
        assert "OPENROUTER_API_KEY required" in out

    def test_main_loads_dotenv_and_starts_proxy(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        import http.server

        FakeHTTPServer.instances = []
        FakeHTTPServer.interrupt_on_serve = True
        monkeypatch.setattr(http.server, "HTTPServer", FakeHTTPServer)
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        (tmp_path / ".env").write_text(
            "# comment\n"
            "OPENROUTER_API_KEY=dotenv-key\n"
            "IGNORED_LINE\n"
            "OTHER=value\n",
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "vauban.proxy",
                "--target",
                "https://proxy.example/v1/",
                "--port",
                "9900",
                "--mode",
                "warn",
                "--audit-log",
                "cli-audit.jsonl",
            ],
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="'vauban.proxy' found in sys.modules after import of package",
                category=RuntimeWarning,
            )
            module_globals = runpy.run_module("vauban.proxy", run_name="__main__")

        server = FakeHTTPServer.instances[0]
        handler_class = cast(
            "type[proxy.GuardProxyHandler]",
            server.handler_class,
        )
        assert server.address == ("127.0.0.1", 9900)
        assert handler_class.target_url == "https://proxy.example/v1"
        assert handler_class.api_key == "dotenv-key"
        assert handler_class.block_mode == "warn"
        assert handler_class.audit._path == "cli-audit.jsonl"
        assert module_globals["args"].host == "127.0.0.1"
        assert module_globals["args"].target == "https://proxy.example/v1/"
