# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Vauban Guard Proxy — OpenAI-compatible middleware that monitors and
filters LLM responses in real-time.

Sits between client and any OpenAI-compatible API endpoint.  Forwards
requests, inspects responses, and blocks harmful content before it
reaches the client.

Usage:

    # Terminal: start the proxy
    uv run python -m vauban.proxy \
        --target https://api.openai.com/v1 \
        --api-key $OPENAI_API_KEY \
        --port 8000

    # Client: point your app at the proxy instead of OpenAI
    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"gpt-4","messages":[{"role":"user","content":"hello"}]}'

The proxy logs every request/response and flags harmful content.
"""

from __future__ import annotations

import datetime
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar
from urllib.request import Request, urlopen

from vauban._network import validate_http_url

# ---------------------------------------------------------------------------
# Safety checks (text-level — works with any remote API)
# ---------------------------------------------------------------------------

_HARMFUL_INDICATORS: list[str] = [
    "how to hack",
    "how to make a bomb",
    "how to pick a lock",
    "phishing email",
    "steal credentials",
    "bypass security",
    "make a weapon",
    "create malware",
    "social engineering attack",
    "exploit vulnerability",
]


def _check_request_safety(messages: list[dict[str, str]]) -> str | None:
    """Check if the request contains known attack patterns.

    Returns a reason string if blocked, None if safe.
    """
    from vauban.bijection import detect_cipher_pattern

    for msg in messages:
        content = msg.get("content", "")

        # Check for cipher/encoding attack patterns
        cipher_score = detect_cipher_pattern(content)
        if cipher_score > 0.6:
            return f"cipher_attack_detected (score={cipher_score:.2f})"

    return None


def _check_response_safety(content: str) -> str | None:
    """Check if the response contains harmful content.

    Returns a reason string if blocked, None if safe.
    """
    lower = content.lower()

    # Check for detailed harmful instructions
    instruction_markers = [
        "step 1", "step 2", "first,", "then,", "next,",
        "here's how", "follow these steps", "instructions:",
    ]
    harmful_topic_markers = [
        "exploit", "vulnerability", "payload", "inject",
        "bypass", "credential", "phishing", "malware",
        "ransomware", "trojan", "backdoor",
    ]

    has_instructions = any(m in lower for m in instruction_markers)
    has_harmful_topic = any(m in lower for m in harmful_topic_markers)

    if has_instructions and has_harmful_topic:
        return "harmful_instructions_detected"

    return None


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class AuditLog:
    """Append-only audit log for proxy decisions."""

    def __init__(self, path: str = "proxy_audit.jsonl") -> None:
        self._path = path

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
        """Append an entry to the audit log."""
        entry = {
            "timestamp": datetime.datetime.now(
                tz=datetime.UTC,
            ).isoformat(timespec="seconds"),
            "model": model,
            "blocked": blocked,
            "reason": reason,
            "latency_ms": round(latency_ms, 1),
            "request_preview": request_messages[-1].get(
                "content", "",
            )[:100] if request_messages else "",
            "response_preview": response_content[:100],
        }
        with open(self._path, "a") as f:
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Proxy server
# ---------------------------------------------------------------------------


class GuardProxyHandler(BaseHTTPRequestHandler):
    """HTTP request handler that proxies to a target API with safety checks."""

    target_url: str = ""
    api_key: str = ""
    audit: AuditLog = AuditLog()
    block_mode: str = "log"  # "log", "block", "warn"
    stats: ClassVar[dict[str, int]] = {
        "total": 0, "blocked": 0, "passed": 0, "errors": 0,
    }

    def do_POST(self) -> None:
        """Handle POST requests (chat completions)."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            request_data = json.loads(body)
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON")
            return

        messages = request_data.get("messages", [])
        model = request_data.get("model", "unknown")
        self.__class__.stats["total"] += 1

        t0 = time.monotonic()

        # -- Pre-request safety check --
        req_block_reason = _check_request_safety(messages)
        if req_block_reason and self.__class__.block_mode == "block":
            self._send_blocked(req_block_reason, model, messages, t0)
            return

        # -- Forward to target --
        try:
            target = f"{self.__class__.target_url}{self.path}"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.__class__.api_key}",
            }
            # Force non-streaming for inspection
            request_data["stream"] = False
            req = Request(
                target,
                data=json.dumps(request_data).encode(),
                headers=headers,
            )
            with urlopen(req, timeout=60) as resp:
                response_data = json.loads(resp.read())
        except Exception as exc:
            self.__class__.stats["errors"] += 1
            self._send_error(502, f"Target API error: {exc}")
            return

        # -- Post-response safety check --
        response_content = ""
        choices = response_data.get("choices", [])
        if choices:
            response_content = (
                choices[0].get("message", {}).get("content", "")
                or ""
            )

        resp_block_reason = _check_response_safety(response_content)
        block_reason = req_block_reason or resp_block_reason
        blocked = block_reason is not None

        latency = (time.monotonic() - t0) * 1000

        if blocked and self.__class__.block_mode == "block":
            self._send_blocked(
                block_reason or "blocked", model, messages, t0,
            )
            return

        if blocked:
            self.__class__.stats["blocked"] += 1
            _log_decision(
                "FLAGGED",
                model, block_reason or "", latency,
            )
        else:
            self.__class__.stats["passed"] += 1

        # -- Audit log --
        self.__class__.audit.log(
            request_messages=messages,
            response_content=response_content,
            model=model,
            blocked=blocked,
            reason=block_reason,
            latency_ms=latency,
        )

        # -- Respond to client --
        if blocked and self.__class__.block_mode == "warn":
            # Inject warning into response
            warning = (
                "\n\n[VAUBAN GUARD: This response was flagged"
                f" — {block_reason}]"
            )
            if choices:
                choices[0]["message"]["content"] = (
                    response_content + warning
                )

        response_bytes = json.dumps(response_data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_GET(self) -> None:
        """Handle GET requests (models list, health check)."""
        if self.path == "/health":
            stats = self.__class__.stats
            body = json.dumps({
                "status": "ok",
                "mode": self.__class__.block_mode,
                **stats,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return

        # Forward GET to target
        try:
            target = f"{self.__class__.target_url}{self.path}"
            headers = {
                "Authorization": f"Bearer {self.__class__.api_key}",
            }
            req = Request(target, headers=headers)
            with urlopen(req, timeout=30) as resp:
                data = resp.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)
        except Exception as exc:
            self._send_error(502, f"Target API error: {exc}")

    def _send_blocked(
        self,
        reason: str,
        model: str,
        messages: list[dict[str, str]],
        t0: float,
    ) -> None:
        self.__class__.stats["blocked"] += 1
        latency = (time.monotonic() - t0) * 1000
        _log_decision("BLOCKED", model, reason, latency)

        self.__class__.audit.log(
            request_messages=messages,
            response_content="[BLOCKED BY VAUBAN GUARD]",
            model=model,
            blocked=True,
            reason=reason,
            latency_ms=latency,
        )

        blocked_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": (
                        "I'm unable to help with that request."
                    ),
                },
                "finish_reason": "stop",
            }],
            "model": model,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
        body = json.dumps(blocked_response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code: int, message: str) -> None:
        body = json.dumps({"error": message}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Suppress default access logs."""


def _log_decision(
    action: str, model: str, reason: str, latency: float,
) -> None:
    ts = datetime.datetime.now(tz=datetime.UTC).strftime("%H:%M:%S")
    print(
        f"[{ts}] {action} | {model} | {reason} | {latency:.0f}ms",
        file=sys.stderr,
        flush=True,
    )


def run_proxy(
    *,
    target_url: str,
    api_key: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    mode: str = "log",
    audit_path: str = "proxy_audit.jsonl",
) -> None:
    """Start the guard proxy server.

    Args:
        target_url: Target API base URL (e.g. https://api.openai.com/v1).
        api_key: API key for the target.
        host: Local interface to bind.
        port: Local port to listen on.
        mode: "log" (monitor only), "block" (reject harmful),
            or "warn" (append warning to response).
        audit_path: Path for the JSONL audit log.
    """
    # Strip trailing slash
    target_url = validate_http_url(
        target_url.rstrip("/"),
        context="proxy target URL",
    )

    GuardProxyHandler.target_url = target_url
    GuardProxyHandler.api_key = api_key
    GuardProxyHandler.block_mode = mode
    GuardProxyHandler.audit = AuditLog(audit_path)
    GuardProxyHandler.stats = {
        "total": 0, "blocked": 0, "passed": 0, "errors": 0,
    }

    server = HTTPServer((host, port), GuardProxyHandler)
    print(
        f"[vauban proxy] Listening on {host}:{port}"
        f" → {target_url} (mode={mode})",
        file=sys.stderr,
    )
    print(
        f"[vauban proxy] Point your client at"
        f" http://localhost:{port}/v1/chat/completions",
        file=sys.stderr,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(
            f"\n[vauban proxy] Stopped."
            f" Stats: {GuardProxyHandler.stats}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Vauban Guard Proxy",
    )
    parser.add_argument(
        "--target", required=True,
        help="Target API base URL",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY", ""),
        help="API key for target (default: $OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
    )
    parser.add_argument(
        "--mode", choices=["log", "block", "warn"],
        default="block",
    )
    parser.add_argument(
        "--audit-log", default="proxy_audit.jsonl",
    )

    args = parser.parse_args()

    # Load .env if present
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    if not args.api_key:
        args.api_key = os.environ.get("OPENROUTER_API_KEY", "")

    if not args.api_key:
        print("Error: --api-key or OPENROUTER_API_KEY required")
        sys.exit(1)

    run_proxy(
        target_url=args.target,
        api_key=args.api_key,
        host=args.host,
        port=args.port,
        mode=args.mode,
        audit_path=args.audit_log,
    )
