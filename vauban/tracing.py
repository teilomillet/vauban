import os
import inspect
import contextvars
import time
import threading
import itertools
import json
import datetime
import re
from pathlib import Path
from contextlib import contextmanager, nullcontext
from collections import deque
from typing import Any, Callable, Optional, Dict, List, Tuple

# Global state to track if weave is initialized
_WEAVE_INITIALIZED = False
_TRACE_STDOUT_ENV = os.getenv("VAUBAN_TRACE_STDOUT")
# Default to stdout tracing unless explicitly disabled; ensures "no attack without trace" even without Weave.
_TRACE_STDOUT = False if _TRACE_STDOUT_ENV and _TRACE_STDOUT_ENV.lower() in {"0", "false", "off", "no"} else True
_TRACE_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar("vauban_trace_depth", default=0)
_TRACE_SPAN_ID: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar("vauban_trace_span_id", default=None)
_TRACE_ROOT_ID: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar("vauban_trace_root_id", default=None)
_TRACE_SESSION_ID: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar("vauban_trace_session_id", default=None)
_TRACE_SESSION_LABEL: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("vauban_trace_session_label", default=None)
_TRACE_MAX = int(os.getenv("VAUBAN_TRACE_BUFFER", "2000"))
# Buffer stores in-memory trace entries; default is generous enough for multi-wave sieges.
# Set VAUBAN_TRACE_BUFFER<=0 to disable truncation.
_TRACE_BUFFER: deque = deque(maxlen=None if _TRACE_MAX <= 0 else _TRACE_MAX)
_TRACE_DROPPED_TOTAL: int = 0  # count of evicted records across sessions
_TRACE_DROPPED_BY_SESSION: Dict[Optional[int], int] = {}
_TRACE_LOCK = threading.Lock()
_TRACE_ID_GEN = itertools.count(1)
_TRACE_SESSION_GEN = itertools.count(1)


def _is_weave_enabled() -> bool:
    """
    Weave spans only when the package is installed and a project is configured.
    Falls back to stdout tracing if either is missing.
    """
    return WEAVE_AVAILABLE and (os.getenv("WEAVE_PROJECT") or _WEAVE_INITIALIZED)


def _record_trace(entry: Dict[str, Any]) -> None:
    """Persist a lightweight trace entry in memory for quick inspection/export."""
    global _TRACE_DROPPED_TOTAL
    with _TRACE_LOCK:
        if _TRACE_BUFFER.maxlen is not None and len(_TRACE_BUFFER) >= _TRACE_BUFFER.maxlen:
            dropped_entry = _TRACE_BUFFER.popleft()
            _TRACE_DROPPED_TOTAL += 1
            sid = dropped_entry.get("session_id")
            _TRACE_DROPPED_BY_SESSION[sid] = _TRACE_DROPPED_BY_SESSION.get(sid, 0) + 1
        _TRACE_BUFFER.append(entry)


def _filter_by_session(records: List[Dict[str, Any]], session_id: Optional[int]) -> List[Dict[str, Any]]:
    if session_id is None:
        return records
    return [r for r in records if r.get("session_id") == session_id]


def get_trace_records(session_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return trace records (optionally filtered to a session)."""
    with _TRACE_LOCK:
        recs = list(_TRACE_BUFFER)
    return _filter_by_session(recs, session_id)


def get_last_trace() -> Optional[Dict[str, Any]]:
    """Return the most recent trace record, or None if empty."""
    with _TRACE_LOCK:
        return _TRACE_BUFFER[-1] if _TRACE_BUFFER else None


def get_trace_subtree(target_name: str, session_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Return the trace slice for the most recent call whose name matches target_name (case-insensitive),
    including its nested calls. Uses root_id to follow the whole request across tasks.
    """
    name_key = target_name.casefold()
    with _TRACE_LOCK:
        if not _TRACE_BUFFER:
            return []
        buf = list(_TRACE_BUFFER)
    buf = _filter_by_session(buf, session_id)
    root_id = None
    for i in range(len(buf) - 1, -1, -1):
        if str(buf[i].get("name", "")).casefold() == name_key:
            root_id = buf[i].get("root_id")
            break
    if root_id is None:
        return []
    return [rec for rec in buf if rec.get("root_id") == root_id]


def clear_trace_records() -> None:
    """Clear buffered trace records."""
    global _TRACE_DROPPED_TOTAL
    with _TRACE_LOCK:
        _TRACE_BUFFER.clear()
        _TRACE_DROPPED_TOTAL = 0
        _TRACE_DROPPED_BY_SESSION.clear()


def export_trace_records(
    directory: str = "reports", session_id: Optional[int] = None, filename: Optional[str] = None
) -> Optional[str]:
    """Persist the in-memory trace buffer to disk for offline inspection.

    Returns the destination path or ``None`` when there is no trace data for the requested session.
    """
    records = get_trace_records(session_id=session_id)
    if not records:
        return None

    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)

    label = records[-1].get("session_label") or "session"
    safe_label = re.sub(r"[^A-Za-z0-9_.:-]+", "-", str(label))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(filename) if filename else target_dir / f"trace_{safe_label}_{stamp}.json"

    def _json_safe(val: Any) -> Any:
        try:
            json.dumps(val)
            return val
        except Exception:
            return repr(val)

    sanitized = [{k: _json_safe(v) for k, v in rec.items()} for rec in records]

    with _TRACE_LOCK:
        dropped_total = _TRACE_DROPPED_TOTAL
        dropped_session = _TRACE_DROPPED_BY_SESSION.get(session_id, 0 if session_id is not None else dropped_total)
    if dropped_total:
        # Prepend a meta record so exported traces indicate truncation.
        sanitized.insert(
            0,
            {
                "type": "trace_meta",
                "dropped_total": dropped_total,
                "dropped_session": dropped_session,
                "buffer_max": _TRACE_BUFFER.maxlen,
                "note": "Oldest trace entries were evicted before export; increase VAUBAN_TRACE_BUFFER to keep full sessions.",
                "session_id": session_id,
                "session_label": records[-1].get("session_label"),
            },
        )
    path.write_text(json.dumps(sanitized, indent=2))
    return str(path)


def set_trace_session(label: Optional[str] = None) -> Tuple[contextvars.Token, contextvars.Token]:
    """
    Start a logical tracing session (e.g., one attack/siege run). Returns tokens to reset.
    """
    sid = next(_TRACE_SESSION_GEN)
    token_id = _TRACE_SESSION_ID.set(sid)
    token_label = _TRACE_SESSION_LABEL.set(label or f"session-{sid}")
    return token_id, token_label


def reset_trace_session(tokens: Tuple[contextvars.Token, contextvars.Token]) -> None:
    """Restore session context."""
    token_id, token_label = tokens
    _TRACE_SESSION_ID.reset(token_id)
    _TRACE_SESSION_LABEL.reset(token_label)


def get_last_session_id() -> Optional[int]:
    """Return the session_id of the most recent trace entry."""
    with _TRACE_LOCK:
        if not _TRACE_BUFFER:
            return None
        return _TRACE_BUFFER[-1].get("session_id")


def checkpoint(name: str, detail: str = "") -> None:
    """
    Record a synthetic trace entry (type=checkpoint) at current depth/session.
    Useful for marking phases like WaveStart/Update.
    """
    span_id = next(_TRACE_ID_GEN)
    parent_id = _TRACE_SPAN_ID.get()
    root_id = _TRACE_ROOT_ID.get() or parent_id
    with _TRACE_LOCK:
        _TRACE_BUFFER.append(
            {
                "span_id": span_id,
                "parent_id": parent_id,
                "root_id": root_id,
                "name": name,
                "args": detail,
                "ok": True,
                "result": "",
                "duration_ms": 0.0,
                "depth": _TRACE_DEPTH.get(),
                "ts": time.time(),
                "session_id": _TRACE_SESSION_ID.get(),
                "session_label": _TRACE_SESSION_LABEL.get(),
                "type": "checkpoint",
            }
        )

try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    weave = None
    WEAVE_AVAILABLE = False


def init_weave(project_name: Optional[str] = None) -> None:
    """
    Initialize weave if available and project name is provided (or in env).
    Idempotent.
    """
    global _WEAVE_INITIALIZED
    
    if not WEAVE_AVAILABLE:
        if project_name or os.getenv("WEAVE_PROJECT"):
            print("[WARNING] WEAVE_PROJECT is set but 'weave' package is not installed. Tracing disabled. Run 'pip install weave'.")
        return

    project = project_name or os.getenv("WEAVE_PROJECT")
    if project and not _WEAVE_INITIALIZED:
        try:
            weave.init(project)
            _WEAVE_INITIALIZED = True
        except Exception as e:
            # Propagate so callers can abort instead of running unlogged traces
            raise RuntimeError(f"Failed to initialize Weave project '{project}': {e}") from e


def require_weave_project() -> Optional[str]:
    """
    Ensure a project is configured before tracing; fail fast if missing.
    """
    if not WEAVE_AVAILABLE:
        return None
    project = os.getenv("WEAVE_PROJECT")
    if not project:
        # If caller opted into stdout tracing, quietly skip weave setup.
        if _TRACE_STDOUT:
            return None
        raise RuntimeError("WEAVE_PROJECT is required to log traces; set env before running.")
    init_weave(project)
    return project


@contextmanager
def weave_thread(label: Optional[str] = None):
    """
    Group downstream ops under a Weave thread when available.

    The thread id mirrors the trace session label so UI segments stay aligned.
    Falls back to a no-op when weave/project is unavailable.
    """
    if not _is_weave_enabled():
        yield
        return
    try:
        require_weave_project()
        with weave.thread(thread_id=label) as t:
            yield t
    except Exception:
        # Do not break runs if weave thread creation fails; tracing still uses stdout/buffer.
        yield


def add_attributes(attributes: Dict[str, Any]) -> None:
    """
    Safely add attributes to the current Weave span if available.
    """
    if not _is_weave_enabled():
        return
    try:
        # weave.attributes is a context manager but can also be used to set attributes on the current run/span?
        # Actually, weave.attributes({...}) returns a context manager that applies attributes to *child* spans or the current scope.
        # To add attributes to the *current* span, we might need a different API or just use the context manager around the code.
        # However, weave.current_context().add_attributes(attributes) might be what we want if it existed.
        # In Weave (wandb), weave.attributes() is often used as a context manager.
        # If we want to add attributes to the *current* span (the one we are inside), we might need to rely on the fact that
        # weave.attributes() merges into the current context.
        # Let's check the usage in engine.py: with weave.attributes({...}): ...
        # This suggests it applies to spans created *within* the block.
        # But we want to add attributes to the *current* span (e.g. the one created by @trace).
        
        # If @trace creates a span, and we call add_attributes inside the function, we want those attributes on that span.
        # If weave.attributes() is a context manager, it might not affect the *parent* span unless we are careful.
        
        # Wait, weave.op() creates a span.
        # If we are inside an op, how do we add attributes to it?
        # Usually, you pass attributes to the op() call.
        # If we want to add them dynamically, we might need to use `weave.current_run().log(attributes)` or similar?
        # But Weave is different from W&B Runs.
        
        # Let's assume for now that we will use this helper to wrap blocks of code or we will use it to inject attributes
        # into the current context which might be picked up by subsequent ops or the current one if supported.
        # Actually, looking at engine.py, they use `with weave.attributes(...)`.
        # So I will just expose a helper that does exactly that but checks for weave availability.
        pass
    except Exception:
        pass

# Re-implementing add_attributes to be a context manager for consistency with weave.attributes
@contextmanager
def attributes(attrs: Dict[str, Any]):
    """
    Context manager to add attributes to the current Weave scope.
    Safe to use even if Weave is disabled.
    """
    if not _is_weave_enabled():
        yield
        return
    try:
        with weave.attributes(attrs):
            yield
    except Exception:
        yield



def trace(func_or_name: Optional[Any] = None, *, name: Optional[str] = None) -> Callable:
    """
    Decorator to trace functions with Weave if available.
    Supports usages:
        @trace
        @trace("custom_name")
        @trace(name="custom_name")
    """
    # Resolve desired span name regardless of how caller passed it.
    resolved_name = name if name is not None else (func_or_name if isinstance(func_or_name, str) else None)

    def _wrap_stdout(func: Callable, explicit_name: Optional[str]) -> Callable:
        """
        Optional stdout tracer to make call flow readable without Weave.
        Controlled by VAUBAN_TRACE_STDOUT env (true/1/on). Uses a ContextVar to keep indentation per task.
        """
        func_name = explicit_name or getattr(func, "__name__", "call")
        is_async = inspect.iscoroutinefunction(func)

        def _summarize(val: Any, limit: int = 120) -> str:
            text = repr(val)
            return text if len(text) <= limit else text[: limit - 3] + "..."

        def _format_args(args, kwargs) -> str:
            parts = []
            for a in args[:2]:
                parts.append(_summarize(a))
            if len(args) > 2:
                parts.append("…")
            for k, v in list(kwargs.items())[:2]:
                parts.append(f"{k}={_summarize(v)}")
            if len(kwargs) > 2:
                parts.append("…")
            return ", ".join(parts)

        def _log(prefix: str, body: str) -> None:
            if not _TRACE_STDOUT:
                return
            depth = _TRACE_DEPTH.get()
            indent = "  " * depth
            print(f"{indent}{prefix} {body}")

        def _enter_span():
            parent_id = _TRACE_SPAN_ID.get()
            root_id = _TRACE_ROOT_ID.get()
            session_id = _TRACE_SESSION_ID.get()
            session_label = _TRACE_SESSION_LABEL.get()
            span_id = next(_TRACE_ID_GEN)
            if root_id is None:
                root_id = span_id
            token_depth = _TRACE_DEPTH.set(_TRACE_DEPTH.get() + 1)
            token_span = _TRACE_SPAN_ID.set(span_id)
            token_root = _TRACE_ROOT_ID.set(root_id)
            return (
                span_id,
                parent_id,
                root_id,
                session_id,
                session_label,
                token_depth,
                token_span,
                token_root,
            )

        def _exit_span(tokens):
            token_depth, token_span, token_root = tokens
            _TRACE_DEPTH.reset(token_depth)
            _TRACE_SPAN_ID.reset(token_span)
            _TRACE_ROOT_ID.reset(token_root)

        is_async_gen = inspect.isasyncgenfunction(func)

        if is_async_gen:
            async def wrapper(*args, **kwargs):
                (
                    span_id,
                    parent_id,
                    root_id,
                    session_id,
                    session_label,
                    t_depth,
                    t_span,
                    t_root,
                ) = _enter_span()
                start = time.perf_counter()
                arg_view = _format_args(args, kwargs)
                _log("▶", f"{func_name}({arg_view})")
                
                # We need to capture items to summarize, but for a generator we might just count them or show last?
                # Let's just count yielded items for the summary.
                count = 0
                error = None
                
                try:
                    async for item in func(*args, **kwargs):
                        count += 1
                        yield item
                    
                    elapsed = (time.perf_counter() - start) * 1000
                    summary = f"yielded {count} items"
                    _record_trace(
                        {
                            "span_id": span_id,
                            "parent_id": parent_id,
                            "root_id": root_id,
                            "name": func_name,
                            "args": arg_view,
                            "ok": True,
                            "result": summary,
                            "duration_ms": elapsed,
                            "depth": _TRACE_DEPTH.get(),
                            "ts": time.time(),
                            "session_id": session_id,
                            "session_label": session_label,
                        }
                    )
                    _log("◀", f"{func_name} -> {summary} [{elapsed:.1f}ms]")

                except Exception as e:
                    error = e
                    elapsed = (time.perf_counter() - start) * 1000
                    _record_trace(
                        {
                            "span_id": span_id,
                            "parent_id": parent_id,
                            "root_id": root_id,
                            "name": func_name,
                            "args": arg_view,
                            "ok": False,
                            "error": repr(e),
                            "duration_ms": elapsed,
                            "depth": _TRACE_DEPTH.get(),
                            "ts": time.time(),
                            "session_id": session_id,
                            "session_label": session_label,
                        }
                    )
                    _log("✖", f"{func_name} ! {e} [{elapsed:.1f}ms]")
                    raise
                finally:
                    _exit_span((t_depth, t_span, t_root))
            return wrapper

        if is_async:

            async def wrapper(*args, **kwargs):
                (
                    span_id,
                    parent_id,
                    root_id,
                    session_id,
                    session_label,
                    t_depth,
                    t_span,
                    t_root,
                ) = _enter_span()
                start = time.perf_counter()
                arg_view = _format_args(args, kwargs)
                _log("▶", f"{func_name}({arg_view})")
                try:
                    result = await func(*args, **kwargs)
                    elapsed = (time.perf_counter() - start) * 1000
                    summary = _summarize(result)
                    _record_trace(
                        {
                            "span_id": span_id,
                            "parent_id": parent_id,
                            "root_id": root_id,
                            "name": func_name,
                            "args": arg_view,
                            "ok": True,
                            "result": summary,
                            "duration_ms": elapsed,
                            "depth": _TRACE_DEPTH.get(),
                            "ts": time.time(),
                            "session_id": session_id,
                            "session_label": session_label,
                        }
                    )
                    _log("◀", f"{func_name} -> {summary} [{elapsed:.1f}ms]")
                    return result
                except Exception as e:
                    elapsed = (time.perf_counter() - start) * 1000
                    _record_trace(
                        {
                            "span_id": span_id,
                            "parent_id": parent_id,
                            "root_id": root_id,
                            "name": func_name,
                            "args": arg_view,
                            "ok": False,
                            "error": repr(e),
                            "duration_ms": elapsed,
                            "depth": _TRACE_DEPTH.get(),
                            "ts": time.time(),
                            "session_id": session_id,
                            "session_label": session_label,
                        }
                    )
                    _log("✖", f"{func_name} ! {e} [{elapsed:.1f}ms]")
                    raise
                finally:
                    _exit_span((t_depth, t_span, t_root))

            return wrapper

        def wrapper(*args, **kwargs):
            (
                span_id,
                parent_id,
                root_id,
                session_id,
                session_label,
                t_depth,
                t_span,
                t_root,
            ) = _enter_span()
            start = time.perf_counter()
            arg_view = _format_args(args, kwargs)
            _log("▶", f"{func_name}({arg_view})")
            try:
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                summary = _summarize(result)
                _record_trace(
                    {
                        "span_id": span_id,
                        "parent_id": parent_id,
                        "root_id": root_id,
                        "name": func_name,
                        "args": arg_view,
                        "ok": True,
                        "result": summary,
                        "duration_ms": elapsed,
                        "depth": _TRACE_DEPTH.get(),
                        "ts": time.time(),
                        "session_id": session_id,
                        "session_label": session_label,
                    }
                )
                _log("◀", f"{func_name} -> {summary} [{elapsed:.1f}ms]")
                return result
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                _record_trace(
                    {
                        "span_id": span_id,
                        "parent_id": parent_id,
                        "root_id": root_id,
                        "name": func_name,
                        "args": arg_view,
                        "ok": False,
                        "error": repr(e),
                        "duration_ms": elapsed,
                        "depth": _TRACE_DEPTH.get(),
                        "ts": time.time(),
                        "session_id": session_id,
                        "session_label": session_label,
                    }
                )
                _log("✖", f"{func_name} ! {e} [{elapsed:.1f}ms]")
                raise
            finally:
                _exit_span((t_depth, t_span, t_root))

        return wrapper

    if not _is_weave_enabled():
        # When weave is absent or not configured, still emit stdout trace so no call goes untraced.
        if callable(func_or_name) and not isinstance(func_or_name, str) and name is None:
            return _wrap_stdout(func_or_name, resolved_name)  # @trace directly on a function

        def decorator(func):
            return _wrap_stdout(func, resolved_name)

        return decorator

    # Weave is present. Build the op with the resolved name if provided.
    def _op():
        return weave.op(name=resolved_name) if resolved_name else weave.op()

    def _wrap_with_init(op_func):
        """Ensure Weave project is configured before each traced call."""
        def wrapper(*args, **kwargs):
            require_weave_project()
            return op_func(*args, **kwargs)
        return wrapper

    if callable(func_or_name) and not isinstance(func_or_name, str):
        # Usage: @trace
        return _wrap_stdout(_wrap_with_init(_op()(func_or_name)), resolved_name)

    # Usage: @trace("name") or @trace(name="name")
    def decorator(func):
        return _wrap_stdout(_wrap_with_init(_op()(func)), resolved_name)

    return decorator


# --- Weave Model Integration ---

try:
    import weave
    # If weave is available, WeaveModel is an alias for weave.Model
    # weave.Model is a Pydantic v2 BaseModel with extra powers
    class WeaveModel(weave.Model):
        model_config = {
            "arbitrary_types_allowed": True,
            "protected_namespaces": (),  # Allow fields starting with "model_"
        }
except ImportError:
    from pydantic import BaseModel
    # Fallback if weave is not installed
    class WeaveModel(BaseModel):  # type: ignore
        model_config = {
            "arbitrary_types_allowed": True,
            "protected_namespaces": (),
        }
