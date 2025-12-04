__version__ = "0.1.2"

from vauban.api import scout, assess, attack, siege, baseline, baseline_async, visualize
from vauban.interfaces import Target, SiegeResult
from vauban.target import ModelTarget, OpenAITarget
from vauban.history import load_latest_history, list_breaches, get_attack_by_id
from vauban.tracing import (
    get_trace_records,
    get_last_trace,
    get_trace_subtree,
    get_last_session_id,
    set_trace_session,
    reset_trace_session,
    checkpoint,
    clear_trace_records,
    export_trace_records,
)

__all__ = [
    "scout",
    "assess",
    "attack",
    "siege",
    "baseline",
    "baseline_async",
    "visualize",
    "Target",
    "ModelTarget",
    "OpenAITarget",
    "SiegeResult",
    "load_latest_history",
    "list_breaches",
    "get_attack_by_id",
    "get_trace_records",
    "get_last_trace",
    "get_trace_subtree",
    "get_last_session_id",
    "set_trace_session",
    "reset_trace_session",
    "checkpoint",
    "clear_trace_records",
    "export_trace_records",
]
