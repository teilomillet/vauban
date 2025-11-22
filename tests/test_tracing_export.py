import json
from pathlib import Path
from collections import deque

from vauban.tracing import (
    trace,
    set_trace_session,
    reset_trace_session,
    clear_trace_records,
    get_last_session_id,
    export_trace_records,
)


def test_export_trace_records_writes_session_buffer(tmp_path):
    clear_trace_records()
    tokens = set_trace_session(label="demo-session")

    @trace
    def add(x, y):
        return x + y

    add(1, 2)

    session_id = get_last_session_id()
    path_str = export_trace_records(directory=tmp_path, session_id=session_id)

    assert path_str is not None
    path = Path(path_str)
    data = json.loads(path.read_text())
    assert data, "export should write at least one trace record"
    assert data[-1]["session_label"] == "demo-session"

    reset_trace_session(tokens)
    clear_trace_records()


def test_export_trace_records_marks_truncation(tmp_path):
    clear_trace_records()
    tokens = set_trace_session(label="truncation-session")

    # Shrink buffer to force eviction and record drop counters.
    from vauban import tracing

    original_buffer = tracing._TRACE_BUFFER
    original_dropped_total = tracing._TRACE_DROPPED_TOTAL
    original_dropped_by_session = tracing._TRACE_DROPPED_BY_SESSION

    tracing._TRACE_BUFFER = deque(maxlen=2)
    tracing._TRACE_DROPPED_TOTAL = 0
    tracing._TRACE_DROPPED_BY_SESSION = {}

    @trace
    def bump(x):
        return x + 1

    bump(1)
    bump(2)
    bump(3)  # Evicts the first record due to tiny buffer

    session_id = get_last_session_id()
    path_str = export_trace_records(directory=tmp_path, session_id=session_id)
    data = json.loads(Path(path_str).read_text())

    assert data[0]["type"] == "trace_meta"
    assert data[0]["dropped_total"] >= 1
    assert data[0]["dropped_session"] >= 1

    # Restore globals to avoid side effects across tests
    tracing._TRACE_BUFFER = original_buffer
    tracing._TRACE_DROPPED_TOTAL = original_dropped_total
    tracing._TRACE_DROPPED_BY_SESSION = original_dropped_by_session

    reset_trace_session(tokens)
    clear_trace_records()
