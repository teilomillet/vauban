import asyncio
import pytest

from vauban import tracing


def test_trace_records_sync(monkeypatch):
    tracing.clear_trace_records()
    monkeypatch.setattr(tracing, "_TRACE_STDOUT", False)

    @tracing.trace
    def add(a, b):
        return a + b

    result = add(1, 2)
    last = tracing.get_last_trace()

    assert result == 3
    assert last["name"] == "add"
    assert last["ok"] is True
    assert "duration_ms" in last


def test_trace_records_exceptions(monkeypatch):
    tracing.clear_trace_records()
    monkeypatch.setattr(tracing, "_TRACE_STDOUT", False)

    @tracing.trace
    def boom():
        raise ValueError("fail")

    with pytest.raises(ValueError):
        boom()

    last = tracing.get_last_trace()
    assert last["name"] == "boom"
    assert last["ok"] is False
    assert "error" in last


@pytest.mark.asyncio
async def test_trace_records_async(monkeypatch):
    tracing.clear_trace_records()
    monkeypatch.setattr(tracing, "_TRACE_STDOUT", False)

    @tracing.trace
    async def echo(x):
        await asyncio.sleep(0)
        return x

    result = await echo("hi")
    last = tracing.get_last_trace()

    assert result == "hi"
    assert last["name"] == "echo"
    assert last["ok"] is True


def test_trace_subtree_siblings(monkeypatch):
    tracing.clear_trace_records()
    monkeypatch.setattr(tracing, "_TRACE_STDOUT", False)

    @tracing.trace
    def root():
        child()
        sibling()

    @tracing.trace
    def child():
        return "c"

    @tracing.trace
    def sibling():
        return "s"

    root()
    tree = tracing.get_trace_subtree("root")

    names = [r["name"] for r in tree]
    assert names == ["root", "child", "sibling"]


def test_trace_subtree_case_insensitive(monkeypatch):
    tracing.clear_trace_records()
    monkeypatch.setattr(tracing, "_TRACE_STDOUT", False)

    @tracing.trace
    def MyFunc():
        return "ok"

    MyFunc()
    tree = tracing.get_trace_subtree("myfunc")
    assert [r["name"] for r in tree] == ["MyFunc"]


def test_trace_subtree_root_id_isolated(monkeypatch):
    tracing.clear_trace_records()
    monkeypatch.setattr(tracing, "_TRACE_STDOUT", False)

    @tracing.trace
    def root_a():
        child_a()

    @tracing.trace
    def child_a():
        return "a"

    @tracing.trace
    def root_b():
        return "b"

    root_a()
    root_b()
    tree = tracing.get_trace_subtree("root_a")
    names = [r["name"] for r in tree]
    assert names == ["root_a", "child_a"]
