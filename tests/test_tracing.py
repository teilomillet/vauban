import os
import pytest
from unittest.mock import patch, MagicMock
from vauban.tracing import trace, _TRACE_BUFFER, clear_trace_records

@pytest.fixture(autouse=True)
def clean_trace_buffer():
    clear_trace_records()
    yield
    clear_trace_records()

def test_trace_decorator_no_args():
    """Test @trace without arguments."""
    
    @trace
    def my_func(x):
        return x + 1
        
    result = my_func(10)
    assert result == 11
    
    # Check buffer
    assert len(_TRACE_BUFFER) == 1
    entry = _TRACE_BUFFER[0]
    assert entry["name"] == "my_func"
    assert entry["ok"] is True
    assert entry["result"] == "11"

def test_trace_decorator_with_name_arg():
    """Test @trace('custom_name')."""
    
    @trace("custom_op")
    def my_func(x):
        return x * 2
        
    result = my_func(5)
    assert result == 10
    
    assert len(_TRACE_BUFFER) == 1
    entry = _TRACE_BUFFER[0]
    assert entry["name"] == "custom_op"

def test_trace_decorator_with_kwarg_name():
    """Test @trace(name='custom_name')."""
    
    @trace(name="named_op")
    def my_func():
        return "done"
        
    assert my_func() == "done"
    
    assert len(_TRACE_BUFFER) == 1
    entry = _TRACE_BUFFER[0]
    assert entry["name"] == "named_op"

def test_trace_async_function():
    """Test tracing an async function."""
    import asyncio
    
    @trace
    async def async_op():
        await asyncio.sleep(0.01)
        return "async_result"
        
    result = asyncio.run(async_op())
    assert result == "async_result"
    
    assert len(_TRACE_BUFFER) == 1
    entry = _TRACE_BUFFER[0]
    assert entry["name"] == "async_op"
    assert entry["duration_ms"] > 0

def test_trace_error_capture():
    """Test that exceptions are captured in trace."""
    
    @trace
    def failing_op():
        raise ValueError("oops")
        
    with pytest.raises(ValueError):
        failing_op()
        
    assert len(_TRACE_BUFFER) == 1
    entry = _TRACE_BUFFER[0]
    assert entry["name"] == "failing_op"
    assert entry["ok"] is False
    assert "ValueError" in entry["error"]
