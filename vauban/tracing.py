import os
from typing import Any, Callable, Optional

# Global state to track if weave is initialized
_WEAVE_INITIALIZED = False

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
        return

    project = project_name or os.getenv("WEAVE_PROJECT")
    if project and not _WEAVE_INITIALIZED:
        try:
            weave.init(project)
            _WEAVE_INITIALIZED = True
        except Exception as e:
            print(f"Warning: Failed to initialize Weave: {e}")


def trace(func_or_name: Optional[Any] = None) -> Callable:
    """
    Decorator to trace functions with Weave if available.
    Usage:
        @trace
        def my_func(...): ...
        
        @trace("custom_name")
        def my_func(...): ...
    """
    if not WEAVE_AVAILABLE:
        # If called as @trace(name="foo") or @trace("foo")
        if isinstance(func_or_name, str):
            def decorator(func):
                return func
            return decorator
        # If called as @trace
        return func_or_name if callable(func_or_name) else (lambda x: x)

    # Weave is available, use weave.op()
    if isinstance(func_or_name, str):
        return weave.op(name=func_or_name)
    elif callable(func_or_name):
        return weave.op()(func_or_name)
    else:
        return weave.op()

