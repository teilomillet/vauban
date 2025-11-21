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
        raise RuntimeError("WEAVE_PROJECT is required to log traces; set env before running.")
    init_weave(project)
    return project


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

    if not WEAVE_AVAILABLE:
        # TODO: Add tests covering @trace, @trace("name"), and @trace(name="name") without Weave.
        # No-op decorator when Weave is absent; must still accept name kwarg to match call sites.
        if callable(func_or_name) and not isinstance(func_or_name, str) and name is None:
            return func_or_name  # @trace directly on a function

        def decorator(func):
            return func

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
        return _wrap_with_init(_op()(func_or_name))

    # Usage: @trace("name") or @trace(name="name")
    def decorator(func):
        return _wrap_with_init(_op()(func))

    return decorator
