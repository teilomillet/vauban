import weave

# weave.op is a decorator, not a context manager.
# It tracks inputs/outputs of the decorated function.

@weave.op()
def test_op(x: int):
    return x * 2

try:
    # Weave requires init() before ops are tracked if you want to log to a project.
    # But for local testing without a project, it might just run without logging or error if not init.
    # Let's assume we just want to verify the import and decorator mechanics work.
    print(f"Op result: {test_op(10)}")
    print("Decorator works")
except Exception as e:
    print(f"Decorator failed: {e}")

try:
    # weave.attributes IS a context manager for adding metadata to the current span.
    with weave.attributes({"a": 1}):
        print("Attributes works")
except Exception as e:
    print(f"Attributes failed: {e}")
