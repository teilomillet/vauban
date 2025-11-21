import weave
print(dir(weave))
try:
    with weave.op(name="test"):
        print("Context manager works")
except Exception as e:
    print(f"Context manager failed: {e}")

try:
    with weave.attributes({"a": 1}):
        print("Attributes works")
except Exception as e:
    print(f"Attributes failed: {e}")
