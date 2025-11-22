import sys
import os

# Add .venv site-packages to path if not present (though running with python in venv should handle it)
# We will run this using the python in the venv

try:
    import weave
    print(f"Weave version: {weave.__version__}")
    if hasattr(weave, "Model"):
        print("weave.Model is available")
    else:
        print("weave.Model is NOT available")
        # Check if it's in a submodule
        print(f"Weave dir: {dir(weave)}")
except ImportError:
    print("weave not installed")
except Exception as e:
    print(f"Error: {e}")
