# pylint: disable=missing-module-docstring, unused-import, import-error
import sys

print("Python works")
print(sys.executable)
try:
    import fastapi  # noqa: F401
    print("FastAPI installed")
except ImportError:
    print("FastAPI missing")
