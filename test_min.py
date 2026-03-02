"""Minimal test script."""
import sys

print("Python works")
print(sys.executable)
try:
    print("FastAPI installed")
except ImportError:
    print("FastAPI missing")
