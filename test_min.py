import sys
print("Python works")
print(sys.executable)
try:
    import fastapi
    print("FastAPI installed")
except ImportError:
    print("FastAPI missing")
