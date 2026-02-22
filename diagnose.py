import sys

with open("diagnosis.txt", "w") as f:
    f.write(f"Python: {sys.version}\n")
    f.write(f"Executable: {sys.executable}\n")
    try:
        import fastapi  # noqa: F401
        f.write("FastAPI: Installed\n")
    except ImportError:
        f.write("FastAPI: Missing\n")

    try:
        import uvicorn  # noqa: F401
        f.write("Uvicorn: Installed\n")
    except ImportError:
        f.write("Uvicorn: Missing\n")
