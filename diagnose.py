# ruff: noqa: F401
# pylint: disable=unused-import
import sys

with open("diagnosis.txt", "w") as f:
    f.write(f"Python: {sys.version}\n")
    f.write(f"Executable: {sys.executable}\n")
    try:
        import fastapi
        f.write("FastAPI: Installed\n")
    except ImportError:
        f.write("FastAPI: Missing\n")

    try:
        import uvicorn
        f.write("Uvicorn: Installed\n")
    except ImportError:
        f.write("Uvicorn: Missing\n")
