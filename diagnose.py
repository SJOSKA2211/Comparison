import sys

with open("diagnosis.txt", "w") as f:
    f.write(f"Python: {sys.version}\n")
    f.write(f"Executable: {sys.executable}\n")
    try:
        f.write("FastAPI: Installed\n")
    except ImportError:
        f.write("FastAPI: Missing\n")

    try:
        f.write("Uvicorn: Installed\n")
    except ImportError:
        f.write("Uvicorn: Missing\n")
