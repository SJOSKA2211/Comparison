# pylint: disable=missing-module-docstring, unspecified-encoding
packages = ["pydantic", "pydantic_settings", "aiosqlite", "structlog", "numpy", "scipy"]

with open("diagnosis_full.txt", "w") as f:
    for pkg in packages:
        try:
            __import__(pkg)
            f.write(f"{pkg}: Installed\n")
        except ImportError as e:
            f.write(f"{pkg}: Missing ({e})\n")
