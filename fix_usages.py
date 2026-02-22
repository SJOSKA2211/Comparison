import re

files_to_fix = [
    "src/ml/autonomous_pipeline.py",
    "src/api/routers/pricing.py",
    "src/api/local_main.py"
]

replacements = {
    r"S=": "spot=",
    r"K=": "strike=",
    r"r=": "rate=",
    r"sigma=": "volatility=",
    r"T=": "time_to_maturity=",
}

for file_path in files_to_fix:
    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Apply replacements
        for pattern, repl in replacements.items():
            content = re.sub(pattern, repl, content)

        with open(file_path, "w") as f:
            f.write(content)
        print(f"Fixed {file_path}")
    except FileNotFoundError:
        print(f"Skipped {file_path} (not found)")
