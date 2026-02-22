import os
import sys

# Add src to path
sys.path.append(os.path.abspath("."))

modules_to_check = [
    "src.database",
    "src.api.main",
    "src.api.routers.trading",
    "src.pricing.numerical_methods",
    "src.models.market",
    "src.ml.feature_engineering",
]

results = []
for mod in modules_to_check:
    try:
        __import__(mod, fromlist=["*"])
        results.append(f"{mod}: OK")
    except Exception as e:
        results.append(f"{mod}: ERROR ({type(e).__name__}: {e})")

with open("import_verification.txt", "w") as f:
    f.write("\n".join(results))
    f.write("\n")

print("\n".join(results))
