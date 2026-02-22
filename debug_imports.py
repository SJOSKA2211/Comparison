# pylint: disable=missing-module-docstring, broad-exception-caught
import os
import sys
import traceback

sys.path.append(os.path.abspath("."))

modules = [
    "src.database",
    "src.api.main",
    "src.api.routers.trading",
    "src.pricing.numerical_methods",
    "src.models.market",
    "src.ml.feature_engineering",
]

for mod in modules:
    print(f"--- Importing {mod} ---")
    try:
        __import__(mod)
        print("OK")
    except Exception:
        traceback.print_exc()
    print("\n")
