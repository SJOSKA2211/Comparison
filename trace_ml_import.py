import traceback
import sys
import os

sys.path.append(os.path.abspath("."))

try:
    import src.ml.feature_engineering
    print("Import successful")
except Exception:
    traceback.print_exc()
