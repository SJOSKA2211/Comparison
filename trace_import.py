import traceback
import sys
import os

sys.path.append(os.path.abspath("."))

try:
    import src.api.main
    print("Import successful")
except Exception:
    traceback.print_exc()
