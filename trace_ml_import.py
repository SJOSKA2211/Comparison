"""Trace ML import script."""
import os
import sys
import traceback

sys.path.append(os.path.abspath("."))

try:
    print("Import successful")
except Exception:
    traceback.print_exc()
