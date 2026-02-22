# pylint: disable=missing-module-docstring, broad-exception-caught, unused-import
import os
import sys
import traceback

sys.path.append(os.path.abspath("."))

try:
    print("Import successful")
except Exception:
    traceback.print_exc()
