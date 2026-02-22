# pylint: disable=missing-module-docstring, import-error, unspecified-encoding
import os
import sys

import pytest

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.getcwd()))

if __name__ == "__main__":
    # Run tests and output to a file
    with open("test_results.txt", "w") as f:
        sys.stdout = f
        sys.stderr = f
        print("Starting tests...")
        ret = pytest.main(["tests/", "-v", "--tb=short"])
        print(f"\nTests finished with exit code {ret}")
