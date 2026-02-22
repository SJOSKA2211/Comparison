import re

content = open("src/pricing/numerical_methods.py").read()

# 1. Rename arguments and variables
replacements = {
    r"\bS\b": "spot",
    r"\bK\b": "strike",
    r"\bT\b": "time_to_maturity",
    r"\br\b": "rate",
    r"\bsigma\b": "volatility",
    r"\bN\b": "steps",  # Careful, check context
    r"\bM\b": "grid_steps",
    r"\bS_max\b": "spot_max",
    r"\bS_grid\b": "spot_grid",
    r"\bS_T\b": "spot_terminal",
    r"\bV\b": "option_values",
    r"\bV_new\b": "option_values_new",
    r"\bV_2N\b": "price_high_res",
    r"\bV_N\b": "price_low_res",
    r"\bsqrt_T\b": "sqrt_time",
    r"\bd1\b": "d_one",
    r"\bd2\b": "d_two",
    r"\bN_d1\b": "cdf_d1",
    r"\bN_d2\b": "cdf_d2",
    r"\bn_d1\b": "pdf_d1",
    r"\bN_neg_d1\b": "cdf_neg_d1",
    r"\bN_neg_d2\b": "cdf_neg_d2",
    r"\bA_diag\b": "a_diag",
    r"\bA_lower\b": "a_lower",
    r"\bA_upper\b": "a_upper",
    r"\bB_diag\b": "b_diag",
    r"\bB_lower\b": "b_lower",
    r"\bB_upper\b": "b_upper",
    r"\bZ\b": "random_variates",
    r"\bdf\b": "discount_factor",
    r"\bdt\b": "delta_t",
    r"\bdS\b": "delta_s",
}

# Apply replacements
for pattern, repl in replacements.items():
    # Use regex to match whole words only, avoid replacing inside other words
    content = re.sub(pattern, repl, content)

# 2. Fix specific issues
# Remove unused variables in trinomial tree
content = re.sub(r"\s+d = 1 / u\n", "", content)
content = re.sub(r"\s+m = 1  # middle factor\n", "", content)

# Fix N variable in crank_nicolson_price which was renamed to steps
# But in crank_nicolson_price arguments were M and N.
# M -> grid_steps
# N -> steps. Wait, N in crank_nicolson is time steps. N in trinomial is also time steps.
# So steps is fine.

# 3. Strip trailing whitespace
lines = content.splitlines()
lines = [line.rstrip() for line in lines]
content = "\n".join(lines) + "\n"

# 4. Fix imports
if "import numpy" not in content:
    content = "import numpy as np\n" + content
if "scipy.stats" not in content:
    content = "from scipy.stats import norm\n" + content

# Ensure imports are at the top and correct
# (Simple approach: replace the top part)
import_block = """\"\"\"
Numerical methods for option pricing.
\"\"\"
import math
import time
from typing import Literal
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
"""

# Replace existing imports with the clean block
# Find where the imports end? Or just blindly replace?
# The original file had docstring then imports.
# Let's try to reconstruct the file.

with open("src/pricing/numerical_methods.py", "w") as f:
    f.write(content)
