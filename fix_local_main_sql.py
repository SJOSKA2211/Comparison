with open("src/api/local_main.py", "r") as f:
    content = f.read()

# Fix SQL Injection
# Original: f"INSERT INTO sessions (user_id, token, expires_at) VALUES (?, ?, {expires_at})",
# New: "INSERT INTO sessions (user_id, token, expires_at) VALUES (?, ?, ?)", (user_id, token, expires_at)

# NOTE: expires_at variable in local_main.py is likely a string "datetime('now', '+24 hours')"
# If it is a string containing SQL function call, we cannot parameterize it directly in aiosqlite/sqlite3
# because ? parameters are for values, not expressions.
# HOWEVER, if expires_at is "datetime('now', '+24 hours')", passing it as a string literal (?) will make it a string 'datetime...' in DB, not the calculated time.
# But looking at local_main.py (read earlier):
# expires_at = "datetime('now', '+24 hours')"
# f"INSERT ... VALUES (?, ?, {expires_at})" -> VALUES (?, ?, datetime('now', '+24 hours'))
# This effectively hardcodes the function call.
# Bandit complains because it sees f-string constructing SQL.
# To fix bandit B608 AND keep functionality:
# We should allow this specific case if it's safe (it is a hardcoded string variable), OR use parameterized query properly.
# But we can't parameterize a function call like that.
# Better approach: Calculate the expiration time in Python and pass it as a parameter.

# Let's see imports in local_main.py
# import time
# from datetime import datetime, timedelta ... wait, I need to check imports.

# I'll check imports first.
import re

if "from datetime import datetime" not in content:
    # Add imports
    content = content.replace("import secrets\n", "import secrets\nfrom datetime import datetime, timedelta\n")

# Now locate the register_user function and fix logic
# Old logic:
# expires_at = "datetime('now', '+24 hours')"
# await db.execute(f"INSERT ... {expires_at})", ...)

# New logic:
# expires_at = datetime.now() + timedelta(hours=24)
# await db.execute("INSERT ... VALUES (?, ?, ?)", (user_id, token, expires_at))

# Regex replacement
# Find the block
pattern_old = r'    expires_at = "datetime\(\'now\', \'\+24 hours\'\)"\n    await db\.execute\(\n        f"INSERT INTO sessions \(user_id, token, expires_at\) VALUES \(\?, \?, \{expires_at\}\)",\n        \(user_id, token\)\n    \)'

pattern_new = r'    expires_at = datetime.now() + timedelta(hours=24)\n    await db.execute(\n        "INSERT INTO sessions (user_id, token, expires_at) VALUES (?, ?, ?)",\n        (user_id, token, expires_at)\n    )'

# Use string replace if regex is too complex
search_str = """    expires_at = "datetime('now', '+24 hours')"
    await db.execute(
        f"INSERT INTO sessions (user_id, token, expires_at) VALUES (?, ?, {expires_at})",
        (user_id, token)
    )"""

replace_str = """    expires_at = datetime.now() + timedelta(hours=24)
    await db.execute(
        "INSERT INTO sessions (user_id, token, expires_at) VALUES (?, ?, ?)",
        (user_id, token, expires_at)
    )"""

if search_str in content:
    content = content.replace(search_str, replace_str)
else:
    print("Could not find SQL injection pattern")

with open("src/api/local_main.py", "w") as f:
    f.write(content)
