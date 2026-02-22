import re

file_path = "src/api/routers/trading.py"
with open(file_path, "r") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    # Fix indentation on line 152 (was 17 spaces, should be 16)
    # Check context around line 152 in current file state
    if "new_order.status = OrderStatus.REJECTED # Insufficient funds" in line:
        line = line.replace("                 ", "                ")

    # Strip trailing whitespace
    line = line.rstrip() + "\n"

    new_lines.append(line)

with open(file_path, "w") as f:
    f.writelines(new_lines)
