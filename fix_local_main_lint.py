import re

file_path = "src/api/local_main.py"
with open(file_path, "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Strip trailing whitespace
    line = line.rstrip() + "\n"
    new_lines.append(line)

with open(file_path, "w") as f:
    f.writelines(new_lines)
