# Fix test_api.py
with open("test_api.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Fix indentation: replace 13 spaces with 12
    if line.startswith("             "):
        line = line.replace("             ", "            ", 1)

    # Strip trailing whitespace
    line = line.rstrip() + "\n"
    new_lines.append(line)

with open("test_api.py", "w") as f:
    f.writelines(new_lines)

# Fix verify_api_local.py
with open("verify_api_local.py", "r") as f:
    content = f.read()

lines = content.splitlines()
lines = [line.rstrip() for line in lines]
content = "\n".join(lines) + "\n"

with open("verify_api_local.py", "w") as f:
    f.write(content)
