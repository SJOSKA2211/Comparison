content = open("src/ml/autonomous_pipeline.py").read()
lines = content.splitlines()
lines = [line.rstrip() for line in lines]
content = "\n".join(lines) + "\n"
with open("src/ml/autonomous_pipeline.py", "w") as f:
    f.write(content)
