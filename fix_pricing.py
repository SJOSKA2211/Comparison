with open("src/pricing/numerical_methods.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "rhs[1:-1] =" in line:
        # rhs[1:-1] = B_lower[:-1] * V[1:-3] + B_diag[1:-1] * V[2:-2] + B_upper[1:] * V[3:-1]
        new_lines.append(
            "        rhs[1:-1] = B_lower[:-1] * V[1:-3] + B_diag[1:-1] * V[2:-2] + B_upper[1:] * V[3:-1]\n"
        )
    else:
        new_lines.append(line)

with open("src/pricing/numerical_methods.py", "w") as f:
    f.writelines(new_lines)
