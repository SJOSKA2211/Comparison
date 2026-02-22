import re

file_path = "src/api/routers/trading.py"
with open(file_path, "r") as f:
    content = f.read()

# Add docstring
if not content.startswith('"""'):
    content = '"""Trading API router."""\n' + content

# Fix unused imports
content = content.replace("from fastapi import APIRouter, Depends, HTTPException, status", "from fastapi import APIRouter, Depends, HTTPException")
content = content.replace("from sqlalchemy import select, func", "from sqlalchemy import select")
# Note: PositionResponse might be used in type hint via string, but linter complained.
# It is imported in `from src.schemas.trading import ... PositionResponse`.
# It is used in PortfolioResponse definition in schemas, but here in router it is used in endpoint signature?
# No, it's not used in this file directly.
content = content.replace(",\n    PositionResponse", "")

with open(file_path, "w") as f:
    f.write(content)
