#!/bin/bash
# =============================================================================
# BS-Opt Local Development Setup (No Docker)
# =============================================================================

set -e

echo "=========================================="
echo "  BS-Opt Local Development Setup"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -----------------------------------------------------------------------------
# Check Python
# -----------------------------------------------------------------------------
check_python() {
    echo -e "\n${GREEN}[1/5] Checking Python...${NC}"
    
    if command -v python3 &> /dev/null; then
        PYTHON=python3
    elif command -v python &> /dev/null; then
        PYTHON=python
    else
        echo "❌ Python not found. Please install Python 3.10+"
        exit 1
    fi
    
    VERSION=$($PYTHON --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
    echo "✅ Found Python $VERSION"
}

# -----------------------------------------------------------------------------
# Create Virtual Environment
# -----------------------------------------------------------------------------
setup_venv() {
    echo -e "\n${GREEN}[2/5] Setting up virtual environment...${NC}"
    
    if [ ! -d ".venv" ]; then
        $PYTHON -m venv .venv
        echo "✅ Created virtual environment"
    else
        echo "✅ Virtual environment exists"
    fi
    
    source .venv/bin/activate
    pip install --upgrade pip -q
}

# -----------------------------------------------------------------------------
# Install Dependencies
# -----------------------------------------------------------------------------
install_deps() {
    echo -e "\n${GREEN}[3/5] Installing dependencies...${NC}"
    
    pip install -r requirements/dev.txt -q
    echo "✅ Dependencies installed"
}

# -----------------------------------------------------------------------------
# Setup Local Database (SQLite for dev)
# -----------------------------------------------------------------------------
setup_database() {
    echo -e "\n${GREEN}[4/5] Setting up local database...${NC}"
    
    if [ ! -f "data/bsopt.db" ]; then
        mkdir -p data
        $PYTHON -c "
import sqlite3
conn = sqlite3.connect('data/bsopt.db')
conn.executescript('''
-- Users table
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT,
    role TEXT DEFAULT 'trader' CHECK (role IN ('trader', 'researcher', 'admin')),
    email_verified INTEGER DEFAULT 0,
    display_name TEXT,
    avatar_url TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    user_id TEXT REFERENCES users(id),
    token TEXT UNIQUE NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Market ticks (simple version)
CREATE TABLE IF NOT EXISTS market_ticks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT DEFAULT (datetime('now')),
    symbol TEXT NOT NULL,
    price REAL NOT NULL,
    volume REAL,
    bid REAL,
    ask REAL
);

-- Experiments table
CREATE TABLE IF NOT EXISTS numerical_experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    researcher_id TEXT REFERENCES users(id),
    spot_price REAL,
    strike REAL,
    risk_free_rate REAL,
    volatility REAL,
    time_to_maturity REAL,
    option_type TEXT,
    analytical_price REAL,
    fdm_price REAL,
    fdm_time_us INTEGER,
    mc_price REAL,
    mc_time_us INTEGER,
    tree_price REAL,
    tree_time_us INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Create test user
INSERT OR IGNORE INTO users (id, email, password_hash, role, email_verified)
VALUES ('test-user-id', 'test@example.com', 'pbkdf2:sha256:test', 'researcher', 1);
''')
conn.close()
print('✅ SQLite database created at data/bsopt.db')
"
    else
        echo "✅ Database exists"
    fi
}

# -----------------------------------------------------------------------------
# Create .env file
# -----------------------------------------------------------------------------
setup_env() {
    echo -e "\n${GREEN}[5/5] Setting up environment...${NC}"
    
    if [ ! -f ".env" ]; then
        cat > .env << 'EOF'
# Local Development Configuration
ENVIRONMENT=development
DATABASE_URL=sqlite:///data/bsopt.db
JWT_SECRET=dev-secret-change-in-production
FRONTEND_URL=http://localhost:3000
API_URL=http://localhost:8000
EOF
        echo "✅ Created .env file"
    else
        echo "✅ .env exists"
    fi
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    check_python
    setup_venv
    install_deps
    setup_database
    setup_env
    
    echo -e "\n${GREEN}=========================================="
    echo "  Setup Complete!"
    echo "==========================================${NC}"
    echo ""
    echo "To activate the environment:"
    echo "  source .venv/bin/activate"
    echo ""
    echo "To run the API:"
    echo "  python -m src.api.local_main"
    echo ""
    echo "To run tests:"
    echo "  PYTHONPATH=. pytest tests/ -v"
    echo ""
    echo "API will be available at: http://localhost:8000"
    echo "Docs at: http://localhost:8000/docs"
}

main "$@"
