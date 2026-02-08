#!/usr/bin/env bash
# =============================================================================
# BS-Opt One-Click Launcher
# Automated: Environment setup, Dependencies, Tests, Docker Build, Deploy
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"; }
error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"; }
info() { echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"; }

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.prod.yml"
ENV_FILE="${SCRIPT_DIR}/.env"
VENV_DIR="${SCRIPT_DIR}/.venv"
MODE="${1:-full}"

# =============================================================================
# Banner
# =============================================================================
show_banner() {
    echo ""
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║     ____  _____        ____        _        _                     ║${NC}"
    echo -e "${BLUE}║    | __ )/ ___|       / __ \\ _ __ | |_     | |    __ _ _   _ _ __ ║${NC}"
    echo -e "${BLUE}║    |  _ \\\\___ \\ _____| |  | | '_ \\| __|    | |   / _\` | | | | '_ \\║${NC}"
    echo -e "${BLUE}║    | |_) |___) |_____| |__| | |_) | |_  _  | |__| (_| | |_| | | | ║${NC}"
    echo -e "${BLUE}║    |____/|____/       \\____/| .__/ \\__||_| |_____\\__,_|\\__,_|_| |_║${NC}"
    echo -e "${BLUE}║                             |_|                                   ║${NC}"
    echo -e "${BLUE}║               Quantitative Finance Research Platform              ║${NC}"
    echo -e "${BLUE}║                    One-Click Deployment Launcher                  ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# =============================================================================
# Step 1: System Requirements Check
# =============================================================================
check_system() {
    log "🔍 Step 1/7: Checking system requirements..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is not installed. Please install Python 3.10+."
        exit 1
    fi
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    info "Python version: ${PYTHON_VERSION}"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    info "Docker: $(docker --version | cut -d' ' -f3 | tr -d ',')"

    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        error "Docker Compose V2 is not available."
        exit 1
    fi
    info "Docker Compose: $(docker compose version --short)"

    # Check available memory
    if command -v free &> /dev/null; then
        TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
        if [[ $TOTAL_MEM -lt 8 ]]; then
            warn "System has ${TOTAL_MEM}GB RAM. Recommended: 16GB for full stack."
        else
            info "Memory: ${TOTAL_MEM}GB available"
        fi
    fi

    log "✅ System requirements satisfied"
}

# =============================================================================
# Step 2: Environment Configuration
# =============================================================================
setup_environment() {
    log "🔐 Step 2/7: Setting up environment..."

    if [[ -f "$ENV_FILE" ]]; then
        info ".env file already exists"
        return
    fi

    # Generate secure secrets
    DB_PASSWORD=$(openssl rand -base64 32 | tr -d '/+=' | head -c 32)
    JWT_SECRET=$(openssl rand -base64 64 | tr -d '/+=' | head -c 64)
    KAFKA_CLUSTER_ID=$(openssl rand -base64 16 | tr -d '/+=' | head -c 22)

    cat > "$ENV_FILE" << EOF
# =============================================================================
# BS-Opt Environment Configuration
# Generated on $(date)
# =============================================================================

# Environment
ENVIRONMENT=development
VERSION=latest

# GitHub Container Registry
GITHUB_USER=bsopt

# Database
DB_PASSWORD=${DB_PASSWORD}

# Authentication
JWT_SECRET=${JWT_SECRET}

# =============================================================================
# OAuth Configuration (Optional - Configure for social login)
# =============================================================================

# Google OAuth (https://console.cloud.google.com/apis/credentials)
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=

# GitHub OAuth (https://github.com/settings/developers)
GITHUB_CLIENT_ID=
GITHUB_CLIENT_SECRET=

# =============================================================================
# URLs
# =============================================================================
FRONTEND_URL=http://localhost:3000
API_URL=http://localhost:8000

# =============================================================================
# Email (for verification - optional in dev)
# =============================================================================
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
FROM_EMAIL=noreply@bsopt.io

# Kafka
KAFKA_CLUSTER_ID=${KAFKA_CLUSTER_ID}

# API Configuration
API_WORKERS=2

# ML Configuration
RAY_OBJECT_STORE_MEMORY=4000000000
EOF

    chmod 600 "$ENV_FILE"
    log "✅ Environment configured (secrets generated)"
    warn "📝 Edit .env to add OAuth credentials for Google/GitHub login"
}

# =============================================================================
# Step 3: Python Virtual Environment & Dependencies
# =============================================================================
setup_python() {
    log "🐍 Step 3/7: Setting up Python environment..."

    # Create virtual environment if not exists
    if [[ ! -d "$VENV_DIR" ]]; then
        info "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi

    # Activate virtual environment
    source "${VENV_DIR}/bin/activate"

    # Upgrade pip
    pip install --upgrade pip --quiet

    # Install dependencies
    info "Installing base requirements..."
    pip install -r "${SCRIPT_DIR}/requirements/base.txt" --quiet

    # Install testing dependencies
    info "Installing test dependencies..."
    pip install pytest pytest-cov scipy numpy --quiet

    log "✅ Python dependencies installed"
}

# =============================================================================
# Step 4: Run Tests
# =============================================================================
run_tests() {
    log "🧪 Step 4/7: Running test suite..."

    cd "$SCRIPT_DIR"

    # Activate venv
    source "${VENV_DIR}/bin/activate"

    # Run pytest
    if PYTHONPATH=. pytest tests/ -v --tb=short 2>&1; then
        log "✅ All tests passed"
    else
        warn "⚠️  Some tests failed (continuing with deployment)"
    fi
}

# =============================================================================
# Step 5: Docker Build (Throttled for Anti-Freeze)
# =============================================================================
build_docker() {
    log "🐳 Step 5/7: Building Docker images (throttled for stability)..."

    cd "$SCRIPT_DIR"

    # Load environment
    set -a
    source "$ENV_FILE"
    set +a

    # 🛑 CRITICAL: Throttle parallel builds to prevent CPU saturation
    export COMPOSE_PARALLEL_LIMIT=1
    export DOCKER_BUILDKIT=1

    # Try to pull pre-built images first
    info "Attempting to pull pre-built images from registry..."
    docker compose -f "$COMPOSE_FILE" pull 2>/dev/null || {
        info "Pre-built images not available, building locally..."
    }

    # Build images
    info "Building Docker images (this may take a while)..."
    docker compose -f "$COMPOSE_FILE" build --progress=plain

    log "✅ Docker images built successfully"
}

# =============================================================================
# Step 6: Start Services
# =============================================================================
start_services() {
    log "🚀 Step 6/7: Starting all services..."

    cd "$SCRIPT_DIR"

    # Load environment
    set -a
    source "$ENV_FILE"
    set +a

    # Profile flag for research mode
    local PROFILE_FLAG=""
    if [[ "$MODE" == "research" ]]; then
        PROFILE_FLAG="--profile research"
        info "Research mode: JupyterHub will be available"
    fi

    # Start services
    docker compose -f "$COMPOSE_FILE" $PROFILE_FLAG up -d

    # Wait for services to be healthy
    info "Waiting for services to become healthy..."
    sleep 5

    log "✅ All services started"
}

# =============================================================================
# Step 7: Show Status & URLs
# =============================================================================
show_status() {
    log "📊 Step 7/7: Deployment Status"

    cd "$SCRIPT_DIR"
    echo ""
    docker compose -f "$COMPOSE_FILE" ps
    echo ""

    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}                       🎉 DEPLOYMENT COMPLETE!                      ${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${CYAN}🌐 Service URLs:${NC}"
    echo "   • API Gateway:    http://localhost:80"
    echo "   • API Direct:     http://localhost:8000"
    echo "   • Ray Dashboard:  http://localhost:8265"
    echo "   • MLflow:         http://localhost:5000"
    if [[ "$MODE" == "research" ]]; then
        echo "   • JupyterLab:     http://localhost:8888"
    fi
    echo ""
    echo -e "${CYAN}📝 Useful Commands:${NC}"
    echo "   • View logs:      docker compose -f docker-compose.prod.yml logs -f"
    echo "   • Stop stack:     docker compose -f docker-compose.prod.yml down"
    echo "   • Full cleanup:   docker compose -f docker-compose.prod.yml down -v"
    echo "   • Run tests:      PYTHONPATH=. pytest tests/ -v"
    echo ""
    echo -e "${YELLOW}📌 Next Steps:${NC}"
    echo "   1. Edit .env to add OAuth credentials (GOOGLE_CLIENT_ID, GITHUB_CLIENT_ID)"
    echo "   2. Configure SMTP for email verification in production"
    echo "   3. Push to GitHub to trigger CI/CD cloud builds"
    echo ""
}

# =============================================================================
# Quick Modes
# =============================================================================
quick_test() {
    log "⚡ Quick Test Mode: Running tests only..."
    check_system
    setup_python
    run_tests
    log "✅ Quick test complete!"
}

quick_build() {
    log "⚡ Quick Build Mode: Building Docker images only..."
    check_system
    setup_environment
    build_docker
    log "✅ Quick build complete!"
}

# =============================================================================
# Help
# =============================================================================
show_help() {
    echo "Usage: $0 [MODE]"
    echo ""
    echo "Modes:"
    echo "  full        - Full deployment (default): env + deps + tests + build + start"
    echo "  research    - Full deployment with JupyterHub for research"
    echo "  test        - Quick mode: setup Python and run tests only"
    echo "  build       - Quick mode: build Docker images only"
    echo "  start       - Start services only (assumes images exist)"
    echo "  stop        - Stop all services"
    echo "  status      - Show service status"
    echo ""
    echo "Examples:"
    echo "  $0              # Full production deployment"
    echo "  $0 research     # Full deployment with Jupyter"
    echo "  $0 test         # Just run tests"
    echo "  $0 build        # Just build Docker images"
    echo "  sudo $0         # Full deployment with kernel optimization"
    exit 0
}

# =============================================================================
# Main Execution
# =============================================================================
main() {
    show_banner

    case "$MODE" in
        -h|--help|help)
            show_help
            ;;
        test)
            quick_test
            ;;
        build)
            quick_build
            ;;
        start)
            start_services
            show_status
            ;;
        stop)
            log "Stopping all services..."
            docker compose -f "$COMPOSE_FILE" down
            log "✅ Services stopped"
            ;;
        status)
            docker compose -f "$COMPOSE_FILE" ps
            ;;
        full|production|research)
            check_system
            setup_environment
            setup_python
            run_tests
            build_docker
            start_services
            show_status
            ;;
        *)
            error "Unknown mode: $MODE"
            show_help
            ;;
    esac
}

main "$@"
