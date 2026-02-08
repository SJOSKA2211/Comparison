#!/usr/bin/env bash
# =============================================================================
# BS-Opt Deployment Script
# One-click deploy with kernel tuning and anti-freeze build strategy
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"; }
error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"; }

# Default configuration
COMPOSE_FILE="docker-compose.prod.yml"
MODE="${1:-production}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# System Requirements Check
# =============================================================================
check_requirements() {
    log "🔍 Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        error "Docker Compose V2 is not available."
        exit 1
    fi
    
    # Check available memory
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $TOTAL_MEM -lt 8 ]]; then
        warn "System has ${TOTAL_MEM}GB RAM. Recommended: 16GB for full stack."
        warn "Consider using './deploy.sh research' for reduced resource mode."
    fi
    
    log "✅ System requirements satisfied"
}

# =============================================================================
# Kernel Optimization for C100k Connections
# =============================================================================
tune_kernel() {
    log "🔧 Optimizing kernel parameters for high-performance networking..."
    
    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        warn "Kernel tuning requires root. Run with sudo for C100k optimization."
        warn "Skipping kernel tuning..."
        return
    fi
    
    # Network performance tuning
    cat > /etc/sysctl.d/99-bsopt.conf << 'EOF'
# BS-Opt Network Performance Tuning
# Optimized for C100k concurrent connections

# Increase system-wide file descriptors
fs.file-max = 2097152
fs.nr_open = 2097152

# TCP/IP Stack Optimization
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 65535
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216

# TCP Buffer Sizes
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216

# TCP Connection Handling
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_max_tw_buckets = 1440000
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 15
net.ipv4.tcp_keepalive_time = 300
net.ipv4.tcp_keepalive_probes = 5
net.ipv4.tcp_keepalive_intvl = 15

# Enable TCP Fast Open
net.ipv4.tcp_fastopen = 3

# Local Port Range
net.ipv4.ip_local_port_range = 1024 65535
EOF

    sysctl -p /etc/sysctl.d/99-bsopt.conf > /dev/null 2>&1
    
    # Increase file descriptor limits
    cat > /etc/security/limits.d/99-bsopt.conf << 'EOF'
*    soft    nofile    1048576
*    hard    nofile    1048576
root soft    nofile    1048576
root hard    nofile    1048576
EOF
    
    log "✅ Kernel parameters optimized"
}

# =============================================================================
# Generate Secrets
# =============================================================================
generate_secrets() {
    log "🔐 Generating secure secrets..."
    
    ENV_FILE="${SCRIPT_DIR}/.env"
    
    if [[ -f "$ENV_FILE" ]]; then
        warn ".env file already exists. Skipping secret generation."
        warn "Delete .env to regenerate secrets."
        return
    fi
    
    # Generate secure random secrets
    DB_PASSWORD=$(openssl rand -base64 32 | tr -d '/+=' | head -c 32)
    JWT_SECRET=$(openssl rand -base64 64 | tr -d '/+=' | head -c 64)
    KAFKA_CLUSTER_ID=$(openssl rand -base64 16 | tr -d '/+=' | head -c 22)
    
    cat > "$ENV_FILE" << EOF
# =============================================================================
# BS-Opt Environment Configuration
# Generated on $(date)
# =============================================================================

# Environment
ENVIRONMENT=${MODE}
VERSION=latest

# GitHub Container Registry (update with your username)
GITHUB_USER=bsopt

# Database
DB_PASSWORD=${DB_PASSWORD}

# Authentication
JWT_SECRET=${JWT_SECRET}

# Kafka
KAFKA_CLUSTER_ID=${KAFKA_CLUSTER_ID}

# API Configuration
API_WORKERS=2

# ML Configuration
RAY_OBJECT_STORE_MEMORY=4000000000
EOF
    
    chmod 600 "$ENV_FILE"
    log "✅ Secrets generated and saved to .env"
}

# =============================================================================
# Create Required Directories
# =============================================================================
scaffold_directories() {
    log "📁 Creating directory structure..."
    
    mkdir -p "${SCRIPT_DIR}"/{config/ssl,notebooks,logs,data}
    
    # Create SSL placeholder if not exists
    if [[ ! -f "${SCRIPT_DIR}/config/ssl/.gitkeep" ]]; then
        touch "${SCRIPT_DIR}/config/ssl/.gitkeep"
    fi
    
    log "✅ Directory structure ready"
}

# =============================================================================
# Pull Images (Anti-Freeze Strategy)
# =============================================================================
pull_images() {
    log "🐳 Pulling pre-built images from registry..."
    log "   This prevents CPU-intensive local builds."
    
    # Load environment
    set -a
    source "${SCRIPT_DIR}/.env"
    set +a
    
    # Try to pull - don't fail if images don't exist yet
    docker compose -f "$COMPOSE_FILE" pull 2>/dev/null || {
        warn "Could not pull images from registry."
        warn "Images will be built locally (may take time)."
    }
}

# =============================================================================
# Build Images (Throttled)
# =============================================================================
build_images() {
    log "🔨 Building missing images (throttled to prevent system freeze)..."
    
    # 🛑 CRITICAL: Throttle parallel builds to prevent CPU saturation
    export COMPOSE_PARALLEL_LIMIT=1
    export DOCKER_BUILDKIT=1
    
    # Load environment
    set -a
    source "${SCRIPT_DIR}/.env"
    set +a
    
    docker compose -f "$COMPOSE_FILE" build
    
    log "✅ Images ready"
}

# =============================================================================
# Start Stack
# =============================================================================
start_stack() {
    log "🚀 Starting BS-Opt stack in ${MODE} mode..."
    
    # Load environment
    set -a
    source "${SCRIPT_DIR}/.env"
    set +a
    
    local PROFILE_FLAG=""
    if [[ "$MODE" == "research" ]]; then
        PROFILE_FLAG="--profile research"
        log "   📓 Research mode: JupyterHub will be available"
    fi
    
    docker compose -f "$COMPOSE_FILE" $PROFILE_FLAG up -d
    
    log "✅ Stack started successfully"
}

# =============================================================================
# Show Status
# =============================================================================
show_status() {
    log "📊 Deployment Status:"
    echo ""
    docker compose -f "$COMPOSE_FILE" ps
    echo ""
    
    log "🌐 Service URLs:"
    echo "   • API Gateway:    http://localhost:80"
    echo "   • API Direct:     http://localhost:8000"
    echo "   • Ray Dashboard:  http://localhost:8265"
    echo "   • MLflow:         http://localhost:5000"
    if [[ "$MODE" == "research" ]]; then
        echo "   • JupyterLab:     http://localhost:8888"
    fi
    echo ""
    
    log "📝 Useful Commands:"
    echo "   • View logs:      docker compose -f $COMPOSE_FILE logs -f"
    echo "   • Stop stack:     docker compose -f $COMPOSE_FILE down"
    echo "   • Full cleanup:   docker compose -f $COMPOSE_FILE down -v"
}

# =============================================================================
# Main Execution
# =============================================================================
main() {
    echo ""
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                   BS-Opt Deployment Script                    ║${NC}"
    echo -e "${BLUE}║          Quantitative Finance Research Platform               ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    log "Mode: ${MODE}"
    
    cd "$SCRIPT_DIR"
    
    check_requirements
    tune_kernel
    generate_secrets
    scaffold_directories
    pull_images
    build_images
    start_stack
    show_status
    
    log "🎉 Deployment complete!"
}

# Help
if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    echo "Usage: $0 [MODE]"
    echo ""
    echo "Modes:"
    echo "  production  - Full production stack (default)"
    echo "  research    - Includes JupyterHub for research"
    echo ""
    echo "Examples:"
    echo "  $0              # Deploy production stack"
    echo "  $0 research     # Deploy with Jupyter"
    echo "  sudo $0         # Deploy with kernel optimization"
    exit 0
fi

main "$@"
