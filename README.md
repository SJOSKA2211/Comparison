# BS-Opt: Black-Scholes Optimization & Research Platform

A next-generation quantitative finance platform bridging **Frontier Markets (NSE Kenya)** and **Global High-Frequency Trading**.

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Nginx      │────▶│  FastAPI     │────▶│  PostgreSQL     │
│  (C100k)    │     │  Gateway     │     │  + TimescaleDB  │
└─────────────┘     └──────────────┘     └─────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Kafka     │    │   Redis     │    │  Ray/MLflow │
│   (KRaft)   │    │   Cluster   │    │  ML Cluster │
└─────────────┘    └─────────────┘    └─────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose v2
- 8GB+ RAM (16GB recommended)

### Deploy

```bash
# Clone and enter directory
cd comparison

# One-click deploy (generates secrets, tunes kernel, starts stack)
./deploy.sh

# Or with research mode (includes JupyterHub)
./deploy.sh research
```

### Access Services

| Service | URL |
|---------|-----|
| API Gateway | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Ray Dashboard | http://localhost:8265 |
| MLflow | http://localhost:5000 |
| Jupyter (research) | http://localhost:8888 |

## 🔐 Authentication

Native PostgreSQL authentication with multiple providers:

- **Email/Password**: Local registration with email verification
- **Google OAuth**: `GET /auth/google`
- **GitHub OAuth**: `GET /auth/github`

Configure OAuth in `.env`:
```bash
GOOGLE_CLIENT_ID=your-client-id
GOOGLE_CLIENT_SECRET=your-secret
GITHUB_CLIENT_ID=your-client-id
GITHUB_CLIENT_SECRET=your-secret
```

## 📊 Pricing Engine

### Endpoints

```bash
# Black-Scholes pricing with Greeks
POST /pricing/black-scholes
{
  "spot": 100,
  "strike": 100,
  "rate": 0.05,
  "volatility": 0.2,
  "time_to_maturity": 1.0,
  "option_type": "call"
}

# Compare all methods (researcher role required)
POST /pricing/compare
```

### Numerical Methods

| Method | Implementation | Use Case |
|--------|---------------|----------|
| Analytical | Black-Scholes formula | Benchmarking |
| FDM | Crank-Nicolson | Stable, 2nd-order accurate |
| Monte Carlo | Antithetic variance reduction | Path-dependent options |
| Trinomial | Richardson extrapolation | American options |

## 🐳 Anti-Freeze Docker Architecture

This project uses **"Remote Build, Local Run"** pattern:

1. **CI builds images** in GitHub Actions
2. **Local pulls** pre-built images from GHCR
3. **Resource limits** prevent system freeze

```yaml
# docker-compose.prod.yml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
```

### Build Locally (Throttled)

```bash
export COMPOSE_PARALLEL_LIMIT=1
docker compose -f docker-compose.prod.yml build
```

## 📁 Project Structure

```
comparison/
├── docker/
│   ├── Dockerfile.api         # API Gateway (multi-stage)
│   └── Dockerfile.ml          # ML Worker (heavy deps)
├── src/
│   ├── api/main.py            # FastAPI application
│   ├── pricing/               # Numerical methods
│   ├── ml/                    # Ray/PyTorch workers
│   └── data/                  # Market data router
├── db/init.sql                # PostgreSQL + TimescaleDB schema
├── config/nginx.conf          # C100k optimized
├── docker-compose.prod.yml    # Production stack
├── deploy.sh                  # One-click deploy
└── .github/workflows/         # CI/CD pipeline
```

## 🧪 Development

```bash
# Install dependencies locally
pip install -r requirements/base.txt

# Run API in dev mode
python -m src.api.main

# Run tests
pytest tests/ -v
```

## 📈 Research Mode

Enable JupyterHub for academic research:

```bash
./deploy.sh research
```

Access notebooks at http://localhost:8888 with direct database connectivity.

## 🔧 Configuration

See [.env.example](.env.example) for all configuration options.

## 📄 License

MIT
