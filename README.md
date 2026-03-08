# 🎰 Multi-Armed Bandit API

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)
![MLflow](https://img.shields.io/badge/MLflow-tracked-orange?logo=mlflow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A **production-ready A/B testing system** using Multi-Armed Bandit algorithms.  
Instead of splitting traffic 50/50, the system **learns which variant performs best** and automatically shifts traffic toward it — minimising wasted exposure to underperforming variants.

🚀 **Live Demo:** [multiarmedbandit.streamlit.app](https://multiarmedbandit.streamlit.app/)

---

## 💡 Why Multi-Armed Bandit over A/B Testing?

Traditional A/B testing splits traffic equally until a winner is declared — wasting 50% of traffic on the losing variant the entire time.

A Multi-Armed Bandit solves this by **learning on the fly**:

| | Traditional A/B | Multi-Armed Bandit |
|---|---|---|
| Traffic split | Fixed 50/50 | Dynamic, shifts toward winner |
| Regret | High — loser runs full duration | Low — loser phased out early |
| Real-world use | Marketing, product, ads | Same, but smarter |

---

## 📐 Architecture

```
┌─────────────────────────────────────┐
│         Client / Simulation         │
└────────────────┬────────────────────┘
                 │ HTTP
      ┌──────────▼──────────┐
      │   FastAPI  :8000    │
      │     routes.py       │
      └──┬───────────────┬──┘
         │               │
┌────────▼──────┐  ┌─────▼──────────┐
│  Experiment   │  │    Reward      │
│  Service      │  │    Service     │
│ (per-exp      │  │ (validate +    │
│  bandit reg.) │  │  persist)      │
└────────┬──────┘  └─────┬──────────┘
         │               │
┌────────▼───────────────▼──────────┐
│           Bandit Layer            │
│  Thompson | UCB | EpsilonGreedy   │
└────────────────┬──────────────────┘
                 │
┌────────────────▼──────────────────┐
│   SQLite via SQLAlchemy (ORM)     │
│  rewards table | experiments table│
└────────────────┬──────────────────┘
                 │
┌────────────────▼──────────────────┐
│  Streamlit Dashboard   :8501      │
│  MLflow Tracking       :5000      │
└───────────────────────────────────┘
```

---

## 📁 Project Structure

```
├── app/
│   ├── api/
│   │   ├── routes.py              # FastAPI endpoints
│   │   └── schemas.py             # Pydantic request/response models
│   ├── bandits/
│   │   ├── base.py                # Abstract base class (ABC)
│   │   ├── thompson_sampling.py   # Beta-Bernoulli Thompson Sampling
│   │   ├── epsilon_greedy.py      # ε-Greedy with incremental mean
│   │   ├── ucb.py                 # UCB1 algorithm
│   │   └── factory.py             # create_bandit() factory function
│   ├── db/
│   │   ├── database.py            # SQLAlchemy engine + session
│   │   ├── models.py              # Reward + Experiment ORM models
│   │   └── crud.py                # DB read/write operations
│   ├── services/
│   │   ├── experiment_service.py  # Per-experiment bandit registry
│   │   └── reward_service.py      # Validation + persistence logic
│   ├── utils/
│   │   └── metrics.py             # Regret, cumulative reward, traffic share
│   ├── config.py                  # .env loader
│   └── main.py                    # FastAPI app entrypoint
├── dashboard/
│   └── streamlit_app.py           # Real-time monitoring + algorithm comparison
├── simulations/
│   └── simulate.py                # CLI simulation + MLflow comparison
├── tests/
│   ├── test_api.py                # Integration tests (FastAPI TestClient)
│   └── test_bandits.py            # Unit tests (all 3 algorithms + factory)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── run_experiments.py             # Create + simulate named experiments with MLflow logging
├── .env                           # Config (not committed to git)
├── .gitignore
├── .dockerignore
├── .python-version                # Pins Python 3.11 for Streamlit Cloud
└── requirements.txt
```

---

## 🚀 Quick Start

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the API
uvicorn app.main:app --reload --port 8000

# 3. Start the dashboard (new terminal)
streamlit run dashboard/streamlit_app.py

# 4. Run experiments — creates 3 named experiments and simulates 1000 pulls each
python run_experiments.py

# 5. View MLflow results (new terminal)
mlflow ui --port 5000
# Open: http://localhost:5000
```

### Docker

```bash
docker-compose -f docker/docker-compose.yml up --build
```

The API will be available at `http://localhost:8000` and the dashboard at `http://localhost:8501`.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/select?experiment=<n>` | Get the next arm to show |
| POST | `/reward` | Submit reward for an arm |
| GET | `/stats?experiment=<n>` | Get per-arm statistics |
| GET | `/state?experiment=<n>` | Inspect internal bandit state |
| GET | `/experiments` | List all experiments |
| POST | `/experiments` | Create a new experiment |

Each experiment runs its own **independent bandit instance** — passing `?experiment=<name>` routes to the correct algorithm automatically.

### Example Usage

```bash
# Create a named experiment with Thompson Sampling
curl -X POST http://localhost:8000/experiments \
  -H "Content-Type: application/json" \
  -d '{"name": "thompson_sampling", "algorithm": "thompson", "n_arms": 3}'

# Get which variant to show for that experiment
curl http://localhost:8000/select?experiment=thompson_sampling

# Submit a reward
curl -X POST http://localhost:8000/reward \
  -H "Content-Type: application/json" \
  -d '{"arm": 1, "reward": 1.0, "experiment": "thompson_sampling"}'

# Get per-arm stats
curl http://localhost:8000/stats?experiment=thompson_sampling

# Create a UCB experiment with 4 arms
curl -X POST http://localhost:8000/experiments \
  -H "Content-Type: application/json" \
  -d '{"name": "homepage_cta", "algorithm": "ucb", "n_arms": 4}'
```

---

## ⚙️ Configuration (`.env`)

```env
DATABASE_URL=sqlite:///./bandit.db
N_ARMS=3
ALGORITHM=epsilon_greedy    # Default: thompson | epsilon_greedy | ucb
EPSILON=0.1                 # Only used by epsilon_greedy
DEBUG=True
```

---

## 🧠 Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| `thompson` | Beta-distribution sampling | Best overall — handles uncertainty naturally |
| `epsilon_greedy` | ε% random exploration, rest exploit best | Simple, predictable, easy to explain |
| `ucb` | Optimistic upper confidence bound (UCB1) | Theoretical regret guarantees |

All three algorithms support:
- Continuous rewards in `[0.0, 1.0]` (not just binary 0/1)
- `get_state()` for full state inspection
- Automatic state replay from DB on server restart — no knowledge is lost on reboot
- Per-experiment isolation — each named experiment runs its own independent bandit

---

## 📊 Dashboard

🚀 **Live:** [multiarmedbandit.streamlit.app](https://multiarmedbandit.streamlit.app/)

The Streamlit dashboard has two modes:

**Live Mode** (when running locally with API + DB):
- Auto-refreshes every 5 seconds from `bandit.db`

**Demo Mode** (Streamlit Cloud / no DB):
- Activates automatically when no database is found
- Shows a realistic simulation across all 3 algorithms

### Dashboard Tabs

**⚔️ Algorithm Comparison tab** — side-by-side across all experiments:
- Summary table (pulls, mean reward, best arm, traffic %)
- Cumulative reward curves
- Cumulative regret curves (lower = better algorithm)
- Rolling mean reward (window = 50)
- Traffic share pie charts per algorithm
- Grouped mean reward per arm bar chart

**Per-experiment tabs** — individual deep-dives:
- KPI row (total pulls, total reward, mean reward, leading arm)
- Mean reward per arm, traffic share, cumulative reward, rolling mean, regret curve

Access locally at `http://localhost:8501`

---

## 🧪 Running Experiments

```bash
python run_experiments.py
```

This script:
1. Creates 3 named experiments (`thompson_sampling`, `ucb`, `epsilon_greedy`) via the API
2. Simulates 1,000 pulls each against true win rates `[0.30, 0.55, 0.15]`
3. Skips experiments that already have enough data — safe to re-run
4. Logs all metrics to MLflow automatically

Change `N_PULLS = 1000` at the top to simulate more pulls.

---

## 📈 MLflow Experiment Tracking

```bash
# Logs automatically when you run:
python run_experiments.py

# Then open the UI:
mlflow ui --port 5000
# Open: http://localhost:5000
```

**Tracked metrics per run:**

| Metric | Description |
|--------|-------------|
| `mean_reward` | Overall efficiency |
| `total_reward` | Total reward collected |
| `cumulative_regret` | Reward lost vs optimal policy |
| `best_arm_traffic_pct` | % of traffic routed to best arm |
| `arm_0/1/2_pulls` | Per-arm pull counts |
| `arm_0/1/2_mean_reward` | Per-arm estimated reward |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Test coverage includes:
- All 3 algorithm unit tests (select, update, state roundtrip, convergence)
- Factory function (valid + invalid algorithm names)
- All API endpoints (health, select, reward, stats, state, experiments)
- Per-experiment arm validation (rejects out-of-range arms)
- Duplicate experiment 409 conflict handling

---

## 🐳 Docker

```bash
docker-compose -f docker/docker-compose.yml up --build

# Services:
# API        → http://localhost:8000
# Dashboard  → http://localhost:8501
# MLflow     → http://localhost:5000
```

The `docker-compose.yml` uses health checks so the dashboard only starts after the API is healthy.