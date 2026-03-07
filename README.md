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
│   └── streamlit_app.py           # Real-time monitoring dashboard
├── simulations/
│   └── simulate.py                # CLI simulation + MLflow comparison
├── tests/
│   ├── test_api.py                # Integration tests (FastAPI TestClient)
│   └── test_bandits.py            # Unit tests (all 3 algorithms + factory)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .env                           # Config (not committed to git)
├── .gitignore
├── .dockerignore
└── requirements.txt
```

---

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API
uvicorn app.main:app --reload

# Start the dashboard (in another terminal)
streamlit run dashboard/streamlit_app.py

# Run a simulation (populates the dashboard)
python simulations/simulate.py --mode api --rounds 1000
```

### Docker

```bash
docker-compose -f docker/docker-compose.yml up --build
```

The API will be available at `http://localhost:8000` and the dashboard at `http://localhost:8501`.

---

## 📡 API Endpoints

| Method | Endpoint       | Description                       |
|--------|----------------|-----------------------------------|
| GET    | `/`            | Health check                      |
| GET    | `/select`      | Get the next arm to show          |
| POST   | `/reward`      | Submit reward for an arm          |
| GET    | `/stats`       | Get per-arm statistics            |
| GET    | `/state`       | Inspect internal bandit state     |
| GET    | `/experiments` | List all experiments              |
| POST   | `/experiments` | Create a new experiment           |

All endpoints accept an optional `?experiment=<name>` query parameter.  
Each experiment runs its own **independent bandit instance**.

### Example Usage

```bash
# Get which variant to show
curl http://localhost:8000/select

# Submit a reward (1 = success, 0 = failure)
curl -X POST http://localhost:8000/reward \
  -H "Content-Type: application/json" \
  -d '{"arm": 0, "reward": 1}'

# Get stats
curl http://localhost:8000/stats

# Create a named experiment with UCB and 4 arms
curl -X POST http://localhost:8000/experiments \
  -H "Content-Type: application/json" \
  -d '{"name": "homepage_cta", "algorithm": "ucb", "n_arms": 4}'
```

---

## ⚙️ Configuration (`.env`)

```env
DATABASE_URL=sqlite:///./bandit.db
N_ARMS=3
ALGORITHM=thompson      # Options: thompson, epsilon_greedy, ucb
EPSILON=0.1             # Only for epsilon_greedy
DEBUG=True
```

---

## 🧠 Algorithms

| Algorithm        | Description                              | Best For                                  |
|------------------|------------------------------------------|-------------------------------------------|
| `thompson`       | Beta-distribution sampling               | Best overall — handles uncertainty naturally |
| `epsilon_greedy` | ε% random exploration, rest exploit best | Simple, predictable, easy to explain      |
| `ucb`            | Optimistic upper confidence bound (UCB1) | Theoretical regret guarantees             |

All three algorithms support:
- Continuous rewards in `[0.0, 1.0]` (not just binary 0/1)
- `get_state()` / `load_state()` for full state serialisation
- Automatic state replay from DB on server restart — no knowledge is lost on reboot

---

## 📊 Dashboard

🚀 **Live:** [multiarmedbandit.streamlit.app](https://multiarmedbandit.streamlit.app/)

The Streamlit dashboard shows live data from `bandit.db`:

- Per-arm mean reward (bar chart)
- Traffic share per arm (pie chart)
- Cumulative reward over time
- **Cumulative regret curve** — lower = better algorithm
- Rolling mean reward (window = 50)

Access locally at: `http://localhost:8501`

---

## 📈 MLflow Experiment Tracking

Compare all 3 algorithms and log metrics automatically:

```bash
python simulations/simulate.py --mode mlflow --rounds 2000
mlflow ui
# Open: http://localhost:5000
```

Tracked metrics per run: `final_regret`, `best_arm_share`, `mean_reward`, `total_reward`

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Test coverage includes:
- All 3 algorithm unit tests (select, update, state roundtrip, convergence)
- Factory function (valid + invalid algorithm names)
- All API endpoints (health, select, reward, stats, state, experiments)
- Per-experiment arm validation (rejects out-of-range arms correctly)
- Duplicate experiment 409 conflict handling