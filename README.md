# 🎰 Multi-Armed Bandit API

A production-ready A/B testing system using Multi-Armed Bandit algorithms. Instead of splitting traffic 50/50, the system **learns which variant performs best** and automatically shifts traffic toward it.

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

# Run a simulation
python simulations/simulate.py
```

### Docker

```bash
docker-compose -f docker/docker-compose.yml up --build
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/select` | Get the next arm to show |
| POST | `/reward` | Submit reward for an arm |
| GET | `/stats` | Get per-arm statistics |
| GET | `/experiments` | List all experiments |
| POST | `/experiments` | Create a new experiment |

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

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| `thompson` | Beta-distribution sampling | Best overall, handles uncertainty naturally |
| `epsilon_greedy` | ε% random, rest exploit best | Simple, predictable |
| `ucb` | Optimistic upper confidence bound | Theoretical guarantees |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📊 Dashboard

The Streamlit dashboard shows:
- Per-arm mean reward
- Pull counts per arm
- Cumulative reward over time
- Regret curve

Access at: `http://localhost:8501`

deployed link: https://multiarmedbandit.streamlit.app/
