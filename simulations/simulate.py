import argparse
import random
import sys
import os
import requests
import mlflow
mlflow.set_tracking_uri("sqlite:///D:/Reinforcement learning projects/Multi armed bandit/mlflow.db")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.bandits.factory import create_bandit
from app.utils.metrics import summary

# ─── Config ──────────────────────────────────────────────────────────────────

TRUE_WIN_RATES = [0.2, 0.5, 0.35]   # Arm 1 is the true best
BEST_ARM = TRUE_WIN_RATES.index(max(TRUE_WIN_RATES))
BEST_RATE = max(TRUE_WIN_RATES)
API_URL = "http://localhost:8000"


# ─── Mode 1: In-Memory Simulation (for MLflow + metrics) ─────────────────────

def run_simulation(algorithm: str, n_rounds: int, epsilon: float = 0.1):
    """
    Runs simulation entirely in memory.
    Used for MLflow tracking and algorithm comparison.
    Does NOT write to bandit.db.
    """
    n_arms = len(TRUE_WIN_RATES)
    bandit = create_bandit(algorithm, n_arms, epsilon=epsilon)

    history = []

    for _ in range(n_rounds):
        arm = bandit.select_arm()
        reward = 1 if random.random() < TRUE_WIN_RATES[arm] else 0
        bandit.update(arm, reward)
        history.append((arm, reward))

    stats = summary(history, n_arms, best_arm_rate=BEST_RATE)
    return stats, bandit


# ─── Mode 2: API Simulation (writes to DB → feeds Streamlit dashboard) ────────

def run_api_simulation(n_rounds: int = 1000):
    """
    Runs simulation by calling the live FastAPI server.
    Every reward is saved to bandit.db via the API.
    This is what populates the Streamlit dashboard.

    Requires: uvicorn app.main:app --reload  (running in another terminal)
    """
    print(f"🚀 Starting API simulation for {n_rounds} rounds...")
    print(f"   Talking to: {API_URL}")
    print(f"   True win rates: {TRUE_WIN_RATES}  (best = arm {BEST_ARM})\n")

    for i in range(n_rounds):
        # 1. Ask the bandit which arm to show
        try:
            response = requests.get(f"{API_URL}/select", timeout=5)
            arm = response.json()["variant"]
        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to API. Is uvicorn running?")
            print("   Run: uvicorn app.main:app --reload")
            sys.exit(1)

        # 2. Simulate reward based on true win rate
        reward = 1 if random.random() < TRUE_WIN_RATES[arm] else 0

        # 3. Send reward back to API → saves to bandit.db + updates bandit
        requests.post(
            f"{API_URL}/reward",
            json={"arm": arm, "reward": float(reward)},
            timeout=5,
        )

        # Progress update every 100 rounds
        if (i + 1) % 100 == 0:
            print(f"   Round {i+1}/{n_rounds} ✓")

    # 4. Fetch final stats from API
    stats = requests.get(f"{API_URL}/stats").json()
    print("\n📊 Final Stats from API:")
    for s in stats:
        print(f"   Arm {s['arm']}: pulls={s['pulls']}, mean_reward={s['mean_reward']}")
    print("\n✅ Done! Refresh your Streamlit dashboard at http://localhost:8501")


# ─── Mode 3: MLflow Comparison (in-memory, all 3 algorithms) ─────────────────

def run_mlflow_comparison(n_rounds: int = 1000):
    print("Running MLflow comparison across all algorithms...\n")
    mlflow.set_experiment("multi-armed-bandit")

    for algo in ["thompson", "epsilon_greedy", "ucb"]:
        with mlflow.start_run(run_name=algo):
            mlflow.log_param("algorithm", algo)
            mlflow.log_param("n_arms", len(TRUE_WIN_RATES))
            mlflow.log_param("rounds", n_rounds)
            mlflow.log_param("true_win_rates", str(TRUE_WIN_RATES))

            stats, _ = run_simulation(algo, n_rounds=n_rounds)

            mlflow.log_metric("final_regret", stats["final_cumulative_regret"])
            mlflow.log_metric("best_arm_share", max(stats["traffic_share_per_arm"]))
            mlflow.log_metric("mean_reward", stats["overall_mean_reward"])
            mlflow.log_metric("total_reward", stats["total_reward"])

            print(f"Done: {algo}")
            print(f"   Regret      : {stats['final_cumulative_regret']}")
            print(f"   Best arm %  : {max(stats['traffic_share_per_arm'])*100:.1f}%")
            print(f"   Mean reward : {stats['overall_mean_reward']}\n")

    print("Done! Launch MLflow UI with: mlflow ui")
    print("Then open: http://localhost:5000")
# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Armed Bandit Simulation")
    parser.add_argument(
        "--mode",
        default="api",
        choices=["api", "mlflow", "single"],
        help=(
            "api    → call live API, populate dashboard (default)\n"
            "mlflow → compare all algorithms, log to MLflow\n"
            "single → run one algorithm in memory, print results"
        ),
    )
    parser.add_argument("--algorithm", default="thompson",
                        choices=["thompson", "epsilon_greedy", "ucb"])
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--epsilon", type=float, default=0.1)
    args = parser.parse_args()

    if args.mode == "api":
        run_api_simulation(n_rounds=args.rounds)

    elif args.mode == "mlflow":
        run_mlflow_comparison(n_rounds=args.rounds)

    elif args.mode == "single":
        stats, _ = run_simulation(args.algorithm, args.rounds, args.epsilon)
        print(f"\n Algorithm : {args.algorithm}")
        print(f" Rounds    : {args.rounds}")
        print(f" Pulls/arm : {stats['pulls_per_arm']}")
        print(f" Traffic % : {stats['traffic_share_per_arm']}")
        print(f" Regret    : {stats['final_cumulative_regret']}")
        print(f" Mean rew  : {stats['overall_mean_reward']}")