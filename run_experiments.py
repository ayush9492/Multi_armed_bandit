"""
run_experiments.py
──────────────────
Creates 3 experiments and simulates pulls — skips if already has data.

Usage:
    python run_experiments.py
"""

import requests
import random
import mlflow
import os

BASE_URL = "http://localhost:8000"
TRUE_PROBS = [0.3, 0.55, 0.15]
N_PULLS = 1000

mlflow_db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
mlflow.set_experiment("multi-armed-bandit")


def create_experiment(name, algorithm, n_arms=3, epsilon=None):
    payload = {"name": name, "algorithm": algorithm, "n_arms": n_arms}
    if epsilon is not None:
        payload["epsilon"] = epsilon
    r = requests.post(f"{BASE_URL}/experiments", json=payload)
    if r.status_code in (200, 201):
        print(f"✅ Created: {name} ({algorithm})")
    elif r.status_code == 409:
        print(f"ℹ️  Already exists: {name} — skipping creation")
    else:
        print(f"⚠️  Error: {r.status_code} {r.text}")


def get_existing_pulls(experiment_name):
    """Check how many pulls already exist for this experiment."""
    try:
        r = requests.get(f"{BASE_URL}/stats", params={"experiment": experiment_name})
        stats = r.json()
        return sum(s["pulls"] for s in stats)
    except Exception:
        return 0


def simulate(experiment_name, algorithm, n_pulls, epsilon=None):
    existing = get_existing_pulls(experiment_name)
    if existing >= n_pulls:
        print(f"⏭️  Skipping '{experiment_name}' — already has {existing} pulls")
        return

    remaining = n_pulls - existing
    print(f"\n🎰 Simulating {remaining} pulls for '{experiment_name}' (existing: {existing}) ...")

    arms_history, rewards_history = [], []

    for i in range(remaining):
        r = requests.get(f"{BASE_URL}/select", params={"experiment": experiment_name})
        arm = r.json()["variant"]
        reward = 1.0 if random.random() < TRUE_PROBS[arm] else 0.0
        arms_history.append(arm)
        rewards_history.append(reward)

        requests.post(f"{BASE_URL}/reward", json={
            "arm": arm, "reward": reward, "experiment": experiment_name
        })

        if (i + 1) % 200 == 0:
            print(f"   Pull {i+1}/{remaining} — mean: {sum(rewards_history)/len(rewards_history):.3f}")

    total_pulls  = existing + len(rewards_history)
    mean_reward  = sum(rewards_history) / len(rewards_history)
    best_arm     = max(set(arms_history), key=arms_history.count)
    best_arm_pct = arms_history.count(best_arm) / len(arms_history) * 100
    regret       = len(rewards_history) * max(TRUE_PROBS) - sum(rewards_history)

    print(f"✅ Done. Mean: {mean_reward:.3f} | Regret: {regret:.1f} | Best arm: Arm {best_arm} ({best_arm_pct:.1f}%)")

    with mlflow.start_run(run_name=experiment_name):
        mlflow.log_param("algorithm", algorithm)
        mlflow.log_param("n_arms", 3)
        mlflow.log_param("n_pulls", total_pulls)
        if epsilon is not None:
            mlflow.log_param("epsilon", epsilon)
        mlflow.log_metric("mean_reward", round(mean_reward, 4))
        mlflow.log_metric("total_reward", sum(rewards_history))
        mlflow.log_metric("cumulative_regret", round(regret, 2))
        mlflow.log_metric("best_arm_traffic_pct", round(best_arm_pct, 2))
        for arm_id in range(3):
            arm_pulls = arms_history.count(arm_id)
            arm_rew   = sum(r for a, r in zip(arms_history, rewards_history) if a == arm_id)
            mlflow.log_metric(f"arm_{arm_id}_pulls", arm_pulls)
            mlflow.log_metric(f"arm_{arm_id}_mean_reward", round(arm_rew / arm_pulls, 4) if arm_pulls else 0)
        print(f"   📊 Logged to MLflow")


if __name__ == "__main__":
    create_experiment("thompson_sampling", "thompson", n_arms=3)
    create_experiment("ucb", "ucb", n_arms=3)
    create_experiment("epsilon_greedy", "epsilon_greedy", n_arms=3, epsilon=0.1)

    simulate("thompson_sampling", "thompson", N_PULLS)
    simulate("ucb", "ucb", N_PULLS)
    simulate("epsilon_greedy", "epsilon_greedy", N_PULLS, epsilon=0.1)

    print("\n🎉 All done!")
    print("   → Streamlit: http://localhost:8501")
    print("   → MLflow:    http://localhost:5000")