"""
metrics.py
──────────
Utility functions to compute standard bandit evaluation metrics.

All functions accept a list of (arm, reward) tuples representing
the history of interactions in chronological order.
"""

from typing import List, Tuple


def cumulative_reward(history: List[Tuple[int, float]]) -> List[float]:
    """
    Running total of rewards over time.
    Useful for seeing overall learning progress.
    """
    total = 0.0
    result = []
    for _, reward in history:
        total += reward
        result.append(total)
    return result


def cumulative_regret(
    history: List[Tuple[int, float]],
    best_arm_rate: float,
) -> List[float]:
    """
    Regret = what you *could* have earned (always picking the best arm)
             minus what you *actually* earned.

    Lower regret = better algorithm.

    Args:
        history: List of (arm, reward) tuples in order.
        best_arm_rate: The true win rate of the optimal arm (known in simulation).
    """
    regret = 0.0
    result = []
    for _, reward in history:
        regret += best_arm_rate - reward
        result.append(regret)
    return result


def arm_pull_counts(history: List[Tuple[int, float]], n_arms: int) -> List[int]:
    """
    Number of times each arm was pulled.
    Useful for checking whether the algorithm converged to the best arm.
    """
    counts = [0] * n_arms
    for arm, _ in history:
        counts[arm] += 1
    return counts


def arm_mean_rewards(history: List[Tuple[int, float]], n_arms: int) -> List[float]:
    """
    Average reward observed for each arm.
    """
    totals = [0.0] * n_arms
    counts = [0] * n_arms
    for arm, reward in history:
        totals[arm] += reward
        counts[arm] += 1
    return [
        round(totals[i] / counts[i], 4) if counts[i] > 0 else 0.0
        for i in range(n_arms)
    ]


def win_rate_per_arm(history: List[Tuple[int, float]], n_arms: int) -> List[float]:
    """
    Fraction of interactions where the arm returned reward=1.
    (Only meaningful for binary rewards.)
    """
    return arm_mean_rewards(history, n_arms)


def traffic_share(history: List[Tuple[int, float]], n_arms: int) -> List[float]:
    """
    Fraction of total pulls assigned to each arm.
    A well-converged bandit should assign ~80–95% to the best arm.
    """
    counts = arm_pull_counts(history, n_arms)
    total = sum(counts)
    if total == 0:
        return [0.0] * n_arms
    return [round(c / total, 4) for c in counts]


def summary(
    history: List[Tuple[int, float]],
    n_arms: int,
    best_arm_rate: float | None = None,
) -> dict:
    """
    Convenience function: compute all metrics in one call.

    Returns a dict ready to be serialised to JSON or printed.
    """
    total_pulls = len(history)
    total_reward = sum(r for _, r in history)

    result = {
        "total_pulls": total_pulls,
        "total_reward": round(total_reward, 4),
        "overall_mean_reward": round(total_reward / total_pulls, 4) if total_pulls else 0,
        "pulls_per_arm": arm_pull_counts(history, n_arms),
        "mean_reward_per_arm": arm_mean_rewards(history, n_arms),
        "traffic_share_per_arm": traffic_share(history, n_arms),
    }

    if best_arm_rate is not None:
        regret = cumulative_regret(history, best_arm_rate)
        result["final_cumulative_regret"] = round(regret[-1], 4) if regret else 0

    return result