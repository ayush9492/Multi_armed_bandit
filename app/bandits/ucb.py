import math
from .base import BaseBandit


class UCB(BaseBandit):
    """
    Upper Confidence Bound (UCB1) algorithm.

    Each arm is scored as:
        UCB(arm) = avg_reward(arm) + sqrt(2 * ln(total_pulls) / pulls(arm))

    The bonus term decreases as an arm is pulled more, ensuring
    every arm gets fair exploration over time.

    Works with continuous rewards.
    """

    def __init__(self, n_arms: int):
        self.n_arms       = n_arms
        self.counts       = [0]   * n_arms   # pulls per arm
        self.values       = [0.0] * n_arms   # running average reward per arm
        self.total_counts = 0                 # total pulls across all arms

    def select_arm(self) -> int:
        # Always pull each arm at least once before applying UCB formula
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                return i

        ucb_values = []
        for i in range(self.n_arms):
            exploration_bonus = math.sqrt(
                2 * math.log(self.total_counts) / self.counts[i]
            )
            ucb_values.append(self.values[i] + exploration_bonus)

        return ucb_values.index(max(ucb_values))

    def update(self, arm: int, reward: float) -> None:
        self.total_counts  += 1
        self.counts[arm]   += 1
        n = self.counts[arm]
        self.values[arm]   += (reward - self.values[arm]) / n

    def get_state(self) -> dict:
        return {
            "algorithm":    "ucb",
            "n_arms":       self.n_arms,
            "counts":       self.counts,
            "values":       self.values,
            "total_counts": self.total_counts,
        }

    def load_state(self, state: dict) -> None:
        self.counts       = state["counts"]
        self.values       = state["values"]
        self.total_counts = state["total_counts"]