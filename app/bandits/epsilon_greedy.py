import random
from .base import BaseBandit


class EpsilonGreedy(BaseBandit):
    """
    Epsilon-Greedy algorithm.

    With probability epsilon, pick a random arm (explore).
    Otherwise, pick the arm with the highest observed average reward (exploit).

    Works with continuous rewards (not just 0/1).
    """

    def __init__(self, n_arms: int, epsilon: float = 0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = [0] * n_arms       # number of times each arm was pulled
        self.values = [0.0] * n_arms     # running average reward per arm

    def select_arm(self) -> int:
        if random.random() < self.epsilon:
            # Explore: pick any arm at random
            return random.randint(0, self.n_arms - 1)
        # Exploit: pick the arm with the best known average reward
        return self.values.index(max(self.values))

    def update(self, arm: int, reward: float) -> None:
        """Incremental mean update: avoids storing all rewards in memory."""
        self.counts[arm] += 1
        n = self.counts[arm]
        # new_avg = old_avg + (reward - old_avg) / n
        self.values[arm] += (reward - self.values[arm]) / n

    def get_state(self) -> dict:
        return {
            "algorithm": "epsilon_greedy",
            "n_arms": self.n_arms,
            "epsilon": self.epsilon,
            "counts": self.counts,
            "values": self.values,
        }

    def load_state(self, state: dict) -> None:
        self.counts = state["counts"]
        self.values = state["values"]