import numpy as np
from .base import BaseBandit


class ThompsonSampling(BaseBandit):
    """
    Thompson Sampling algorithm (Beta-Bernoulli model).

    Each arm maintains a Beta(alpha, beta) distribution representing
    our belief about its true win rate.

    At each step:
      1. Sample a value from each arm's Beta distribution.
      2. Pull the arm with the highest sample.
      3. Update that arm's distribution based on the reward.

    Over time, the best arm's distribution concentrates near a high
    value and gets selected most often.

    NOTE: Reward is binarised at 0.5 threshold so this works correctly
    with both binary (0/1) and continuous ([0.0, 1.0]) reward inputs.
    """

    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        # Start with Beta(1, 1) = uniform prior (no prior knowledge)
        self.alpha = [1] * n_arms   # successes + 1
        self.beta  = [1] * n_arms   # failures  + 1

    def select_arm(self) -> int:
        samples = [
            np.random.beta(self.alpha[i], self.beta[i])
            for i in range(self.n_arms)
        ]
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        """
        Update Beta distribution.

        Reward is binarised at 0.5 so that:
          - reward >= 0.5 (success): increment alpha
          - reward <  0.5 (failure): increment beta

        This makes the algorithm robust to continuous rewards in [0, 1],
        not just strict 0/1 binary values.
        """
        if reward >= 0.5:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    def get_state(self) -> dict:
        return {
            "algorithm": "thompson",
            "n_arms": self.n_arms,
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def load_state(self, state: dict) -> None:
        self.alpha = state["alpha"]
        self.beta  = state["beta"]