from .base import BaseBandit
from .thompson_sampling import ThompsonSampling
from .epsilon_greedy import EpsilonGreedy
from .ucb import UCB


def create_bandit(algorithm: str, n_arms: int, epsilon: float = 0.1) -> BaseBandit:
    """
    Factory function: create a bandit by algorithm name.

    Args:
        algorithm: One of 'thompson', 'epsilon_greedy', 'ucb'
        n_arms:    Number of arms (variants)
        epsilon:   Exploration rate (only used by epsilon_greedy)

    Returns:
        An instance of the requested bandit algorithm.

    Raises:
        ValueError if algorithm name is unknown.
    """
    algorithm = algorithm.lower().strip()

    if algorithm == "thompson":
        return ThompsonSampling(n_arms)
    elif algorithm == "epsilon_greedy":
        return EpsilonGreedy(n_arms, epsilon=epsilon)
    elif algorithm == "ucb":
        return UCB(n_arms)
    else:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Choose from: thompson, epsilon_greedy, ucb"
        )