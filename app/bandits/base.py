from abc import ABC, abstractmethod


class BaseBandit(ABC):
    """Abstract base class for all bandit algorithms."""

    @abstractmethod
    def select_arm(self) -> int:
        """Return the index of the arm to pull next."""
        ...

    @abstractmethod
    def update(self, arm: int, reward: float) -> None:
        """Update internal state after observing a reward for the given arm."""
        ...

    @abstractmethod
    def get_state(self) -> dict:
        """Return serializable state for persistence/inspection."""
        ...

    @abstractmethod
    def load_state(self, state: dict) -> None:
        """Restore bandit state from a previously saved dict."""
        ...