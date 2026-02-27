class BaseBandit:
    """Abstract base class for all bandit algorithms."""

    def select_arm(self) -> int:
        """Return the index of the arm to pull next."""
        raise NotImplementedError

    def update(self, arm: int, reward: float) -> None:
        """Update internal state after observing a reward for the given arm."""
        raise NotImplementedError

    def get_state(self) -> dict:
        """Return serializable state for persistence/inspection."""
        raise NotImplementedError

    def load_state(self, state: dict) -> None:
        """Restore bandit state from a previously saved dict."""
        raise NotImplementedError