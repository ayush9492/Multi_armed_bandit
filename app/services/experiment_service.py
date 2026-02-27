"""
experiment_service.py
─────────────────────
Core bandit logic. Manages:
  - Creating the bandit from config
  - Selecting the next arm
  - Updating reward
  - Persisting / reloading state from DB on startup
"""

from sqlalchemy.orm import Session

from app.bandits.factory import create_bandit
from app.bandits.base import BaseBandit
from app.config import N_ARMS, ALGORITHM, EPSILON
from app.db.crud import get_rewards


# Global in-memory bandit instance (shared across requests)
bandit: BaseBandit = create_bandit(ALGORITHM, N_ARMS, EPSILON)


def select_variant() -> int:
    """Ask the bandit which arm to show next."""
    return bandit.select_arm()


def update_reward(arm: int, reward: float) -> None:
    """Inform the bandit of the observed reward so it can learn."""
    bandit.update(arm, reward)


def get_bandit_state() -> dict:
    """Return the current internal state of the bandit (for diagnostics)."""
    return bandit.get_state()


def load_bandit_state_from_db(db: Session, experiment: str = "default") -> None:
    """
    Replay all past reward events from the DB to restore bandit knowledge.

    Called once at server startup so that a restart doesn't erase
    everything the bandit has learned from real traffic.
    """
    global bandit
    # Re-create a fresh bandit to start from clean state
    bandit = create_bandit(ALGORITHM, N_ARMS, EPSILON)

    rewards = get_rewards(db, experiment=experiment)
    for r in rewards:
        bandit.update(r.arm, r.reward)

    print(f"[startup] Replayed {len(rewards)} reward events into bandit ({ALGORITHM})")