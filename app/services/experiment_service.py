"""
experiment_service.py
─────────────────────
Core bandit logic. Manages:
  - Creating bandits from config (one per experiment)
  - Selecting the next arm
  - Updating reward
  - Persisting / reloading state from DB on startup
"""

from sqlalchemy.orm import Session

from app.bandits.factory import create_bandit
from app.bandits.base import BaseBandit
from app.config import N_ARMS, ALGORITHM, EPSILON
from app.db.crud import get_rewards, list_experiments


# ─── Global bandit registry ───────────────────────────────────────────────────
# Maps experiment name → its own BaseBandit instance.
# This means each experiment runs its own independent bandit.
_bandits: dict[str, BaseBandit] = {}


def _get_or_create_bandit(
    experiment: str = "default",
    algorithm: str = ALGORITHM,
    n_arms: int = N_ARMS,
    epsilon: float = EPSILON,
) -> BaseBandit:
    """
    Return the bandit for the given experiment, creating it if it doesn't
    exist yet. Falls back to the global config values when not specified.
    """
    if experiment not in _bandits:
        _bandits[experiment] = create_bandit(algorithm, n_arms, epsilon)
    return _bandits[experiment]


# ─── Public API ───────────────────────────────────────────────────────────────

def select_variant(experiment: str = "default") -> int:
    """Ask the bandit for the given experiment which arm to show next."""
    return _get_or_create_bandit(experiment).select_arm()


def update_reward(arm: int, reward: float, experiment: str = "default") -> None:
    """Inform the bandit of the observed reward so it can learn."""
    _get_or_create_bandit(experiment).update(arm, reward)


def get_bandit_state(experiment: str = "default") -> dict:
    """Return the current internal state of the bandit (for diagnostics)."""
    return _get_or_create_bandit(experiment).get_state()


def load_bandit_state_from_db(db: Session) -> None:
    """
    On server startup, replay all past reward events from the DB so that
    a restart doesn't erase what the bandits have already learned.

    - Loads the 'default' experiment always.
    - Also loads every named experiment stored in the experiments table,
      using its own saved algorithm / n_arms / epsilon configuration.
    """
    global _bandits
    _bandits = {}

    # ── 1. Load the default experiment ───────────────────────────────────────
    _bandits["default"] = create_bandit(ALGORITHM, N_ARMS, EPSILON)
    default_rewards = get_rewards(db, experiment="default")
    for r in default_rewards:
        _bandits["default"].update(r.arm, r.reward)
    print(f"[startup] 'default' → replayed {len(default_rewards)} events ({ALGORITHM})")

    # ── 2. Load every named experiment from the experiments table ─────────────
    try:
        experiments = list_experiments(db)
    except Exception:
        experiments = []

    for exp in experiments:
        if exp.name == "default":
            continue   # already loaded above
        bandit = create_bandit(exp.algorithm, exp.n_arms, exp.epsilon)
        rewards = get_rewards(db, experiment=exp.name)
        for r in rewards:
            bandit.update(r.arm, r.reward)
        _bandits[exp.name] = bandit
        print(f"[startup] '{exp.name}' → replayed {len(rewards)} events ({exp.algorithm})")