"""
experiment_service.py
─────────────────────
Per-experiment bandit registry.
Each experiment gets its own bandit instance keyed by name.
"""

from sqlalchemy.orm import Session

from app.bandits.factory import create_bandit
from app.bandits.base import BaseBandit
from app.config import N_ARMS, EPSILON
from app.db import crud

# Registry: experiment_name → BaseBandit instance
_bandits: dict[str, BaseBandit] = {}


def _get_or_create_bandit(experiment: str, db: Session = None) -> BaseBandit:
    """Return existing bandit for experiment, or create one from DB metadata."""
    if experiment in _bandits:
        return _bandits[experiment]

    # Look up algorithm from experiments table
    algorithm = "epsilon_greedy"  # fallback default
    n_arms = N_ARMS
    epsilon = EPSILON

    if db is not None:
        record = crud.get_experiment(db, experiment)
        if record:
            algorithm = record.algorithm
            n_arms    = record.n_arms
            epsilon   = record.epsilon if record.epsilon is not None else EPSILON

    bandit = create_bandit(algorithm, n_arms, epsilon)
    _bandits[experiment] = bandit
    return bandit


def select_variant(experiment: str = "default", db: Session = None) -> int:
    """Ask the bandit for this experiment which arm to show next."""
    bandit = _get_or_create_bandit(experiment, db)
    return bandit.select_arm()


def update_reward(arm: int, reward: float, experiment: str = "default", db: Session = None) -> None:
    """Inform the correct experiment's bandit of the observed reward."""
    bandit = _get_or_create_bandit(experiment, db)
    bandit.update(arm, reward)


def get_bandit_state(experiment: str = "default", db: Session = None) -> dict:
    """Return internal state of a specific experiment's bandit."""
    bandit = _get_or_create_bandit(experiment, db)
    return bandit.get_state()


def load_all_bandits_from_db(db: Session) -> None:
    """
    On startup: replay all experiments from DB so bandits restore their learned state.
    """
    experiments = crud.list_experiments(db)
    for exp in experiments:
        bandit = create_bandit(
            exp.algorithm,
            exp.n_arms,
            exp.epsilon if exp.epsilon is not None else EPSILON,
        )
        rewards = crud.get_rewards(db, experiment=exp.name)
        for r in rewards:
            bandit.update(r.arm, r.reward)
        _bandits[exp.name] = bandit
        print(f"[startup] Replayed {len(rewards)} rewards → '{exp.name}' ({exp.algorithm})")