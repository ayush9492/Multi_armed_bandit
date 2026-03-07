"""
reward_service.py
─────────────────
Business logic layer for processing reward submissions.

Responsibilities:
  - Validate the arm index against the correct experiment's n_arms
  - Validate reward values are within [0.0, 1.0]
  - Save to DB
  - Update the correct in-memory bandit for the experiment
"""

from sqlalchemy.orm import Session

from app.config import N_ARMS
from app.db import crud
from app.services.experiment_service import update_reward


class RewardValidationError(Exception):
    pass


def process_reward(
    db: Session,
    arm: int,
    reward: float,
    experiment: str = "default",
) -> dict:
    """
    Validate, persist, and feed a reward to the correct experiment's bandit.

    Args:
        db:         Active database session.
        arm:        Which arm (variant) was shown to the user.
        reward:     Observed outcome. Expected to be in [0.0, 1.0].
        experiment: Experiment name (each experiment has its own bandit).

    Returns:
        A dict summarising what was stored.

    Raises:
        RewardValidationError if the arm index is out of range or reward is invalid.
    """
    # ── 1. Resolve the correct n_arms for this experiment ────────────────────
    # Named experiments may have a different arm count than the global default.
    n_arms = N_ARMS
    if experiment != "default":
        exp_record = crud.get_experiment(db, experiment)
        if exp_record is not None:
            n_arms = exp_record.n_arms

    # ── 2. Validate arm index ─────────────────────────────────────────────────
    if arm < 0 or arm >= n_arms:
        raise RewardValidationError(
            f"Arm index {arm} is out of range for experiment '{experiment}'. "
            f"Valid range: 0 to {n_arms - 1}."
        )

    # ── 3. Validate reward value ──────────────────────────────────────────────
    if not (0.0 <= reward <= 1.0):
        raise RewardValidationError(
            f"Reward value {reward} is outside the expected range [0.0, 1.0]."
        )

    # ── 4. Persist to database ────────────────────────────────────────────────
    record = crud.add_reward(db, arm=arm, reward=reward, experiment=experiment)

    # ── 5. Update the correct in-memory bandit so it learns immediately ───────
    update_reward(arm, reward, experiment=experiment)

    return {
        "id":         record.id,
        "arm":        arm,
        "reward":     reward,
        "experiment": experiment,
    }