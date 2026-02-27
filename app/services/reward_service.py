"""
reward_service.py
─────────────────
Business logic layer for processing reward submissions.

Responsibilities:
  - Validate the arm index
  - Clamp / normalize reward values
  - Save to DB
  - Update the in-memory bandit
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
    Validate, persist, and feed a reward to the bandit.

    Args:
        db: Active database session.
        arm: Which arm (variant) was shown to the user.
        reward: Observed outcome. Expected to be in [0.0, 1.0].
        experiment: Experiment name (allows multiple experiments in one DB).

    Returns:
        A dict summarising what was stored.

    Raises:
        RewardValidationError if the arm index is out of range or reward is invalid.
    """
    # 1. Validate arm index
    if arm < 0 or arm >= N_ARMS:
        raise RewardValidationError(
            f"Arm index {arm} is out of range. Valid range: 0 to {N_ARMS - 1}."
        )

    # 2. Validate reward value
    if not (0.0 <= reward <= 1.0):
        raise RewardValidationError(
            f"Reward value {reward} is outside the expected range [0.0, 1.0]."
        )

    # 3. Persist to database
    record = crud.add_reward(db, arm=arm, reward=reward, experiment=experiment)

    # 4. Update in-memory bandit so it learns immediately
    update_reward(arm, reward)

    return {
        "id": record.id,
        "arm": arm,
        "reward": reward,
        "experiment": experiment,
    }