from sqlalchemy.orm import Session
from sqlalchemy import func
from .models import Reward, Experiment


# ─── Reward CRUD ────────────────────────────────────────────────────────────

def add_reward(db: Session, arm: int, reward: float, experiment: str = "default") -> Reward:
    """Persist a reward event to the database."""
    r = Reward(arm=arm, reward=reward, experiment=experiment)
    db.add(r)
    db.commit()
    db.refresh(r)
    return r


def get_rewards(db: Session, experiment: str = "default") -> list[Reward]:
    """Fetch all reward rows for a given experiment."""
    return db.query(Reward).filter(Reward.experiment == experiment).all()


def get_arm_stats(db: Session, experiment: str = "default") -> list[dict]:
    """
    Return per-arm aggregated stats:
      - pull count
      - total reward
      - mean reward
    """
    rows = (
        db.query(
            Reward.arm,
            func.count(Reward.id).label("pulls"),
            func.sum(Reward.reward).label("total_reward"),
            func.avg(Reward.reward).label("mean_reward"),
        )
        .filter(Reward.experiment == experiment)
        .group_by(Reward.arm)
        .all()
    )
    return [
        {
            "arm": row.arm,
            "pulls": row.pulls,
            "total_reward": row.total_reward,
            "mean_reward": round(row.mean_reward, 4),
        }
        for row in rows
    ]


def get_rewards_ordered(db: Session, experiment: str = "default") -> list[Reward]:
    """Fetch rewards in chronological order (useful for time-series metrics)."""
    return (
        db.query(Reward)
        .filter(Reward.experiment == experiment)
        .order_by(Reward.created_at)
        .all()
    )


# ─── Experiment CRUD ─────────────────────────────────────────────────────────

def create_experiment(
    db: Session, name: str, algorithm: str, n_arms: int, epsilon: float = 0.1
) -> Experiment:
    exp = Experiment(name=name, algorithm=algorithm, n_arms=n_arms, epsilon=epsilon)
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return exp


def get_experiment(db: Session, name: str) -> Experiment | None:
    return db.query(Experiment).filter(Experiment.name == name).first()


def list_experiments(db: Session) -> list[Experiment]:
    return db.query(Experiment).all()