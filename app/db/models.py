from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.sql import func
from .database import Base


class Reward(Base):
    """
    Stores every reward event — one row per user interaction.
    Used both for analytics and to reload bandit state after restart.
    """
    __tablename__ = "rewards"

    id = Column(Integer, primary_key=True, index=True)
    arm = Column(Integer, nullable=False, index=True)
    reward = Column(Float, nullable=False)
    experiment = Column(String, default="default", index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Experiment(Base):
    """
    Tracks named experiments (e.g. 'homepage_cta', 'email_subject').
    Each experiment has its own bandit with its own algorithm config.
    """
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    algorithm = Column(String, default="thompson")
    n_arms = Column(Integer, default=3)
    epsilon = Column(Float, default=0.1)
    created_at = Column(DateTime(timezone=True), server_default=func.now())