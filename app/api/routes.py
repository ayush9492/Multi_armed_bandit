from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db import crud
from app.services.experiment_service import select_variant, get_bandit_state
from app.services.reward_service import process_reward, RewardValidationError
from app.api.schemas import (
    RewardRequest,
    RewardResponse,
    SelectResponse,
    ArmStats,
    ExperimentCreate,
    ExperimentResponse,
)

router = APIRouter()


# ─── Dependency ──────────────────────────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─── Bandit Endpoints ─────────────────────────────────────────────────────────

@router.get(
    "/select",
    response_model=SelectResponse,
    summary="Get the next arm to show",
    tags=["Bandit"],
)
def select(experiment: str = "default"):
    """
    Ask the bandit which variant to show the next user.

    Returns the arm index (e.g. 0, 1, or 2) based on the current
    algorithm's exploration/exploitation strategy.

    Each experiment uses its own independent bandit instance.
    """
    arm = select_variant(experiment=experiment)
    return SelectResponse(variant=arm, experiment=experiment)


@router.post(
    "/reward",
    response_model=RewardResponse,
    summary="Submit a reward for an arm",
    tags=["Bandit"],
)
def reward(data: RewardRequest, db: Session = Depends(get_db)):
    """
    Submit the outcome of showing a variant to a user.

    - `arm`:        which variant was shown (must match what `/select` returned)
    - `reward`:     1.0 for success (click, conversion), 0.0 for failure
    - `experiment`: name of the experiment (optional, defaults to "default")
    """
    try:
        result = process_reward(db, data.arm, data.reward, data.experiment)
    except RewardValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return RewardResponse(
        status="updated",
        id=result["id"],
        arm=result["arm"],
        reward=result["reward"],
        experiment=result["experiment"],
    )


@router.get(
    "/stats",
    response_model=list[ArmStats],
    summary="Get per-arm statistics",
    tags=["Bandit"],
)
def stats(experiment: str = "default", db: Session = Depends(get_db)):
    """
    Return aggregated stats for each arm in a given experiment:
    pull count, total reward, and mean reward.
    """
    return crud.get_arm_stats(db, experiment=experiment)


@router.get(
    "/state",
    summary="Inspect internal bandit state",
    tags=["Bandit"],
)
def state(experiment: str = "default"):
    """
    Return the raw internal state of the bandit for a given experiment
    (counts, values, priors, etc.). Useful for debugging and monitoring.
    """
    return get_bandit_state(experiment=experiment)


# ─── Experiment Endpoints ─────────────────────────────────────────────────────

@router.post(
    "/experiments",
    response_model=ExperimentResponse,
    summary="Create a new experiment",
    tags=["Experiments"],
)
def create_experiment(data: ExperimentCreate, db: Session = Depends(get_db)):
    """
    Register a named experiment with a specific algorithm and arm count.

    Each experiment gets its own independent in-memory bandit instance,
    so different experiments can run different algorithms simultaneously.
    """
    existing = crud.get_experiment(db, data.name)
    if existing:
        raise HTTPException(status_code=409, detail=f"Experiment '{data.name}' already exists.")

    exp = crud.create_experiment(
        db,
        name=data.name,
        algorithm=data.algorithm,
        n_arms=data.n_arms,
        epsilon=data.epsilon,
    )
    return ExperimentResponse(
        id=exp.id,
        name=exp.name,
        algorithm=exp.algorithm,
        n_arms=exp.n_arms,
        epsilon=exp.epsilon,
    )


@router.get(
    "/experiments",
    response_model=list[ExperimentResponse],
    summary="List all experiments",
    tags=["Experiments"],
)
def list_experiments(db: Session = Depends(get_db)):
    """Return all registered experiments."""
    experiments = crud.list_experiments(db)
    return [
        ExperimentResponse(
            id=e.id,
            name=e.name,
            algorithm=e.algorithm,
            n_arms=e.n_arms,
            epsilon=e.epsilon,
        )
        for e in experiments
    ]