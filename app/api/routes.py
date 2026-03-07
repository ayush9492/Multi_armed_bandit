from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db import crud
from app.services.experiment_service import select_variant, get_bandit_state, update_reward
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


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/select", response_model=SelectResponse, tags=["Bandit"])
def select(experiment: str = "default", db: Session = Depends(get_db)):
    """Ask the bandit which arm to show next for a given experiment."""
    arm = select_variant(experiment=experiment, db=db)
    return SelectResponse(variant=arm, experiment=experiment)


@router.post("/reward", response_model=RewardResponse, tags=["Bandit"])
def reward(data: RewardRequest, db: Session = Depends(get_db)):
    """Submit reward and update the correct experiment's bandit."""
    try:
        result = process_reward(db, data.arm, data.reward, data.experiment)
    except RewardValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Update the in-memory bandit for this experiment
    update_reward(data.arm, data.reward, experiment=data.experiment, db=db)

    return RewardResponse(
        status="updated",
        id=result["id"],
        arm=result["arm"],
        reward=result["reward"],
        experiment=result["experiment"],
    )


@router.get("/stats", response_model=list[ArmStats], tags=["Bandit"])
def stats(experiment: str = "default", db: Session = Depends(get_db)):
    """Return per-arm statistics for a given experiment."""
    return crud.get_arm_stats(db, experiment=experiment)


@router.get("/state", tags=["Bandit"])
def state(experiment: str = "default", db: Session = Depends(get_db)):
    """Inspect internal bandit state for a given experiment."""
    return get_bandit_state(experiment=experiment, db=db)


@router.post("/experiments", response_model=ExperimentResponse, tags=["Experiments"])
def create_experiment(data: ExperimentCreate, db: Session = Depends(get_db)):
    """Register a new experiment with a specific algorithm."""
    existing = crud.get_experiment(db, data.name)
    if existing:
        raise HTTPException(status_code=409, detail=f"Experiment '{data.name}' already exists.")
    exp = crud.create_experiment(db, name=data.name, algorithm=data.algorithm,
                                  n_arms=data.n_arms, epsilon=data.epsilon)
    return ExperimentResponse(id=exp.id, name=exp.name, algorithm=exp.algorithm,
                               n_arms=exp.n_arms, epsilon=exp.epsilon)


@router.get("/experiments", response_model=list[ExperimentResponse], tags=["Experiments"])
def list_experiments(db: Session = Depends(get_db)):
    """List all registered experiments."""
    return [
        ExperimentResponse(id=e.id, name=e.name, algorithm=e.algorithm,
                           n_arms=e.n_arms, epsilon=e.epsilon)
        for e in crud.list_experiments(db)
    ]