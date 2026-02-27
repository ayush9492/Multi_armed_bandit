from pydantic import BaseModel, Field, field_validator


class RewardRequest(BaseModel):
    arm: int = Field(..., ge=0, description="Index of the arm that was shown")
    reward: float = Field(..., ge=0.0, le=1.0, description="Observed reward (0 or 1 for binary)")
    experiment: str = Field(default="default", description="Experiment name")

    @field_validator("reward")
    @classmethod
    def reward_must_be_valid(cls, v):
        if v not in (0, 1) and not (0.0 < v < 1.0):
            # Allow 0, 1 (binary) and any value in between (continuous)
            raise ValueError("reward must be between 0.0 and 1.0")
        return v


class SelectResponse(BaseModel):
    variant: int
    experiment: str = "default"


class RewardResponse(BaseModel):
    status: str
    id: int
    arm: int
    reward: float
    experiment: str


class ArmStats(BaseModel):
    arm: int
    pulls: int
    total_reward: float
    mean_reward: float


class ExperimentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    algorithm: str = Field(default="thompson", pattern="^(thompson|epsilon_greedy|ucb)$")
    n_arms: int = Field(default=3, ge=2, le=20)
    epsilon: float = Field(default=0.1, ge=0.0, le=1.0)


class ExperimentResponse(BaseModel):
    id: int
    name: str
    algorithm: str
    n_arms: int
    epsilon: float