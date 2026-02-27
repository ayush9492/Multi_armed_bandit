"""
test_api.py
────────────
Integration tests for the FastAPI endpoints.

Run with:
    pytest tests/test_api.py -v

Uses TestClient (built on httpx) — no real server needed.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.db.database import Base;get_db if False else None  # noqa
from app.db.database import SessionLocal


# ─── Test DB Setup ────────────────────────────────────────────────────────────

TEST_DATABASE_URL = "sqlite:///./test_bandit.db"

test_engine = create_engine(
    TEST_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Create tables in test DB
Base.metadata.create_all(bind=test_engine)

# Override the real DB dependency with the test one
from app.api.routes import get_db
app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_db():
    """Wipe the test DB between tests."""
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)
    yield


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "Bandit API Running"


class TestSelectEndpoint:
    def test_select_returns_variant(self):
        response = client.get("/select")
        assert response.status_code == 200
        data = response.json()
        assert "variant" in data
        assert isinstance(data["variant"], int)

    def test_select_returns_valid_arm_index(self):
        for _ in range(20):
            response = client.get("/select")
            data = response.json()
            assert data["variant"] >= 0


class TestRewardEndpoint:
    def test_reward_returns_updated(self):
        response = client.post("/reward", json={"arm": 0, "reward": 1.0})
        assert response.status_code == 200
        assert response.json()["status"] == "updated"

    def test_reward_stores_correct_values(self):
        response = client.post("/reward", json={"arm": 1, "reward": 0.0})
        data = response.json()
        assert data["arm"] == 1
        assert data["reward"] == 0.0

    def test_reward_rejects_invalid_arm(self):
        response = client.post("/reward", json={"arm": 999, "reward": 1.0})
        assert response.status_code == 422

    def test_reward_rejects_invalid_reward_value(self):
        response = client.post("/reward", json={"arm": 0, "reward": 5.0})
        assert response.status_code == 422

    def test_reward_with_experiment_name(self):
        response = client.post(
            "/reward", json={"arm": 0, "reward": 1.0, "experiment": "my_exp"}
        )
        assert response.status_code == 200
        assert response.json()["experiment"] == "my_exp"


class TestStatsEndpoint:
    def test_stats_empty_initially(self):
        response = client.get("/stats")
        assert response.status_code == 200
        assert response.json() == []

    def test_stats_after_rewards(self):
        client.post("/reward", json={"arm": 0, "reward": 1.0})
        client.post("/reward", json={"arm": 0, "reward": 0.0})
        client.post("/reward", json={"arm": 1, "reward": 1.0})

        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        arms = {d["arm"]: d for d in data}

        assert arms[0]["pulls"] == 2
        assert arms[0]["mean_reward"] == pytest.approx(0.5)
        assert arms[1]["pulls"] == 1
        assert arms[1]["mean_reward"] == pytest.approx(1.0)


class TestStateEndpoint:
    def test_state_returns_algorithm_info(self):
        response = client.get("/state")
        assert response.status_code == 200
        data = response.json()
        assert "algorithm" in data
        assert "n_arms" in data


class TestExperimentEndpoints:
    def test_create_experiment(self):
        response = client.post(
            "/experiments",
            json={"name": "test_exp", "algorithm": "thompson", "n_arms": 3},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_exp"
        assert data["algorithm"] == "thompson"

    def test_list_experiments(self):
        client.post("/experiments", json={"name": "exp1", "algorithm": "ucb", "n_arms": 2})
        client.post("/experiments", json={"name": "exp2", "algorithm": "epsilon_greedy", "n_arms": 4})
        response = client.get("/experiments")
        assert response.status_code == 200
        names = [e["name"] for e in response.json()]
        assert "exp1" in names
        assert "exp2" in names

    def test_duplicate_experiment_returns_409(self):
        client.post("/experiments", json={"name": "dup", "algorithm": "thompson", "n_arms": 3})
        response = client.post("/experiments", json={"name": "dup", "algorithm": "ucb", "n_arms": 3})
        assert response.status_code == 409