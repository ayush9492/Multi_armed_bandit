from fastapi import FastAPI
from app.api.routes import router
from app.db.database import Base, engine
from app.services.experiment_service import load_all_bandits_from_db
from app.db.database import SessionLocal

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Multi-Armed Bandit API",
    description="A/B testing powered by bandit algorithms (Thompson Sampling, Epsilon-Greedy, UCB1)",
    version="1.0.0",
)

app.include_router(router)


@app.on_event("startup")
def startup_event():
    """On startup, reload all experiment bandits from DB."""
    db = SessionLocal()
    try:
        load_all_bandits_from_db(db)
    finally:
        db.close()


@app.get("/", tags=["Health"])
def home():
    return {"status": "Bandit API Running"}