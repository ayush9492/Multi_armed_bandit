from fastapi import FastAPI
from app.api.routes import router
from app.db.database import Base, engine
from app.services.experiment_service import load_bandit_state_from_db
from app.db.database import SessionLocal

# Create all DB tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Multi-Armed Bandit API",
    description="A/B testing powered by bandit algorithms (Thompson Sampling, Epsilon-Greedy, UCB1)",
    version="1.0.0",
)

app.include_router(router)


@app.on_event("startup")
def startup_event():
    """
    On startup, reload bandit state from the database so that a server
    restart doesn't lose all previously learned knowledge.
    """
    db = SessionLocal()
    try:
        load_bandit_state_from_db(db)
    finally:
        db.close()


@app.get("/", tags=["Health"])
def home():
    return {"status": "Bandit API Running"}