import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./bandit.db")
N_ARMS = int(os.getenv("N_ARMS", 3))
ALGORITHM = os.getenv("ALGORITHM", "thompson")   # thompson | epsilon_greedy | ucb
EPSILON = float(os.getenv("EPSILON", 0.1))       # used by epsilon_greedy
DEBUG = os.getenv("DEBUG", "True") == "True"