"""Placeholder module to mirror original repository structure.
If your original file adds stochastic time-to-maturity or scenarios,
plug them here and call from FastAPI as needed."""
import random

def randomize_t(t: float, low: float = 0.5, high: float = 1.5) -> float:
    return t * random.uniform(low, high)
