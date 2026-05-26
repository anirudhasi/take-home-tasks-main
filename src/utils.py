"""
utils.py — Cross-cutting utilities: seeding, logging, timing, I/O.
"""
import random
import time
import logging
import sys
import pickle
from pathlib import Path
from functools import wraps

import numpy as np

# ─────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


log = get_logger("utils")


# ─────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────
def seed_everything(seed: int = 42) -> None:
    """Pin all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    log.debug(f"All seeds set to {seed}")


# ─────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────
def timer(func):
    """Decorator that logs function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        log.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


# ─────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────
def save_pickle(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    log.info(f"Saved → {path}")


def load_pickle(path: Path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    log.info(f"Loaded <- {path}")
    return obj


# ─────────────────────────────────────────────────
# DataFrame helpers
# ─────────────────────────────────────────────────
def memory_usage_mb(df) -> float:
    """Return DataFrame memory usage in MB."""
    return df.memory_usage(deep=True).sum() / 1024 ** 2


def downcast_numerics(df):
    """Reduce memory footprint by downcasting numeric columns."""
    import pandas as pd
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df
