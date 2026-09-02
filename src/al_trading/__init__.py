"""Active-learning trading system.

Phase 1 is the prediction + daily-result storage pipeline. This module
currently provides the baseline LightGBM trainer that bootstraps the daily
prediction job.
"""

from .config import TrainConfig, load_config
from .data import load_ohlcv, make_synthetic_ohlcv
from .features import build_features
from .split import time_split
from .train import TrainResult, train_baseline

__all__ = [
    "TrainConfig",
    "TrainResult",
    "build_features",
    "load_config",
    "load_ohlcv",
    "make_synthetic_ohlcv",
    "time_split",
    "train_baseline",
]
