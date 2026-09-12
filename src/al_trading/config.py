"""Training configuration (YAML-backed)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["PredictConfig", "TrainConfig", "load_config", "load_predict_config"]

_DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 40,
    "n_jobs": 1,
    "verbose": -1,
}


@dataclass
class TrainConfig:
    model_name: str = "lgbm_baseline"
    data_path: str | None = None  # None -> synthetic
    tickers: list[str] = field(default_factory=lambda: ["AAA", "BBB", "CCC", "DDD"])
    start: str = "2018-01-01"
    end: str = "2022-12-31"
    train_end: str = "2021-06-30"
    val_end: str = "2022-03-31"
    early_stopping_rounds: int = 40
    random_state: int = 7
    output_dir: str = "models"
    lgbm_params: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_LGBM_PARAMS))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainConfig:
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config key(s): {sorted(unknown)}")
        merged = dict(raw)
        params = dict(_DEFAULT_LGBM_PARAMS)
        params.update(merged.get("lgbm_params", {}) or {})
        merged["lgbm_params"] = params
        return cls(**merged)


def load_config(path: str | Path | None) -> TrainConfig:
    """Load a training config.

    No path -> built-in defaults. An explicitly-passed path that does not
    exist is an error (never a silent fall-through to defaults).
    """
    if path is None:
        return TrainConfig()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return TrainConfig.from_dict(raw)


@dataclass
class PredictConfig:
    """Config for the daily prediction job (Phase 1)."""

    model_dir: str = "models"
    tickers: list[str] = field(default_factory=lambda: ["AAA", "BBB", "CCC", "DDD"])
    data_path: str | None = None  # None -> synthetic
    lookback_days: int = 120  # enough calendar history for the longest rolling window
    postgres_dsn: str | None = None  # e.g. postgresql://user:pass@host:5432/dbname

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PredictConfig:
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config key(s): {sorted(unknown)}")
        return cls(**raw)


def load_predict_config(path: str | Path | None) -> PredictConfig:
    """Load a prediction-job config. Same lenient-default/strict-explicit rule as :func:`load_config`."""
    if path is None:
        return PredictConfig()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return PredictConfig.from_dict(raw)
