"""Baseline LightGBM trainer.

Loads historical OHLCV (real CSV or synthetic), builds no-lookahead
features, splits chronologically, trains an ``LGBMClassifier`` with
early stopping on the validation window, then serializes the model plus a
metadata sidecar (features, date ranges, hyperparameters, metrics) under a
timestamped name.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .config import TrainConfig
from .data import load_ohlcv, make_synthetic_ohlcv
from .features import build_features
from .split import time_split

__all__ = ["TrainResult", "train_baseline"]


@dataclass
class TrainResult:
    model_path: Path
    meta_path: Path
    metrics: dict[str, float]
    run_id: str


def _directional_sharpe(y_pred: np.ndarray, next_ret: np.ndarray) -> float:
    # take a long/short position by predicted direction, 1-day holding
    pnl = np.where(y_pred == 1, next_ret, -next_ret)
    if pnl.std(ddof=1) == 0 or len(pnl) < 2:
        return 0.0
    return float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(252))


def _evaluate(model, X, y, next_ret) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "directional_sharpe": _directional_sharpe(pred, np.asarray(next_ret)),
        "n_samples": int(len(y)),
        "positive_rate": float(np.mean(y)),
    }
    if len(np.unique(y)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y, proba))
    return metrics


def _atomic_dump(obj: Any, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        joblib.dump(obj, tmp)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _atomic_write_text(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def train_baseline(config: TrainConfig | None = None) -> TrainResult:
    import lightgbm as lgb

    config = config or TrainConfig()

    if config.data_path:
        ohlcv = load_ohlcv(config.data_path)
        data_source = f"csv:{config.data_path}"
    else:
        ohlcv = make_synthetic_ohlcv(
            config.tickers, config.start, config.end, seed=config.random_state
        )
        data_source = "synthetic"

    bundle = build_features(ohlcv)
    split = time_split(bundle.frame, config.train_end, config.val_end)

    feat = bundle.feature_columns
    model = lgb.LGBMClassifier(random_state=config.random_state, **config.lgbm_params)
    model.fit(
        split.train[feat],
        split.train["label"],
        eval_set=[(split.val[feat], split.val["label"])],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False)],
    )

    val_metrics = _evaluate(model, split.val[feat], split.val["label"], split.val["next_ret"])
    test_metrics = (
        _evaluate(model, split.test[feat], split.test["label"], split.test["next_ret"])
        if not split.test.empty
        else {}
    )

    now = datetime.now(timezone.utc)
    run_id = f"{config.model_name}_{now:%Y%m%dT%H%M%SZ}"
    out_dir = Path(config.output_dir)
    model_path = out_dir / f"{run_id}.pkl"
    meta_path = out_dir / f"{run_id}.meta.json"

    _atomic_dump(model, model_path)

    meta = {
        "run_id": run_id,
        "created_at": now.isoformat(timespec="seconds"),
        "data_source": data_source,
        "features": feat,
        "date_ranges": {
            "data": [config.start, config.end],
            "train": ["<=", config.train_end],
            "val": [config.train_end, config.val_end],
            "test": [">", config.val_end],
        },
        "split_sizes": split.sizes(),
        "hyperparameters": {"random_state": config.random_state, **config.lgbm_params},
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        "metrics": {"val": val_metrics, "test": test_metrics},
        "config": asdict(config),
        "library_versions": {
            "lightgbm": lgb.__version__,
        },
    }
    _atomic_write_text(meta_path, json.dumps(meta, indent=2, default=str))

    print(
        f"[{run_id}] val accuracy={val_metrics['accuracy']:.3f} "
        f"roc_auc={val_metrics.get('roc_auc', float('nan')):.3f} "
        f"directional_sharpe={val_metrics['directional_sharpe']:.2f}"
    )
    print(f"  model -> {model_path}")
    print(f"  meta  -> {meta_path}")

    return TrainResult(
        model_path=model_path,
        meta_path=meta_path,
        metrics=val_metrics,
        run_id=run_id,
    )
