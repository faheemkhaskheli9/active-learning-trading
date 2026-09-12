"""Daily prediction generation job (Phase 1).

Loads the most recently trained model + its metadata sidecar, computes
today's feature row per ticker, and writes
``(symbol, date, predicted value/direction, model version)`` rows to the
``predictions`` table via a :class:`~al_trading.storage.PredictionStore`.

Storage is injected so this is testable without a real PostgreSQL server
(``InMemoryPredictionStore``) and swappable for the real one
(``PostgresPredictionStore``, the CLI default). A failed run -- missing
model, insufficient price history, a store error -- raises
:class:`PredictionError` with a clear message and writes nothing (the store
contract is all-or-nothing; see ``storage.py``), so a scheduler (cron/Airflow)
sees a non-zero exit rather than silently-missing or partial predictions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PredictConfig
from .data import load_ohlcv, make_synthetic_ohlcv
from .features import latest_features
from .storage import PredictionRow, PredictionStore, PredictionStoreError

__all__ = [
    "PredictionError",
    "PredictionRunResult",
    "find_latest_model",
    "run_daily_predictions",
]


class PredictionError(RuntimeError):
    """The daily prediction run could not be completed."""


@dataclass(frozen=True)
class PredictionRunResult:
    model_version: str
    n_predictions: int
    rows: list[PredictionRow]


def find_latest_model(model_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    """Return the most recently trained model's artifact path + metadata.

    Selection is by ``run_id``, which embeds a UTC timestamp
    (``<name>_<YYYYmmddTHHMMSSZ>``) -- lexical sort of the metadata filenames
    is therefore chronological, so this needs no filesystem mtimes.
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"model directory not found: {model_dir}")

    meta_files = sorted(model_dir.glob("*.meta.json"))
    if not meta_files:
        raise FileNotFoundError(f"no trained model metadata (*.meta.json) found in {model_dir}")

    latest_meta_path = meta_files[-1]
    try:
        meta = json.loads(latest_meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictionError(f"could not read model metadata {latest_meta_path}: {exc}") from exc

    run_id = meta.get("run_id")
    if not run_id:
        raise PredictionError(f"{latest_meta_path} has no 'run_id'")
    model_path = model_dir / f"{run_id}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"model artifact missing for run {run_id!r}: {model_path}")

    return model_path, meta


def run_daily_predictions(
    config: PredictConfig,
    store: PredictionStore,
) -> PredictionRunResult:
    """Generate and store today's predictions. Raises :class:`PredictionError`
    (or lets ``FileNotFoundError``/:class:`PredictionStoreError` propagate) on
    any failure -- no partial rows are ever written.
    """
    import joblib

    model_path, meta = find_latest_model(config.model_dir)
    feature_columns = meta.get("features")
    if not feature_columns:
        raise PredictionError(f"model metadata for {meta.get('run_id')} has no 'features' list")

    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise PredictionError(f"could not load model artifact {model_path}: {exc}") from exc

    if config.data_path:
        ohlcv = load_ohlcv(config.data_path)
    else:
        ohlcv = make_synthetic_ohlcv(config.tickers, seed=7)

    ohlcv = ohlcv[ohlcv["ticker"].isin(config.tickers)]
    if ohlcv.empty:
        raise PredictionError(f"no OHLCV rows found for tickers {config.tickers}")

    try:
        latest = latest_features(ohlcv)
    except ValueError as exc:
        raise PredictionError(f"could not build today's features: {exc}") from exc

    missing = set(feature_columns) - set(latest.columns)
    if missing:
        raise PredictionError(
            f"model {meta.get('run_id')} expects feature(s) {sorted(missing)} "
            "not produced by the current feature pipeline (model/feature-code mismatch)"
        )

    try:
        proba = model.predict_proba(latest[feature_columns])[:, 1]
    except Exception as exc:
        raise PredictionError(f"model inference failed: {exc}") from exc

    rows = [
        PredictionRow(
            symbol=str(ticker),
            prediction_date=date.strftime("%Y-%m-%d"),
            predicted_value=float(p),
            predicted_direction=int(p >= 0.5),
            model_version=str(meta["run_id"]),
        )
        for ticker, date, p in zip(latest["ticker"], latest["date"], proba)
    ]

    try:
        store.insert_predictions(rows)
    except PredictionStoreError as exc:
        raise PredictionError(f"failed to write predictions to the store: {exc}") from exc

    return PredictionRunResult(model_version=str(meta["run_id"]), n_predictions=len(rows), rows=rows)
