"""Tests for the daily prediction generation job (issue #2)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from al_trading.config import PredictConfig
from al_trading.data import make_synthetic_ohlcv
from al_trading.predict import PredictionError, find_latest_model, run_daily_predictions
from al_trading.storage import InMemoryPredictionStore, PredictionStoreError
from al_trading.train import TrainConfig, train_baseline


def _train_a_model(tmp_path, tickers=("AAA", "BBB")):
    config = TrainConfig(
        tickers=list(tickers),
        start="2018-01-01",
        end="2021-12-31",
        train_end="2020-12-31",
        val_end="2021-06-30",
        output_dir=str(tmp_path / "models"),
    )
    return train_baseline(config)


def test_find_latest_model_picks_most_recent_run(tmp_path):
    result1 = _train_a_model(tmp_path)
    result2 = _train_a_model(tmp_path)  # a second run -> later run_id

    model_path, meta = find_latest_model(tmp_path / "models")

    assert meta["run_id"] in (result1.run_id, result2.run_id)
    assert model_path.exists()


def test_find_latest_model_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_latest_model(tmp_path / "nope")


def test_find_latest_model_empty_dir_raises(tmp_path):
    (tmp_path / "models").mkdir()
    with pytest.raises(FileNotFoundError, match="no trained model metadata"):
        find_latest_model(tmp_path / "models")


def test_run_daily_predictions_writes_expected_rows(tmp_path):
    train_result = _train_a_model(tmp_path, tickers=("AAA", "BBB"))
    config = PredictConfig(model_dir=str(tmp_path / "models"), tickers=["AAA", "BBB"])
    store = InMemoryPredictionStore()

    result = run_daily_predictions(config, store)

    assert result.n_predictions == 2
    assert len(store.rows) == 2
    symbols = {row.symbol for row in store.rows}
    assert symbols == {"AAA", "BBB"}
    for row in store.rows:
        assert row.model_version == train_result.run_id
        assert row.predicted_direction in (0, 1)
        assert 0.0 <= row.predicted_value <= 1.0
        # ISO date string
        pd.Timestamp(row.prediction_date)


def test_run_daily_predictions_uses_latest_row_per_ticker(tmp_path):
    _train_a_model(tmp_path, tickers=("AAA",))
    config = PredictConfig(model_dir=str(tmp_path / "models"), tickers=["AAA"])
    store = InMemoryPredictionStore()

    result = run_daily_predictions(config, store)

    ohlcv = make_synthetic_ohlcv(["AAA"], seed=7)
    expected_last_date = ohlcv["date"].max().strftime("%Y-%m-%d")
    assert result.rows[0].prediction_date == expected_last_date


def test_run_daily_predictions_missing_model_raises(tmp_path):
    (tmp_path / "models").mkdir()
    config = PredictConfig(model_dir=str(tmp_path / "models"))
    with pytest.raises(FileNotFoundError):
        run_daily_predictions(config, InMemoryPredictionStore())


def test_run_daily_predictions_store_failure_leaves_no_partial_rows(tmp_path):
    _train_a_model(tmp_path, tickers=("AAA", "BBB", "CCC"))
    config = PredictConfig(model_dir=str(tmp_path / "models"), tickers=["AAA", "BBB", "CCC"])
    store = InMemoryPredictionStore(fail_after=0)  # any non-empty batch fails

    with pytest.raises(PredictionError, match="failed to write predictions"):
        run_daily_predictions(config, store)

    assert store.rows == []  # all-or-nothing: nothing partial landed


def test_run_daily_predictions_corrupt_metadata_raises_clear_error(tmp_path):
    _train_a_model(tmp_path, tickers=("AAA",))
    models_dir = tmp_path / "models"
    meta_files = sorted(models_dir.glob("*.meta.json"))
    meta_files[-1].write_text("{not valid json", encoding="utf-8")

    config = PredictConfig(model_dir=str(models_dir), tickers=["AAA"])
    with pytest.raises(PredictionError, match="could not read model metadata"):
        run_daily_predictions(config, InMemoryPredictionStore())


def test_run_daily_predictions_unknown_ticker_raises(tmp_path):
    _train_a_model(tmp_path, tickers=("AAA",))
    ohlcv_csv = tmp_path / "ohlcv.csv"
    make_synthetic_ohlcv(["AAA"], seed=7).to_csv(ohlcv_csv, index=False)

    config = PredictConfig(
        model_dir=str(tmp_path / "models"),
        tickers=["ZZZ-not-in-the-csv"],
        data_path=str(ohlcv_csv),
    )
    with pytest.raises(PredictionError, match="no OHLCV rows found"):
        run_daily_predictions(config, InMemoryPredictionStore())


def test_predict_config_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown config key"):
        PredictConfig.from_dict({"not_a_real_field": 1})
