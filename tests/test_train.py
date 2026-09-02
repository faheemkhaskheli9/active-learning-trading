"""End-to-end baseline training tests (issue #1)."""

from __future__ import annotations

import json

import joblib
import pytest

from al_trading.config import TrainConfig, load_config
from al_trading.train import train_baseline


@pytest.fixture
def fast_config(tmp_path):
    return TrainConfig(
        tickers=["AAA", "BBB", "CCC"],
        start="2018-01-01",
        end="2021-12-31",
        train_end="2020-06-30",
        val_end="2021-03-31",
        output_dir=str(tmp_path / "models"),
        random_state=7,
        lgbm_params={
            "objective": "binary",
            "n_estimators": 60,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "n_jobs": 1,
            "verbose": -1,
        },
    )


def test_training_writes_versioned_model_and_metadata(fast_config):
    result = train_baseline(fast_config)

    assert result.model_path.exists()
    assert result.meta_path.exists()
    assert result.run_id in result.model_path.name
    assert result.model_path.name.endswith(".pkl")
    # timestamped identifier
    assert result.run_id.startswith("lgbm_baseline_")
    # atomic: nothing left behind
    assert not list(result.model_path.parent.glob(".*.tmp"))


def test_metadata_records_features_dates_hyperparams_and_metrics(fast_config):
    result = train_baseline(fast_config)
    meta = json.loads(result.meta_path.read_text())

    assert len(meta["features"]) >= 8
    assert meta["date_ranges"]["train"] == ["<=", "2020-06-30"]
    assert meta["hyperparameters"]["n_estimators"] == 60
    assert 0.0 <= meta["metrics"]["val"]["accuracy"] <= 1.0
    assert "directional_sharpe" in meta["metrics"]["val"]
    assert meta["data_source"] == "synthetic"
    assert meta["split_sizes"]["train"] > 0


def test_saved_model_loads_and_predicts(fast_config):
    result = train_baseline(fast_config)
    model = joblib.load(result.model_path)

    from al_trading.data import make_synthetic_ohlcv
    from al_trading.features import build_features

    bundle = build_features(make_synthetic_ohlcv(["AAA"], "2021-01-01", "2021-12-31", seed=7))
    preds = model.predict(bundle.X)
    assert len(preds) == len(bundle.X)
    assert set(map(int, set(preds))).issubset({0, 1})


def test_training_is_deterministic(fast_config):
    m1 = train_baseline(fast_config).metrics
    m2 = train_baseline(fast_config).metrics
    assert m1["accuracy"] == m2["accuracy"]
    assert m1["f1"] == m2["f1"]


def test_printed_validation_metric(fast_config, capsys):
    train_baseline(fast_config)
    out = capsys.readouterr().out
    assert "val accuracy=" in out


def test_load_config_missing_explicit_path_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_load_config_none_returns_defaults():
    cfg = load_config(None)
    assert cfg.model_name == "lgbm_baseline"
    assert cfg.data_path is None


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("model_name: x\nbogus_key: 1\n")
    with pytest.raises(ValueError, match="unknown config key"):
        load_config(path)
