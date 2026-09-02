"""Feature no-lookahead + chronological split tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from al_trading.data import make_synthetic_ohlcv
from al_trading.features import FEATURE_COLUMNS, build_features
from al_trading.split import time_split


@pytest.fixture(scope="module")
def bundle():
    ohlcv = make_synthetic_ohlcv(["AAA", "BBB"], "2019-01-01", "2020-12-31", seed=1)
    return build_features(ohlcv)


def test_feature_columns_present_and_finite(bundle):
    assert list(bundle.X.columns) == FEATURE_COLUMNS
    assert np.isfinite(bundle.X.to_numpy()).all()
    assert set(bundle.y.unique()).issubset({0, 1})


def test_label_is_strictly_next_day_return_sign():
    ohlcv = make_synthetic_ohlcv(["AAA"], "2019-01-01", "2019-06-30", seed=3)
    bundle = build_features(ohlcv)
    merged = bundle.frame.merge(
        ohlcv.rename(columns={"close": "close_raw"})[["date", "ticker", "close_raw"]],
        on=["date", "ticker"],
    ).sort_values("date").reset_index(drop=True)

    close = merged["close_raw"].to_numpy()
    expected = (close[1:] > close[:-1]).astype(int)
    assert np.array_equal(merged["label"].to_numpy()[:-1], expected[: len(merged) - 1])


def test_last_row_per_ticker_is_dropped():
    ohlcv = make_synthetic_ohlcv(["AAA", "BBB"], "2019-01-01", "2019-12-31", seed=5)
    bundle = build_features(ohlcv)
    for ticker in ("AAA", "BBB"):
        last_raw = ohlcv[ohlcv.ticker == ticker]["date"].max()
        last_feat = bundle.frame[bundle.frame.ticker == ticker]["date"].max()
        assert last_feat < last_raw


def test_features_are_deterministic():
    a = build_features(make_synthetic_ohlcv(["AAA"], "2019-01-01", "2019-12-31", seed=9))
    b = build_features(make_synthetic_ohlcv(["AAA"], "2019-01-01", "2019-12-31", seed=9))
    pd.testing.assert_frame_equal(a.frame, b.frame)


def test_build_features_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        build_features(make_synthetic_ohlcv(["AAA"]).iloc[0:0])


def test_time_split_is_strictly_chronological(bundle):
    split = time_split(bundle.frame, "2019-09-30", "2020-06-30")
    assert split.train["date"].max() <= pd.Timestamp("2019-09-30")
    assert split.val["date"].min() > pd.Timestamp("2019-09-30")
    assert split.val["date"].max() <= pd.Timestamp("2020-06-30")
    assert split.test["date"].min() > pd.Timestamp("2020-06-30")
    total = sum(split.sizes().values())
    assert total == len(bundle.frame)


def test_time_split_requires_ordered_cuts(bundle):
    with pytest.raises(ValueError, match="must be <"):
        time_split(bundle.frame, "2020-06-30", "2019-09-30")


def test_time_split_errors_on_empty_train(bundle):
    with pytest.raises(ValueError, match="train split is empty"):
        time_split(bundle.frame, "2000-01-01", "2001-01-01")
