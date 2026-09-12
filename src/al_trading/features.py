"""Feature engineering with a strict no-lookahead contract.

Every feature at row ``t`` is computed from data at or before ``t``; the
label at row ``t`` is the sign of the ``t -> t+1`` close-to-close return.
The final row per ticker (no next day) is dropped. This alignment is what
the Phase 2/4 lookahead-bias checks defend -- keep it honest here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["FeatureBundle", "FEATURE_COLUMNS", "build_features", "latest_features"]

FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "vol_10d",
    "vol_20d",
    "mom_10d",
    "ma_ratio_5_20",
    "high_low_range",
    "volume_z_20d",
    "rsi_14",
]


@dataclass
class FeatureBundle:
    frame: pd.DataFrame  # date, ticker, <features>, next_ret, label
    feature_columns: list[str]

    @property
    def X(self) -> pd.DataFrame:
        return self.frame[self.feature_columns]

    @property
    def y(self) -> pd.Series:
        return self.frame["label"]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _features_for_ticker(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("date").copy()
    close = g["close"]

    g["ret_1d"] = close.pct_change(1)
    g["ret_5d"] = close.pct_change(5)
    g["ret_10d"] = close.pct_change(10)
    g["vol_10d"] = g["ret_1d"].rolling(10).std()
    g["vol_20d"] = g["ret_1d"].rolling(20).std()
    g["mom_10d"] = close / close.shift(10) - 1.0
    g["ma_ratio_5_20"] = close.rolling(5).mean() / close.rolling(20).mean() - 1.0
    g["high_low_range"] = (g["high"] - g["low"]) / close
    vol = g["volume"].astype(float)
    g["volume_z_20d"] = (vol - vol.rolling(20).mean()) / vol.rolling(20).std()
    g["rsi_14"] = _rsi(close, 14)

    # label / realized outcome: strictly next day, no leakage
    g["next_ret"] = close.shift(-1) / close - 1.0
    g["label"] = (g["next_ret"] > 0).astype("int8")
    g.loc[g["next_ret"].isna(), "label"] = pd.NA

    return g


def build_features(ohlcv: pd.DataFrame) -> FeatureBundle:
    if ohlcv.empty:
        raise ValueError("cannot build features from an empty OHLCV frame")

    enriched = (
        ohlcv.groupby("ticker", group_keys=False)[list(ohlcv.columns)]
        .apply(_features_for_ticker)
        .reset_index(drop=True)
    )

    keep = ["date", "ticker", *FEATURE_COLUMNS, "next_ret", "label"]
    out = enriched[keep].replace([np.inf, -np.inf], np.nan).dropna()
    out = out.astype({"label": "int8"})
    return FeatureBundle(frame=out.sort_values(["date", "ticker"]).reset_index(drop=True),
                         feature_columns=list(FEATURE_COLUMNS))


def latest_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Feature row for the most recent date per ticker, with no label needed.

    ``build_features`` drops each ticker's last row because it has no
    "next day" to compute ``next_ret``/``label`` from -- which is exactly the
    row a *live* daily prediction job needs (today has no next day yet
    either). Used by :mod:`al_trading.predict`.
    """
    if ohlcv.empty:
        raise ValueError("cannot build features from an empty OHLCV frame")

    enriched = (
        ohlcv.groupby("ticker", group_keys=False)[list(ohlcv.columns)]
        .apply(_features_for_ticker)
        .reset_index(drop=True)
    )
    latest = (
        enriched.sort_values("date")
        .groupby("ticker", as_index=False, group_keys=False)
        .tail(1)
    )

    keep = ["date", "ticker", *FEATURE_COLUMNS]
    out = latest[keep].replace([np.inf, -np.inf], np.nan)
    missing = out[FEATURE_COLUMNS].isna().any(axis=1)
    if missing.any():
        bad = sorted(out.loc[missing, "ticker"].tolist())
        raise ValueError(
            f"insufficient price history to compute features for: {bad} "
            "(rolling windows need more lookback days)"
        )
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)
