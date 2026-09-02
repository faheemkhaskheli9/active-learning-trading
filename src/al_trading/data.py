"""OHLCV data access for the baseline trainer.

Real deployments point ``load_ohlcv`` at a CSV / database export. For
offline runs and CI, ``make_synthetic_ohlcv`` produces a deterministic
multi-ticker random walk so the whole pipeline is reproducible with no
network or credentials.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["OHLCV_COLUMNS", "load_ohlcv", "make_synthetic_ohlcv"]

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
_REQUIRED = {"date", "ticker", *OHLCV_COLUMNS}


def make_synthetic_ohlcv(
    tickers: list[str] | None = None,
    start: str = "2018-01-01",
    end: str = "2022-12-31",
    seed: int = 7,
) -> pd.DataFrame:
    tickers = tickers or ["AAA", "BBB", "CCC", "DDD"]
    index = pd.bdate_range(start=start, end=end)
    rng = np.random.default_rng(seed)

    frames = []
    for i, ticker in enumerate(tickers):
        n = len(index)
        drift = rng.uniform(-2e-4, 4e-4)
        shocks = rng.normal(drift, 0.014, size=n)
        close = 50.0 * (1.0 + i * 0.1) * np.cumprod(1.0 + shocks)
        prev = np.concatenate([[close[0]], close[:-1]])
        open_ = prev * (1.0 + rng.normal(0, 0.004, size=n))
        high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.005, size=n)))
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.005, size=n)))
        volume = rng.integers(1_000_000, 8_000_000, size=n)
        frames.append(
            pd.DataFrame(
                {
                    "date": index,
                    "ticker": ticker,
                    "open": open_.round(4),
                    "high": high.round(4),
                    "low": low.round(4),
                    "close": close.round(4),
                    "volume": volume,
                }
            )
        )

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    """Load a long-format OHLCV CSV with date, ticker, open/high/low/close/volume."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"OHLCV file not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    df.columns = [c.strip().lower() for c in df.columns]
    missing = _REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV file {path} missing columns: {sorted(missing)}")
    return (
        df[["date", "ticker", *OHLCV_COLUMNS]]
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
