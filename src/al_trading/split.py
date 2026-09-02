"""Chronological train / validation / test splitting.

No shuffling, no interleaving: every training row predates every
validation row, which predates every test row. Splitting is by calendar
date so all tickers share the same cut points.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = ["DataSplit", "time_split"]


@dataclass
class DataSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def time_split(
    frame: pd.DataFrame,
    train_end: str | pd.Timestamp,
    val_end: str | pd.Timestamp,
    *,
    date_column: str = "date",
) -> DataSplit:
    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end)
    if not train_end < val_end:
        raise ValueError(f"train_end ({train_end.date()}) must be < val_end ({val_end.date()})")

    dates = pd.to_datetime(frame[date_column])
    train = frame[dates <= train_end]
    val = frame[(dates > train_end) & (dates <= val_end)]
    test = frame[dates > val_end]

    for name, part in (("train", train), ("val", val)):
        if part.empty:
            raise ValueError(f"{name} split is empty for the given cut dates")

    return DataSplit(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )
