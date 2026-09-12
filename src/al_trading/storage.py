"""Prediction storage: a `predictions` PostgreSQL table behind a small
Protocol, so the daily prediction job is unit-testable without a real
PostgreSQL server.

``PostgresPredictionStore`` writes the whole batch inside one transaction --
a failure partway through (bad row, dropped connection) rolls back rather
than leaving partial rows in the table. ``InMemoryPredictionStore`` is the
same all-or-nothing contract for tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

__all__ = [
    "InMemoryPredictionStore",
    "PostgresPredictionStore",
    "PredictionRow",
    "PredictionStore",
    "PredictionStoreError",
]


class PredictionStoreError(RuntimeError):
    """Predictions could not be written to the store."""


@dataclass(frozen=True)
class PredictionRow:
    """One row of the ``predictions`` table."""

    symbol: str
    prediction_date: str  # ISO YYYY-MM-DD
    predicted_value: float
    predicted_direction: int
    model_version: str


class PredictionStore(Protocol):
    def insert_predictions(self, rows: list[PredictionRow]) -> None: ...


@dataclass
class InMemoryPredictionStore:
    """Test double with the same all-or-nothing contract as the real store."""

    rows: list[PredictionRow] = field(default_factory=list)
    fail_after: int | None = None
    """Set to simulate a mid-batch failure: raises before any row of a batch
    larger than this is recorded, so tests can assert nothing partial landed.
    """

    def insert_predictions(self, rows: list[PredictionRow]) -> None:
        if self.fail_after is not None and len(rows) > self.fail_after:
            raise PredictionStoreError(
                f"synthetic failure: batch of {len(rows)} exceeds fail_after={self.fail_after}"
            )
        self.rows.extend(rows)


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    prediction_date DATE NOT NULL,
    predicted_value DOUBLE PRECISION NOT NULL,
    predicted_direction SMALLINT NOT NULL,
    model_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

_INSERT_SQL = """
INSERT INTO predictions
    (symbol, prediction_date, predicted_value, predicted_direction, model_version)
VALUES (%s, %s, %s, %s, %s)
"""


@dataclass
class PostgresPredictionStore:
    """The production store. ``psycopg`` is imported lazily so importing this
    module (and running it against ``InMemoryPredictionStore``) never
    requires a Postgres client library to be installed.
    """

    dsn: str

    def _connect(self):
        try:
            import psycopg  # noqa: PLC0415 - optional heavy dep
        except ImportError as exc:
            raise PredictionStoreError(
                "The 'psycopg' package is not installed; run `pip install -r "
                "requirements.txt`, or inject InMemoryPredictionStore for testing."
            ) from exc
        try:
            return psycopg.connect(self.dsn)
        except Exception as exc:
            raise PredictionStoreError(f"could not connect to PostgreSQL: {exc}") from exc

    def ensure_schema(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(_CREATE_TABLE_SQL)

    def insert_predictions(self, rows: list[PredictionRow]) -> None:
        if not rows:
            return
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(_CREATE_TABLE_SQL)
                    cur.executemany(
                        _INSERT_SQL,
                        [
                            (
                                r.symbol,
                                r.prediction_date,
                                r.predicted_value,
                                r.predicted_direction,
                                r.model_version,
                            )
                            for r in rows
                        ],
                    )
                # psycopg3's connection context manager commits here on a
                # clean exit and rolls back automatically if an exception
                # propagates out of the `with` block -- one transaction,
                # never a partial write.
        except PredictionStoreError:
            raise
        except Exception as exc:
            raise PredictionStoreError(f"failed to insert predictions: {exc}") from exc
