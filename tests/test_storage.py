"""Tests for the PredictionStore abstraction (issue #2)."""

from __future__ import annotations

import pytest

from al_trading.storage import (
    InMemoryPredictionStore,
    PostgresPredictionStore,
    PredictionRow,
    PredictionStoreError,
)


def _row(symbol="AAA", date="2024-01-02") -> PredictionRow:
    return PredictionRow(
        symbol=symbol,
        prediction_date=date,
        predicted_value=0.62,
        predicted_direction=1,
        model_version="lgbm_baseline_20240101T000000Z",
    )


def test_in_memory_store_records_rows():
    store = InMemoryPredictionStore()
    store.insert_predictions([_row("AAA"), _row("BBB")])
    assert [r.symbol for r in store.rows] == ["AAA", "BBB"]


def test_in_memory_store_simulated_failure_writes_nothing():
    store = InMemoryPredictionStore(fail_after=1)
    with pytest.raises(PredictionStoreError):
        store.insert_predictions([_row("AAA"), _row("BBB"), _row("CCC")])
    assert store.rows == []


def test_in_memory_store_empty_batch_is_a_noop():
    store = InMemoryPredictionStore()
    store.insert_predictions([])
    assert store.rows == []


def test_postgres_store_raises_clear_error_without_psycopg_installed():
    try:
        import psycopg  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("a real psycopg install is present; the fallback path is not exercised")

    store = PostgresPredictionStore(dsn="postgresql://localhost/doesnotmatter")
    with pytest.raises(PredictionStoreError, match="psycopg"):
        store.insert_predictions([_row()])
