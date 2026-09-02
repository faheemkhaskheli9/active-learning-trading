"""Tests for the training CLI."""

from __future__ import annotations

from al_trading import cli


def test_cli_train_synthetic(tmp_path, capsys):
    rc = cli.main(["train", "--output-dir", str(tmp_path / "models")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "val accuracy=" in out
    assert "done: lgbm_baseline_" in out
    assert list((tmp_path / "models").glob("*.pkl"))
    assert list((tmp_path / "models").glob("*.meta.json"))


def test_cli_train_bad_config_returns_2(tmp_path, capsys):
    rc = cli.main(["train", "--config", str(tmp_path / "missing.yaml")])
    assert rc == 2
    assert "error:" in capsys.readouterr().out


def test_cli_train_missing_data_csv_returns_2(tmp_path, capsys):
    rc = cli.main(["train", "--data", str(tmp_path / "nope.csv"), "--output-dir", str(tmp_path)])
    assert rc == 2
    assert "error:" in capsys.readouterr().out
