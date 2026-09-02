"""CLI for baseline model training.

    python -m al_trading train                      # synthetic data, defaults
    python -m al_trading train --config configs/train.yaml
    python -m al_trading train --data data/ohlcv.csv --output-dir models
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .config import load_config
from .train import train_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="al_trading")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train the baseline LightGBM model.")
    train.add_argument("--config", default=None, help="YAML training config.")
    train.add_argument("--data", default=None, help="Long-format OHLCV CSV (overrides config).")
    train.add_argument("--output-dir", default=None, help="Where to write model + metadata.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "train":
        try:
            config = load_config(args.config)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}")
            return 2
        if args.data is not None:
            config.data_path = args.data
        if args.output_dir is not None:
            config.output_dir = args.output_dir

        try:
            result = train_baseline(config)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}")
            return 2
        print(f"done: {result.run_id}")
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
