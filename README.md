# Active-Learning Trading System

> Time-Series, Finance & Trading portfolio project — independent open-source implementation.
> This is an original, from-scratch build. It is not affiliated with, and does not
> contain any code, prompts, data, or business logic from, any employer or client.

![status](https://img.shields.io/badge/status-in%20progress-yellow)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## 1. Problem

Static trading models degrade over time; a system that retrains on new outcomes and tracks its own performance can adapt continuously.

## 2. Architecture

```text
Daily Data -> Predict -> Store Prediction -> Realize Outcome -> Feedback into Training Set -> Periodic Retrain
```

## 3. Technology Stack

- Python
- scikit-learn/LightGBM
- PostgreSQL
- MLflow (experiment tracking)
- Airflow or cron scheduler

## 4. Feature List

- Historical model training
- Daily prediction generation
- Daily result storage
- Model feedback loop from realized outcomes
- Periodic retraining schedule
- Experiment tracking
- Duplicate-day prevention safeguards

## 5. Implementation Plan

1. Phase 1: Prediction and daily-result storage pipeline
2. Phase 2: Feedback loop wiring realized outcomes back into training data
3. Phase 3: Experiment tracking and periodic retraining scheduler
4. Phase 4: Safeguards against duplicate/lookahead data issues

## Task Tracking

Work is broken into phase-tagged user stories tracked as GitHub Issues, not in this file. To see what's open:

    gh issue list --repo faheemkhaskheli9/active-learning-trading --state open --label type:user-story

Implement Phase 1 issues first (later phases depend on it). When you start one, add label `status:in-progress`. When you finish, close it referencing the commit (e.g. `git commit -m "... Closes #4"`) and push.

## 6. Repository Structure

```text
active-learning-trading/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── .env.example
├── docker/
├── docs/
│   ├── architecture.md
│   └── evaluation.md
├── src/
├── tests/
├── configs/
├── scripts/
├── notebooks/
├── examples/
├── assets/
└── .github/
    └── workflows/
```

## 7. Setup

```bash
git clone <this-repo-url>
cd active-learning-trading
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # or: pip install -e .
cp .env.example .env              # fill in API keys / config
```

## 8. Dataset

Document which public dataset(s) or synthetic data generators are used here.
No proprietary, employer-owned, or client-identifiable data is used in this project.

## 9. Training / Execution

Phase 1 — baseline LightGBM trainer (implemented):

```bash
pip install -r requirements.txt
export PYTHONPATH=src            # or: pip install -e .

# Train on a deterministic synthetic multi-ticker series (no data needed):
python -m al_trading train --config configs/train.yaml --output-dir models

# Or point at a real long-format OHLCV CSV (date,ticker,open,high,low,close,volume):
python -m al_trading train --data data/ohlcv.csv --output-dir models
```

Each run writes `models/<model_name>_<UTC timestamp>.pkl` plus a
`.meta.json` sidecar recording the features, train/val/test date ranges,
hyperparameters, and validation metrics (accuracy, F1, ROC-AUC, a toy
directional Sharpe). Features are built with a strict no-lookahead
contract; the label is the sign of the next-day close-to-close return.

On the synthetic random-walk data the baseline is near chance by design —
it exists to bootstrap the daily prediction job, and the active-learning
loop (Phase 2+) is what improves it from realized outcomes.

## 10. Evaluation

Document evaluation metrics and how to reproduce them here (see `docs/evaluation.md`).

## 11. Results

_To be filled in as the implementation progresses — screenshots, metrics tables, and
sample outputs go here._

## 12. API

_If this project exposes an API, document the main endpoints here (or link to
auto-generated OpenAPI docs, e.g. `/docs` for FastAPI)._

## 13. Docker

```bash
docker build -t active-learning-trading .
docker run -p 8000:8000 active-learning-trading
```

## 14. Tests

```bash
pytest tests/
```

## 15. Limitations

- This is a from-scratch, independent recreation built for portfolio purposes.
- Performance numbers, once added, are based on public datasets and are not
  representative of any production system's real-world results.

## 16. Future Work

- Expand evaluation coverage and add CI-based regression checks.
- Add more configuration presets and deployment targets.
- Track open items as GitHub Issues.

## 17. Disclosure

This repository is an **independent open-source recreation inspired by the kind of
production systems I have worked on professionally**. It contains no employer or
client source code, prompts, datasets, credentials, architecture diagrams, or
business logic. All code, data, and documentation here are original or built on
publicly available datasets and open-source tools.

---
_Last updated: 2026-09-02_
