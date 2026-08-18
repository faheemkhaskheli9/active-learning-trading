# Architecture Notes: Active-Learning Trading System

## Pipeline

```text
Daily Data -> Predict -> Store Prediction -> Realize Outcome -> Feedback into Training Set -> Periodic Retrain
```

## Components

- Historical model training
- Daily prediction generation
- Daily result storage
- Model feedback loop from realized outcomes
- Periodic retraining schedule
- Experiment tracking
- Duplicate-day prevention safeguards

## Design Notes

- Keep provider/model choices swappable behind interfaces (see `multi-llm-router`
  and similar projects in this portfolio for the general pattern).
- Prefer configuration-driven pipelines (YAML/JSON in `configs/`) over hardcoded
  parameters so experiments are reproducible.
