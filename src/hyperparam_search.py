"""Seleção automática de hiperparâmetros das políticas adaptativas.

Percorre uma grade pequena e reproduzível, avalia cada candidato com validação
cruzada offline e persiste o melhor conjunto em reports/best_hyperparams.json.
"""
from __future__ import annotations

import json

import mlflow
import pandas as pd

from src import train as train_module
from src.train import cross_validate_policy

EXPERIMENT_NAME = "datathon-bandit-hyperparams"
SEARCH_SPACE = {
    "epsilon_greedy": [{"epsilon": value} for value in (0.05, 0.1, 0.2)],
    "thompson_sampling": [
        {"prior_alpha": 1.0, "prior_beta": 1.0},
        {"prior_alpha": 2.0, "prior_beta": 2.0},
        {"prior_alpha": 1.0, "prior_beta": 2.0},
    ],
    "thompson_sampling_contextual": [
        {"prior_alpha": 1.0, "prior_beta": 1.0},
        {"prior_alpha": 2.0, "prior_beta": 2.0},
        {"prior_alpha": 1.0, "prior_beta": 2.0},
    ],
}


def search_hyperparameters(df: pd.DataFrame, n_splits: int = 3) -> pd.DataFrame:
    """Percorre a grade, registra no MLflow e escolhe o melhor por política."""
    arms = sorted(df["arm"].unique().tolist())
    mlflow.set_tracking_uri(f"sqlite:///{(train_module.ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    rows: list[dict] = []
    for name, candidates in SEARCH_SPACE.items():
        for params in candidates:
            run_name = f"{name}-{json.dumps(params, sort_keys=True)}"
            with mlflow.start_run(run_name=run_name):
                metrics, _folds = cross_validate_policy(
                    name, arms, df, n_splits=n_splits, hyperparams=params
                )
                mlflow.log_params({"policy": name, **params})
                mlflow.log_metrics(
                    {
                        key: float(value)
                        for key, value in metrics.items()
                        if isinstance(value, (int, float))
                    }
                )
                rows.append({"policy": name, **params, **metrics})

    results = pd.DataFrame(rows)
    winners = (
        results.sort_values(
            ["policy", "conversion_rate", "match_rate"],
            ascending=[True, False, False],
        )
        .groupby("policy", as_index=False)
        .first()
    )
    train_module.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(train_module.REPORTS_DIR / "hyperparam_search.csv", index=False)
    best = {
        row["policy"]: {
            key: row[key]
            for key in ("epsilon", "prior_alpha", "prior_beta", "conversion_rate", "match_rate")
            if key in row and pd.notna(row[key])
        }
        for _, row in winners.iterrows()
    }
    (train_module.REPORTS_DIR / "best_hyperparams.json").write_text(
        json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return results


def load_best_hyperparams() -> dict:
    path = train_module.REPORTS_DIR / "best_hyperparams.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    from src.train import load_data

    summary = search_hyperparameters(load_data().reset_index(drop=True))
    print(summary.to_string(index=False))
