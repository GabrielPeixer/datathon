"""Etapa 3 + Etapa 7 - Executa a comparação entre o baseline e as políticas
adaptativas e registra parâmetros/métricas no MLflow (tracking local em SQLite).

Uso:
    python -m src.train
"""
from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
from sklearn.model_selection import KFold

from src.bandits import (
    POLICY_PARAMS,
    BaselinePolicy,
    EpsilonGreedy,
    ThompsonSampling,
    replay_evaluation,
)
from src.data_prep import PROCESSED_DIR, prepare

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
EXPERIMENT_NAME = "datathon-bandit-ofertas"
SEED = 42
N_SPLITS = 5


def load_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "bank_prepared.csv"
    if not path.exists():
        return prepare()
    return pd.read_csv(path)


def create_policy(name: str, arms: list[str], training_df: pd.DataFrame, seed: int):
    if name == "baseline_fixed":
        return BaselinePolicy(arms, fixed_arm="telephone", seed=seed)
    if name == "best_historical_arm":
        historical_best = training_df.groupby("arm")["converted"].mean().idxmax()
        return BaselinePolicy(arms, fixed_arm=historical_best, seed=seed)
    if name == "epsilon_greedy":
        return EpsilonGreedy(arms, epsilon=0.1, seed=seed)
    if name == "thompson_sampling":
        return ThompsonSampling(arms, prior_alpha=1.0, prior_beta=1.0, seed=seed)
    if name == "thompson_sampling_contextual":
        return ThompsonSampling(
            arms, prior_alpha=1.0, prior_beta=1.0, contextual=True, seed=seed
        )
    raise ValueError(f"Política desconhecida: {name}")


def cross_validate_policy(
    name: str, arms: list[str], df: pd.DataFrame, n_splits: int = N_SPLITS
) -> tuple[dict, list[dict]]:
    """Treina e avalia uma política em folds independentes de replay offline."""
    fold_metrics = []
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for fold, (train_indices, validation_indices) in enumerate(splitter.split(df), start=1):
        training_df = df.iloc[train_indices]
        validation_df = df.iloc[validation_indices]
        policy = create_policy(name, arms, training_df, seed=SEED + fold)

        replay_evaluation(policy, training_df)
        metrics = replay_evaluation(policy, validation_df, update_policy=False)
        fold_metrics.append(
            {
                "fold": fold,
                **{
                    key: metrics[key]
                    for key in ("conversion_rate", "match_rate", "matched_events", "conversions")
                },
            }
        )

    folds = pd.DataFrame(fold_metrics)
    aggregate = {
        metric: folds[metric].mean()
        for metric in ("conversion_rate", "match_rate", "matched_events", "conversions")
    }
    aggregate["conversion_rate_std"] = folds["conversion_rate"].std(ddof=0)
    aggregate["match_rate_std"] = folds["match_rate"].std(ddof=0)
    return aggregate, fold_metrics


def run_experiments() -> pd.DataFrame:
    df = load_data().reset_index(drop=True)
    arms = sorted(df["arm"].unique().tolist())
    policy_names = (
        "baseline_fixed",
        "best_historical_arm",
        "epsilon_greedy",
        "thompson_sampling",
        "thompson_sampling_contextual",
    )

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []
    for name in policy_names:
        policy = create_policy(name, arms, df, seed=SEED)
        with mlflow.start_run(run_name=name):
            params = {
                "policy": type(policy).__name__,
                "arms": ",".join(arms),
                "contextual": policy.contextual,
                "seed": SEED,
                "dataset": "kaggle henriqueyamahata/bank-marketing (bank-additional-full)",
                "n_events": len(df),
                "cv_folds": N_SPLITS,
            }
            params.update({k: v for k, v in policy.state().items() if k in POLICY_PARAMS})
            mlflow.log_params(params)

            logged, fold_metrics = cross_validate_policy(name, arms, df)
            mlflow.log_metrics(logged)
            for metrics in fold_metrics:
                fold = int(metrics["fold"])
                mlflow.log_metrics(
                    {f"cv_{key}": value for key, value in metrics.items() if key != "fold"},
                    step=fold,
                )

            replay_evaluation(policy, df)
            state_path = MODELS_DIR / f"{name}.json"
            policy.save(state_path)
            mlflow.log_artifact(str(state_path))

            results.append({"policy": name, **logged})

    summary = pd.DataFrame(results).sort_values("conversion_rate", ascending=False)
    baseline_rate = summary.loc[summary["policy"] == "baseline_fixed", "conversion_rate"].iloc[0]
    summary["uplift_vs_baseline_pct"] = (
        100 * (summary["conversion_rate"] - baseline_rate) / baseline_rate
    ).round(1)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORTS_DIR / "experiment_summary.csv", index=False)
    return summary


if __name__ == "__main__":
    summary = run_experiments()
    print("\n=== Comparação de políticas (média da validação cruzada) ===")
    print(summary.to_string(index=False))
