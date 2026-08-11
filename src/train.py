"""Etapa 3 + Etapa 7 - Executa a comparação entre o baseline e as políticas
adaptativas e registra parâmetros/métricas no MLflow (tracking local em SQLite).

Uso:
    python -m src.train
"""
from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd

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
EXPERIMENT_NAME = "datathon-bandit-ofertas"
SEED = 42


def load_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "bank_prepared.csv"
    if not path.exists():
        return prepare()
    return pd.read_csv(path)


def run_experiments() -> pd.DataFrame:
    df = load_data().sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    arms = sorted(df["arm"].unique().tolist())

    # Baseline = regra fixa incumbente: canal tradicional (telephone).
    # Controle adicional = melhor braço histórico (moda do log), que um bandit
    # bem calibrado deve aprender a igualar sem conhecê-lo de antemão.
    incumbent_arm = "telephone"
    historical_best = df.groupby("arm")["converted"].mean().idxmax()

    policies = {
        "baseline_fixed": BaselinePolicy(arms, fixed_arm=incumbent_arm, seed=SEED),
        "best_historical_arm": BaselinePolicy(arms, fixed_arm=historical_best, seed=SEED),
        "epsilon_greedy": EpsilonGreedy(arms, epsilon=0.1, seed=SEED),
        "thompson_sampling": ThompsonSampling(arms, prior_alpha=1.0, prior_beta=1.0, seed=SEED),
        "thompson_sampling_contextual": ThompsonSampling(
            arms, prior_alpha=1.0, prior_beta=1.0, contextual=True, seed=SEED
        ),
    }

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []
    for name, policy in policies.items():
        with mlflow.start_run(run_name=name):
            params = {
                "policy": type(policy).__name__,
                "arms": ",".join(arms),
                "contextual": policy.contextual,
                "seed": SEED,
                "dataset": "kaggle henriqueyamahata/bank-marketing (bank-additional-full)",
                "n_events": len(df),
            }
            params.update({k: v for k, v in policy.state().items() if k in POLICY_PARAMS})
            mlflow.log_params(params)

            metrics = replay_evaluation(policy, df)
            logged = {
                k: metrics[k]
                for k in ("conversion_rate", "match_rate", "matched_events", "conversions")
            }
            mlflow.log_metrics(logged)

            state_path = MODELS_DIR / f"{name}.json"
            policy.save(state_path)
            mlflow.log_artifact(str(state_path))

            results.append({"policy": name, **logged})

    summary = pd.DataFrame(results).sort_values("conversion_rate", ascending=False)
    baseline_rate = summary.loc[summary["policy"] == "baseline_fixed", "conversion_rate"].iloc[0]
    summary["uplift_vs_baseline_pct"] = (
        100 * (summary["conversion_rate"] - baseline_rate) / baseline_rate
    ).round(1)
    return summary


if __name__ == "__main__":
    summary = run_experiments()
    print("\n=== Comparação de políticas (replay offline) ===")
    print(summary.to_string(index=False))
