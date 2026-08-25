"""Esteira local de dados, drift, hiperparâmetros e treino.

Uso:
    python -m src.pipeline
    python -m src.pipeline --raw-path tests/fixtures/bank_sample.csv --n-splits 3
"""
from __future__ import annotations

import argparse
import os

from src.data_prep import prepare
from src.drift import detect_drift, write_drift_report
from src.feature_store import load_features, publish_features
from src.hyperparam_search import search_hyperparameters
from src.train import run_experiments


def run_pipeline(
    raw_path: str | None = None,
    search: bool = True,
    n_splits: int = 3,
    fail_on_drift: bool = False,
) -> dict:
    raw_path = raw_path or os.environ.get("DATATHON_RAW_CSV")
    prepared = prepare(raw_path=raw_path)
    metadata = publish_features(prepared)
    current = load_features()
    report = detect_drift(current)
    report_path = write_drift_report(report)
    if fail_on_drift and report["drift_detected"]:
        raise RuntimeError(f"Data drift detectado. Relatório: {report_path}")
    search_results = search_hyperparameters(prepared.reset_index(drop=True), n_splits=n_splits) if search else None
    summary = run_experiments()
    return {
        "rows": int(len(prepared)),
        "feature_view_version": metadata["version"],
        "drift_detected": report["drift_detected"],
        "drift_report": str(report_path),
        "hyperparam_candidates": 0 if search_results is None else int(len(search_results)),
        "policies": summary["policy"].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Esteira de dados, drift e treino")
    parser.add_argument("--raw-path", default=None, help="CSV local no schema Bank Marketing")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args()
    result = run_pipeline(
        raw_path=args.raw_path,
        search=not args.skip_search,
        n_splits=args.n_splits,
        fail_on_drift=args.fail_on_drift,
    )
    print(result)


if __name__ == "__main__":
    main()
