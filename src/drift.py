"""Detecção de data drift sobre a Feature Store.

Compara um lote corrente com as estatísticas de referência da feature view.
Usa PSI para categóricas e z-score da média para numéricas.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.feature_store import (
    CATEGORICAL_FEATURES,
    FEATURE_VIEW,
    NUMERIC_FEATURES,
    compute_statistics,
    load_metadata,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
PSI_THRESHOLD = 0.2
ZSCORE_THRESHOLD = 3.0


def _log_ratio(actual: float, expected: float) -> float:
    import math

    return math.log(actual / expected)


def calculate_psi(expected: dict[str, float], actual: dict[str, float], epsilon: float = 1e-6) -> float:
    keys = set(expected) | set(actual)
    psi = 0.0
    for key in keys:
        e = max(float(expected.get(key, 0.0)), epsilon)
        a = max(float(actual.get(key, 0.0)), epsilon)
        psi += (a - e) * _log_ratio(a, e)
    return float(psi)


def detect_drift(
    current: pd.DataFrame,
    reference_stats: dict | None = None,
    feature_view: str = FEATURE_VIEW,
    psi_threshold: float = PSI_THRESHOLD,
    zscore_threshold: float = ZSCORE_THRESHOLD,
) -> dict:
    """Compara o lote corrente com o baseline da Feature Store."""
    if reference_stats is None:
        metadata = load_metadata(feature_view)
        if metadata is None:
            raise FileNotFoundError(
                "Feature Store sem metadata. Execute `python -m src.data_prep` antes do drift."
            )
        reference_stats = metadata["statistics"]

    current_stats = compute_statistics(current)
    findings: list[dict] = []

    for column in NUMERIC_FEATURES:
        ref = reference_stats.get("numeric", {}).get(column)
        cur = current_stats.get("numeric", {}).get(column)
        if not ref or not cur:
            continue
        std = ref["std"] if ref["std"] > 1e-9 else 1.0
        zscore = abs(cur["mean"] - ref["mean"]) / std
        drifted = zscore > zscore_threshold
        findings.append(
            {
                "feature": column,
                "type": "numeric",
                "metric": "mean_zscore",
                "value": round(float(zscore), 4),
                "threshold": zscore_threshold,
                "drifted": drifted,
            }
        )

    for column in CATEGORICAL_FEATURES:
        ref = reference_stats.get("categorical", {}).get(column)
        cur = current_stats.get("categorical", {}).get(column)
        if not ref or not cur:
            continue
        psi = calculate_psi(ref, cur)
        drifted = psi > psi_threshold
        findings.append(
            {
                "feature": column,
                "type": "categorical",
                "metric": "psi",
                "value": round(float(psi), 4),
                "threshold": psi_threshold,
                "drifted": drifted,
            }
        )

    report = {
        "feature_view": feature_view,
        "n_current_rows": int(len(current)),
        "drift_detected": any(item["drifted"] for item in findings),
        "findings": findings,
    }
    return report


def write_drift_report(report: dict, path: Path | None = None) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = path or (REPORTS_DIR / "drift_report.json")
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


if __name__ == "__main__":
    from src.feature_store import load_features

    current = load_features()
    report = detect_drift(current)
    output = write_drift_report(report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Relatório gravado em {output}")
