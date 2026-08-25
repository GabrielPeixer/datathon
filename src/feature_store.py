"""Feature Store local, versionada por metadados.

Persiste a view de features usada pelo bandit (`campaign_context`), o schema
e as estatísticas de referência para detecção de data drift. Os arquivos
tabulares ficam fora do Git; o metadata.json é o contrato versionado.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURE_STORE_DIR = ROOT / "data" / "feature_store"
FEATURE_VIEW = "campaign_context"
FEATURE_COLUMNS = ["age", "job", "housing", "loan", "segment", "arm", "converted"]
CATEGORICAL_FEATURES = ["job", "housing", "loan", "segment", "arm"]
NUMERIC_FEATURES = ["age", "converted"]


def feature_view_dir(name: str = FEATURE_VIEW) -> Path:
    path = FEATURE_STORE_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_statistics(df: pd.DataFrame) -> dict:
    """Resumo estatístico usado como baseline de drift."""
    stats: dict = {"n_rows": int(len(df)), "numeric": {}, "categorical": {}}
    for column in NUMERIC_FEATURES:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        stats["numeric"][column] = {
            "mean": float(series.mean()) if len(series) else 0.0,
            "std": float(series.std(ddof=0)) if len(series) else 0.0,
            "min": float(series.min()) if len(series) else 0.0,
            "max": float(series.max()) if len(series) else 0.0,
        }
    for column in CATEGORICAL_FEATURES:
        if column not in df.columns:
            continue
        frequencies = df[column].astype(str).value_counts(normalize=True)
        stats["categorical"][column] = {str(k): float(v) for k, v in frequencies.items()}
    return stats


def publish_features(df: pd.DataFrame, name: str = FEATURE_VIEW) -> dict:
    """Materializa a feature view e atualiza o contrato de metadados."""
    available = [column for column in FEATURE_COLUMNS if column in df.columns]
    view = df.loc[:, available].copy()
    directory = feature_view_dir(name)
    table_path = directory / "current.csv"
    metadata_path = directory / "metadata.json"
    view.to_csv(table_path, index=False)

    previous = load_metadata(name)
    version = 1 if previous is None else int(previous.get("version", 0)) + 1
    metadata = {
        "feature_view": name,
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(view)),
        "schema": {column: str(view[column].dtype) for column in view.columns},
        "entity_keys": ["segment", "arm"],
        "features": available,
        "statistics": compute_statistics(view),
        "table_path": str(table_path.as_posix()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def load_metadata(name: str = FEATURE_VIEW) -> dict | None:
    path = feature_view_dir(name) / "metadata.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_features(name: str = FEATURE_VIEW) -> pd.DataFrame:
    path = feature_view_dir(name) / "current.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature view '{name}' ainda não foi publicada. Execute `python -m src.data_prep`."
        )
    return pd.read_csv(path)


if __name__ == "__main__":
    from src.data_prep import prepare

    prepared = prepare()
    published = publish_features(prepared)
    print(f"Feature view {published['feature_view']} v{published['version']} ({published['n_rows']} linhas)")
