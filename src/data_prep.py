"""Preparação de dados da base Bank Marketing (Kaggle: henriqueyamahata/bank-marketing).

Baixa o espelho do UCI com a mesma base, remove colunas de vazamento temporal e
cria as features usadas pela simulação de bandits e pela API.
"""
from __future__ import annotations

import io
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# O UCI hospeda exatamente o mesmo arquivo distribuído no Kaggle (bank-additional-full.csv)
UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

# 'duration' é coluna de vazamento temporal: só é conhecida DEPOIS do fim da ligação.
LEAKAGE_COLUMNS = ["duration"]

AGE_BINS = [0, 30, 45, 60, 200]
AGE_LABELS = ["jovem", "adulto", "meia_idade", "senior"]


def download_raw(force: bool = False) -> Path:
    """Baixa e extrai o bank-additional-full.csv em data/raw."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RAW_DIR / "bank-additional-full.csv"
    if csv_path.exists() and not force:
        return csv_path

    resp = requests.get(UCI_URL, timeout=120)
    resp.raise_for_status()
    outer = zipfile.ZipFile(io.BytesIO(resp.content))
    inner_bytes = outer.read("bank-additional.zip")
    inner = zipfile.ZipFile(io.BytesIO(inner_bytes))
    with inner.open("bank-additional/bank-additional-full.csv") as f:
        csv_path.write_bytes(f.read())
    return csv_path


def build_segment(df: pd.DataFrame) -> pd.Series:
    """Segmento contextual = faixa etária x posse de produto de crédito (loan/housing)."""
    age_band = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
    has_credit = ((df["loan"] == "yes") | (df["housing"] == "yes")).map(
        {True: "com_credito", False: "sem_credito"}
    )
    return (age_band + "_" + has_credit).rename("segment")


def prepare(force_download: bool = False) -> pd.DataFrame:
    """Preparação completa: download -> limpeza -> engenharia de features -> salvar."""
    csv_path = download_raw(force=force_download)
    df = pd.read_csv(csv_path, sep=";")
    raw_rows = len(df)

    # Remove duplicatas reais antes de descartar a coluna que distingue contatos.
    df = df.drop_duplicates().drop(columns=LEAKAGE_COLUMNS).reset_index(drop=True)

    # Alvo: convertido (assinou o depósito a prazo)
    df["converted"] = (df["y"] == "yes").astype(int)

    # Braço efetivamente jogado no log histórico: canal de contato utilizado
    df["arm"] = df["contact"]  # 'cellular' | 'telephone'

    # Contexto usado pelo bandit contextual
    df["segment"] = build_segment(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "bank_prepared.csv"
    df.to_csv(out, index=False)
    manifest = {
        "fonte": UCI_URL,
        "arquivo": csv_path.name,
        "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "linhas_brutas": raw_rows,
        "linhas_processadas": len(df),
        "colunas_processadas": list(df.columns),
        "colunas_removidas": LEAKAGE_COLUMNS,
    }
    (PROCESSED_DIR / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return df


if __name__ == "__main__":
    data = prepare()
    print(f"Base preparada: {data.shape[0]} linhas, {data.shape[1]} colunas")
    print(data[["age", "job", "segment", "arm", "converted"]].head())
