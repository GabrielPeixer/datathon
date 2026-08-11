"""Etapa 5 - Serviço demonstrável: aplicação FastAPI que recebe os dados de um
cliente e retorna a oferta/canal recomendado usando a política contextual de
Thompson Sampling treinada.

Execução local:
    uvicorn src.api:app --reload
Depois abra http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.bandits import ThompsonSampling, load_policy
from src.data_prep import build_segment as build_segment_frame

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "models" / "thompson_sampling_contextual.json"

app = FastAPI(
    title="Datathon - Recomendação Adaptativa de Ofertas",
    description="Thompson Sampling contextual treinado sobre a base Kaggle bank-marketing.",
    version="1.0.0",
)

_policy: ThompsonSampling | None = None


def get_policy() -> ThompsonSampling:
    global _policy
    if _policy is None:
        if not POLICY_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Política não treinada. Execute `python -m src.train` primeiro.",
            )
        _policy = load_policy(POLICY_PATH)
    return _policy


class Client(BaseModel):
    age: int = Field(..., ge=18, le=120, examples=[35])
    job: str = Field("unknown", examples=["technician"])
    housing: str = Field("no", pattern="^(yes|no|unknown)$", examples=["yes"])
    loan: str = Field("no", pattern="^(yes|no|unknown)$", examples=["no"])


class Recommendation(BaseModel):
    segment: str
    recommended_arm: str
    expected_conversion_rate: float
    posterior: dict[str, dict[str, float]]
    human_in_the_loop: str


def build_segment(client: Client) -> str:
    """Deriva o segmento contextual reutilizando a mesma regra do treino."""
    row = pd.DataFrame([{"age": client.age, "loan": client.loan, "housing": client.housing}])
    return build_segment_frame(row).iloc[0]


@app.get("/health")
def health() -> dict:
    try:
        get_policy()
    except (HTTPException, OSError, ValueError, KeyError):
        return {"status": "degraded", "policy_loaded": False}
    return {"status": "ok", "policy_loaded": True}


@app.post("/recommend", response_model=Recommendation)
def recommend(client: Client) -> Recommendation:
    policy = get_policy()
    segment = build_segment(client)
    arm = policy.select_arm(segment)

    posterior = {}
    for a in policy.arms:
        alpha, beta = policy.posterior(a, segment)
        posterior[a] = {
            "alpha": round(alpha, 1),
            "beta": round(beta, 1),
            "mean_conversion": round(alpha / (alpha + beta), 4),
        }

    return Recommendation(
        segment=segment,
        recommended_arm=arm,
        expected_conversion_rate=posterior[arm]["mean_conversion"],
        posterior=posterior,
        human_in_the_loop=(
            "Recomendação sujeita a revisão humana para decisões sensíveis, "
            "conforme política de governança documentada no README."
        ),
    )
