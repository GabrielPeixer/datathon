"""Etapa 5 - Serviço demonstrável: aplicação FastAPI que recebe os dados de um
cliente e retorna a oferta/canal recomendado usando a política contextual de
Thompson Sampling treinada.

Execução local:
    uvicorn src.api:app --reload
Depois abra http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src.bandits import ThompsonSampling, load_policy
from src.data_prep import build_segment as build_segment_frame

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "models" / "thompson_sampling_contextual.json"

REQUEST_COUNTER = Counter(
    "datathon_http_requests_total",
    "Total de requisições HTTP recebidas pela API.",
    labelnames=("method", "endpoint", "status"),
)
REQUEST_LATENCY = Histogram(
    "datathon_http_request_duration_seconds",
    "Tempo de resposta de cada requisição HTTP em segundos.",
    labelnames=("method", "endpoint"),
)
RECOMMENDATION_COUNTER = Counter(
    "datathon_recommendations_total",
    "Total de recomendações geradas por braço e segmento.",
    labelnames=("arm", "segment"),
)
RECOMMENDATION_LATENCY = Histogram(
    "datathon_recommendation_latency_seconds",
    "Tempo de processamento da recomendação em segundos.",
)
POLICY_LOADED = Gauge(
    "datathon_policy_loaded",
    "Indicador se a política contextual foi carregada corretamente.",
)
POLICY_EXPECTED_RATE = Gauge(
    "datathon_policy_expected_conversion_rate",
    "Taxa de conversão esperada por braço e segmento.",
    labelnames=("arm", "segment"),
)

app = FastAPI(
    title="Datathon - Recomendação Adaptativa de Ofertas",
    description="Thompson Sampling contextual treinado sobre a base Kaggle bank-marketing.",
    version="1.0.0",
)

_policy: ThompsonSampling | None = None


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    started_at = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - started_at
    endpoint = request.url.path or "/"
    REQUEST_COUNTER.labels(request.method, endpoint, str(response.status_code)).inc()
    REQUEST_LATENCY.labels(request.method, endpoint).observe(elapsed)
    if request.url.path == "/health":
        try:
            get_policy()
            POLICY_LOADED.set(1)
        except (HTTPException, OSError, ValueError, KeyError):
            POLICY_LOADED.set(0)
    return response


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
        POLICY_LOADED.set(1)
        return {"status": "ok", "policy_loaded": True}
    except (HTTPException, OSError, ValueError, KeyError):
        POLICY_LOADED.set(0)
        return {"status": "degraded", "policy_loaded": False}


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/recommend", response_model=Recommendation)
def recommend(client: Client) -> Recommendation:
    start = time.perf_counter()
    policy = get_policy()
    segment = build_segment(client)
    arm = policy.select_arm(segment)

    posterior = {}
    for a in policy.arms:
        alpha, beta = policy.posterior(a, segment)
        mean_conversion = alpha / (alpha + beta)
        posterior[a] = {
            "alpha": round(alpha, 1),
            "beta": round(beta, 1),
            "mean_conversion": round(mean_conversion, 4),
        }
        POLICY_EXPECTED_RATE.labels(a, segment).set(mean_conversion)

    RECOMMENDATION_COUNTER.labels(arm, segment).inc()
    RECOMMENDATION_LATENCY.observe(time.perf_counter() - start)

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
