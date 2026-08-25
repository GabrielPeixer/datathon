"""Etapa 5 - Serviço demonstrável: aplicação FastAPI que recebe os dados de um
cliente e retorna a oferta/canal recomendado usando a política contextual de
Thompson Sampling treinada.

Execução local:
    uvicorn src.api:app --reload
Depois abra http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src.bandits import ThompsonSampling, load_policy
from src.data_prep import build_segment as build_segment_frame
from src.governance import explain_decision

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "models" / "thompson_sampling_contextual.json"
STATIC_DIR = ROOT / "static"

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

REQUEST_ID_CONTEXT: ContextVar[str] = ContextVar("request_id", default="n/a")


class RequestIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = REQUEST_ID_CONTEXT.get()
        return True


def configure_observability() -> tuple[logging.Logger, object]:
    logger = logging.getLogger("datathon.api")
    logger.setLevel(getattr(logging, os.getenv("DATATHON_LOG_LEVEL", "INFO").upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s datathon.api request_id=%(request_id)s %(message)s"
            )
        )
        handler.addFilter(RequestIDFilter())
        logger.addHandler(handler)
    logger.propagate = False

    resource = Resource.create({"service.name": "datathon-api", "service.version": "1.0.0"})
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        provider = TracerProvider(resource=resource)
        if "pytest" not in sys.modules and "PYTEST_CURRENT_TEST" not in os.environ:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)

    tracer = trace.get_tracer("datathon.api")
    return logger, tracer


logger, tracer = configure_observability()

app = FastAPI(
    title="Datathon - Recomendação Adaptativa de Ofertas",
    description="Thompson Sampling contextual treinado sobre a base Kaggle bank-marketing.",
    version="1.0.0",
)

_policy: ThompsonSampling | None = None


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    token = REQUEST_ID_CONTEXT.set(request_id)
    started_at = time.perf_counter()
    try:
        with tracer.start_as_current_span("http.server.request") as span:
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.route", request.url.path)
            span.set_attribute("http.request_id", request_id)
            try:
                response = await call_next(request)
            except Exception as exc:  # pragma: no cover - caminho coberto pela robustez do middleware
                span.record_exception(exc)
                span.set_attribute("error.type", type(exc).__name__)
                logger.exception(
                    "request_failed",
                    extra={
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": 500,
                        "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
                        "request_id": request_id,
                    },
                )
                raise

            elapsed = time.perf_counter() - started_at
            span.set_attribute("http.status_code", response.status_code)
            span.set_attribute("http.response_time_ms", round(elapsed * 1000, 3))
            logger.info(
                "http_request_completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": round(elapsed * 1000, 3),
                    "request_id": request_id,
                },
            )
            return response
    finally:
        REQUEST_ID_CONTEXT.reset(token)


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
    model_config = ConfigDict(extra="forbid")

    age: int = Field(..., ge=18, le=120, examples=[35])
    job: str = Field("unknown", max_length=64, examples=["technician"])
    housing: str = Field("no", pattern="^(yes|no|unknown)$", examples=["yes"])
    loan: str = Field("no", pattern="^(yes|no|unknown)$", examples=["no"])


class Recommendation(BaseModel):
    segment: str
    recommended_arm: str
    expected_conversion_rate: float
    posterior: dict[str, dict[str, float]]
    explanation: dict
    human_in_the_loop: str


def build_segment(client: Client) -> str:
    """Deriva o segmento contextual reutilizando a mesma regra do treino."""
    row = pd.DataFrame([{"age": client.age, "loan": client.loan, "housing": client.housing}])
    return build_segment_frame(row).iloc[0]


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health(response: Response) -> dict:
    try:
        get_policy()
        POLICY_LOADED.set(1)
        logger.info("health_check_ok", extra={"policy_loaded": True})
        return {"status": "ok", "policy_loaded": True}
    except (HTTPException, OSError, ValueError, KeyError):
        POLICY_LOADED.set(0)
        logger.warning("health_check_degraded", extra={"policy_loaded": False})
        response.status_code = 503
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

    with tracer.start_as_current_span("recommendation.generate") as span:
        span.set_attribute("segment", segment)
        span.set_attribute("client_age", client.age)
        span.set_attribute("arm_selected", arm)

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
            span.set_attribute(f"posterior.{a}.mean_conversion", mean_conversion)

        RECOMMENDATION_COUNTER.labels(arm, segment).inc()
        RECOMMENDATION_LATENCY.observe(time.perf_counter() - start)

        logger.info(
            "recommendation_generated",
            extra={
                "segment": segment,
                "recommended_arm": arm,
                "expected_conversion_rate": posterior[arm]["mean_conversion"],
                "latency_ms": round((time.perf_counter() - start) * 1000, 3),
            },
        )

        result = Recommendation(
            segment=segment,
            recommended_arm=arm,
            expected_conversion_rate=posterior[arm]["mean_conversion"],
            posterior=posterior,
            explanation=explain_decision(policy, segment, arm, posterior),
            human_in_the_loop=(
                "Recomendação sujeita a revisão humana para decisões sensíveis, "
                "conforme política de governança documentada no README."
            ),
        )
        span.set_attribute("recommendation.output", result.model_dump_json())
        return result
