import pytest
from fastapi.testclient import TestClient

from src import api
from src.bandits import ThompsonSampling, load_policy


client = TestClient(api.app)


def test_artefato_versionado_da_api_pode_ser_carregado():
    politica = load_policy(api.POLICY_PATH)

    assert isinstance(politica, ThompsonSampling)
    assert politica.contextual is True
    assert set(politica.arms) == {"cellular", "telephone"}


def test_health_reflete_disponibilidade_do_modelo(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_policy", None)
    monkeypatch.setattr(api, "POLICY_PATH", tmp_path / "inexistente.json")

    resposta = client.get("/health")

    assert resposta.status_code == 503
    assert resposta.json() == {"status": "degraded", "policy_loaded": False}


def test_frontend_e_servido_na_raiz():
    resposta = client.get("/")

    assert resposta.status_code == 200
    assert resposta.headers["content-type"].startswith("text/html")
    assert "Recomendação Adaptativa de Ofertas" in resposta.text


@pytest.mark.parametrize("age", [17, 121])
def test_recomendacao_rejeita_idade_fora_dos_limites(age):
    resposta = client.post(
        "/recommend",
        json={"age": age, "job": "unknown", "housing": "no", "loan": "no"},
    )

    assert resposta.status_code == 422


@pytest.mark.parametrize(
    ("age", "expected_band"),
    [(30, "jovem"), (31, "adulto"), (45, "adulto"), (46, "meia_idade"), (60, "meia_idade"), (61, "senior")],
)
def test_faixas_etarias_nos_limites(age, expected_band):
    cliente = api.Client(age=age, housing="no", loan="no")

    assert api.build_segment(cliente) == f"{expected_band}_sem_credito"


def test_recomendacao_retorna_oferta_e_evidencias(monkeypatch):
    politica = ThompsonSampling(["cellular", "telephone"], contextual=True, seed=42)
    for _ in range(10):
        politica.update("cellular", 1, "senior_sem_credito")
        politica.update("telephone", 0, "senior_sem_credito")
    monkeypatch.setattr(api, "_policy", politica)

    resposta = client.post(
        "/recommend",
        json={"age": 67, "job": "retired", "housing": "no", "loan": "no"},
    )

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["segment"] == "senior_sem_credito"
    assert corpo["recommended_arm"] == "cellular"
    assert corpo["posterior"]["cellular"]["mean_conversion"] > corpo["posterior"]["telephone"]["mean_conversion"]
    assert "revisão humana" in corpo["human_in_the_loop"]
    assert corpo["explanation"]["recommended_arm"] == "cellular"
    assert corpo["explanation"]["segment"] == "senior_sem_credito"
    assert "média posterior" in corpo["explanation"]["reason"]


def test_metrics_endpoint_expoe_prometheus_metrics():
    resposta = client.get("/metrics")

    assert resposta.status_code == 200
    assert resposta.headers["content-type"].startswith("text/plain")
    corpo = resposta.text
    assert "datathon_http_requests_total" in corpo
    assert "datathon_recommendations_total" in corpo
    assert "datathon_policy_loaded" in corpo


def test_observabilidade_esta_configurada():
    assert hasattr(api, "logger")
    assert api.logger.name == "datathon.api"
    assert hasattr(api, "tracer")
    assert api.tracer is not None
