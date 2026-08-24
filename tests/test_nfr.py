import pandas as pd
from fastapi.testclient import TestClient

from src import api
from src.bandits import ThompsonSampling
from src.feature_store import FEATURE_COLUMNS
from src.governance import (
    conversion_gap_by_segment,
    explain_decision,
    leaked_sensitive_columns,
    unauthorized_model_columns,
)


client = TestClient(api.app)


def test_api_rejeita_campos_sensiveis_e_payload_extra():
    resposta = client.post(
        "/recommend",
        json={
            "age": 35,
            "job": "technician",
            "housing": "no",
            "loan": "no",
            "marital": "married",
            "education": "university.degree",
        },
    )

    assert resposta.status_code == 422
    assert "marital" in resposta.text or "extra" in resposta.text.lower()


def test_feature_store_nao_aceita_colunas_sensiveis_no_contrato():
    assert leaked_sensitive_columns(FEATURE_COLUMNS) == set()
    assert unauthorized_model_columns(set(FEATURE_COLUMNS) | {"duration", "marital"}) == {
        "duration",
        "marital",
    }


def test_disparidade_de_conversao_entre_segmentos_e_auditavel():
    eventos = pd.DataFrame(
        {
            "segment": ["jovem_sem_credito", "jovem_sem_credito", "senior_sem_credito", "senior_sem_credito"],
            "converted": [1, 1, 0, 0],
        }
    )

    auditoria = conversion_gap_by_segment(eventos)

    assert auditoria["n_segments"] == 2
    assert auditoria["gap"] == 1.0
    assert auditoria["rates"]["jovem_sem_credito"] == 1.0
    assert auditoria["rates"]["senior_sem_credito"] == 0.0


def test_explicacao_local_usa_apenas_contexto_minimizado():
    politica = ThompsonSampling(["cellular", "telephone"], contextual=True, seed=42)
    posterior = {
        "cellular": {"alpha": 8.0, "beta": 2.0, "mean_conversion": 0.8},
        "telephone": {"alpha": 2.0, "beta": 8.0, "mean_conversion": 0.2},
    }

    explicacao = explain_decision(politica, "senior_sem_credito", "cellular", posterior)

    assert explicacao["context_features"] == ["segment"]
    assert explicacao["recommended_arm"] == "cellular"
    assert "senior_sem_credito" in explicacao["reason"]
    assert explicacao["evidence"]["cellular"]["mean_conversion"] > explicacao["evidence"]["telephone"]["mean_conversion"]
