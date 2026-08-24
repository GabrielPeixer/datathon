import pandas as pd

from src.drift import calculate_psi, detect_drift


def test_psi_e_zero_quando_as_distribuicoes_coincidem():
    dist = {"cellular": 0.6, "telephone": 0.4}

    assert calculate_psi(dist, dist) == 0


def test_detecta_drift_quando_o_canal_muda_de_dominio():
    reference = {
        "numeric": {"age": {"mean": 40.0, "std": 5.0, "min": 30.0, "max": 50.0}, "converted": {"mean": 0.2, "std": 0.4, "min": 0.0, "max": 1.0}},
        "categorical": {
            "job": {"admin.": 1.0},
            "housing": {"no": 1.0},
            "loan": {"no": 1.0},
            "segment": {"adulto_sem_credito": 1.0},
            "arm": {"cellular": 0.9, "telephone": 0.1},
        },
    }
    atual = pd.DataFrame(
        {
            "age": [41, 39, 42],
            "job": ["admin.", "admin.", "admin."],
            "housing": ["no", "no", "no"],
            "loan": ["no", "no", "no"],
            "segment": ["adulto_sem_credito"] * 3,
            "arm": ["telephone", "telephone", "telephone"],
            "converted": [0, 0, 1],
        }
    )

    relatorio = detect_drift(atual, reference_stats=reference)

    assert relatorio["drift_detected"] is True
    arm_finding = next(item for item in relatorio["findings"] if item["feature"] == "arm")
    assert arm_finding["metric"] == "psi"
    assert arm_finding["drifted"] is True
