import pandas as pd
import pytest

from src.train import cross_validate_policy


def test_validacao_cruzada_calcula_media_de_cinco_folds():
    eventos = pd.DataFrame(
        {
            "arm": ["telephone"] * 10,
            "converted": [0, 1] * 5,
            "segment": ["segmento"] * 10,
        }
    )

    metricas, folds = cross_validate_policy(
        "baseline_fixed", ["telephone"], eventos, n_splits=5
    )

    assert len(folds) == 5
    assert [fold["fold"] for fold in folds] == [1, 2, 3, 4, 5]
    assert metricas["conversion_rate"] == pytest.approx(0.5)
    assert metricas["match_rate"] == pytest.approx(1.0)
    assert metricas["matched_events"] == pytest.approx(2.0)
    assert metricas["conversions"] == pytest.approx(1.0)