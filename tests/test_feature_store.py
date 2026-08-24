import pandas as pd

from src import feature_store


def test_feature_store_publica_contrato_e_estatisticas(tmp_path, monkeypatch):
    monkeypatch.setattr(feature_store, "FEATURE_STORE_DIR", tmp_path)
    eventos = pd.DataFrame(
        {
            "age": [28, 61],
            "job": ["student", "retired"],
            "housing": ["no", "no"],
            "loan": ["no", "no"],
            "segment": ["jovem_sem_credito", "senior_sem_credito"],
            "arm": ["cellular", "telephone"],
            "converted": [1, 0],
            "duration": [120, 50],
        }
    )

    metadata = feature_store.publish_features(eventos)
    view = feature_store.load_features()

    assert metadata["feature_view"] == "campaign_context"
    assert metadata["version"] == 1
    assert metadata["n_rows"] == 2
    assert "duration" not in view.columns
    assert set(view.columns) == set(feature_store.FEATURE_COLUMNS)
    assert "age" in metadata["statistics"]["numeric"]
    assert "segment" in metadata["statistics"]["categorical"]
