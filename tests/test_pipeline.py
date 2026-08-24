from pathlib import Path

from src import data_prep, drift, feature_store, pipeline, train

FIXTURE = Path(__file__).parent / "fixtures" / "bank_sample.csv"


def test_esteira_processa_dados_drift_e_treino_em_diretorio_isolado(tmp_path, monkeypatch):
    monkeypatch.setattr(data_prep, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(data_prep, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(feature_store, "FEATURE_STORE_DIR", tmp_path / "feature_store")
    monkeypatch.setattr(drift, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path / "models")

    resultado = pipeline.run_pipeline(raw_path=str(FIXTURE), search=False, n_splits=3)

    assert resultado["rows"] == 12
    assert resultado["drift_detected"] is False
    assert (tmp_path / "reports" / "drift_report.json").exists()
    assert (tmp_path / "feature_store" / "campaign_context" / "metadata.json").exists()
    assert "thompson_sampling_contextual" in resultado["policies"]
    assert (tmp_path / "models" / "thompson_sampling_contextual.json").exists()
