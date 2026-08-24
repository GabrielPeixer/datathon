import json

import pandas as pd

from src import data_prep, feature_store


def test_preparo_remove_apenas_duplicatas_reais_e_duration(tmp_path, monkeypatch):
    bruto = pd.DataFrame(
        [
            {"age": 30, "loan": "no", "housing": "no", "contact": "cellular", "duration": 10, "y": "yes"},
            {"age": 30, "loan": "no", "housing": "no", "contact": "cellular", "duration": 20, "y": "no"},
            {"age": 30, "loan": "no", "housing": "no", "contact": "cellular", "duration": 10, "y": "yes"},
        ]
    )
    caminho_bruto = tmp_path / "bank-additional-full.csv"
    bruto.to_csv(caminho_bruto, sep=";", index=False)
    processados = tmp_path / "processed"
    monkeypatch.setattr(data_prep, "PROCESSED_DIR", processados)
    monkeypatch.setattr(feature_store, "FEATURE_STORE_DIR", tmp_path / "feature_store")
    monkeypatch.setattr(data_prep, "download_raw", lambda force=False: caminho_bruto)

    resultado = data_prep.prepare()

    assert len(resultado) == 2
    assert "duration" not in resultado.columns
    assert resultado["converted"].tolist() == [1, 0]
    manifesto = json.loads((processados / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifesto["linhas_processadas"] == 2
    assert len(manifesto["sha256"]) == 64
