import json

import pandas as pd

from src import hyperparam_search, train


def test_busca_escolhe_e_persiste_o_melhor_candidato(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(
        hyperparam_search,
        "SEARCH_SPACE",
        {
            "epsilon_greedy": [{"epsilon": 0.05}, {"epsilon": 0.2}],
            "thompson_sampling": [{"prior_alpha": 1.0, "prior_beta": 1.0}],
            "thompson_sampling_contextual": [{"prior_alpha": 1.0, "prior_beta": 1.0}],
        },
    )
    eventos = pd.DataFrame(
        {
            "arm": ["cellular", "telephone"] * 6,
            "converted": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "segment": ["adulto_sem_credito"] * 12,
        }
    )

    resultados = hyperparam_search.search_hyperparameters(eventos, n_splits=3)
    escolhidos = json.loads((tmp_path / "reports" / "best_hyperparams.json").read_text(encoding="utf-8"))

    assert set(resultados["policy"]) >= {"epsilon_greedy", "thompson_sampling", "thompson_sampling_contextual"}
    assert "epsilon_greedy" in escolhidos
    assert escolhidos["epsilon_greedy"]["epsilon"] in {0.05, 0.2}
    assert train.load_selected_hyperparams()["epsilon_greedy"]["epsilon"] == escolhidos["epsilon_greedy"]["epsilon"]
