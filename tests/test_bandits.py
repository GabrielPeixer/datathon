import json

import pandas as pd
import pytest

from src.bandits import BaselinePolicy, EpsilonGreedy, ThompsonSampling, load_policy, replay_evaluation


@pytest.mark.parametrize(
    ("policy", "attribute", "expected"),
    [
        (BaselinePolicy(["a", "b"], fixed_arm="b"), "fixed_arm", "b"),
        (EpsilonGreedy(["a", "b"], epsilon=0.25), "epsilon", 0.25),
        (ThompsonSampling(["a", "b"], prior_alpha=2, prior_beta=3), "prior_alpha", 2),
    ],
)
def test_politica_preserva_configuracao_ao_serializar(tmp_path, policy, attribute, expected):
    policy.update("a", 1, "segmento")
    caminho = tmp_path / "politica.json"

    policy.save(caminho)
    restaurada = load_policy(caminho)

    assert type(restaurada) is type(policy)
    assert getattr(restaurada, attribute) == expected
    assert restaurada.rewards["__global__"]["a"] == 1
    assert json.loads(caminho.read_text())[attribute] == expected


def test_replay_calcula_metricas_de_casamento():
    eventos = pd.DataFrame(
        {
            "arm": ["a", "b", "a"],
            "converted": [1, 0, 1],
            "segment": ["s", "s", "s"],
        }
    )

    metricas = replay_evaluation(BaselinePolicy(["a", "b"], fixed_arm="a"), eventos)

    assert metricas["matched_events"] == 2
    assert metricas["conversions"] == 2
    assert metricas["conversion_rate"] == 1
    assert metricas["match_rate"] == pytest.approx(2 / 3)


def test_replay_de_validacao_nao_atualiza_politica():
    eventos = pd.DataFrame(
        {"arm": ["a"], "converted": [1], "segment": ["s"]}
    )
    policy = BaselinePolicy(["a"], fixed_arm="a")

    metricas = replay_evaluation(policy, eventos, update_policy=False)

    assert metricas["conversions"] == 1
    assert policy.pulls["__global__"]["a"] == 0
    assert policy.rewards["__global__"]["a"] == 0
