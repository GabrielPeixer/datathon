"""Políticas de multi-armed bandit para seleção adaptativa de oferta/canal.

Políticas implementadas:
- BaselinePolicy: controle determinístico (sempre joga um braço fixo).
- EpsilonGreedy: explora com probabilidade epsilon, explota no restante.
- ThompsonSampling: exploração bayesiana com posteriores Beta(alpha, beta).

Todas as políticas aceitam um contexto opcional (segmento), mantendo uma
posterior/estimativa por par (segmento, braço) - um bandit contextual simples.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

GLOBAL_CONTEXT = "__global__"


class BanditPolicy:
    """Classe base: contabiliza jogadas e recompensas por (contexto, braço)."""

    def __init__(self, arms: list[str], contextual: bool = False, seed: int = 42):
        self.arms = list(arms)
        self.contextual = contextual
        self.rng = np.random.default_rng(seed)
        self.pulls: dict[str, dict[str, int]] = defaultdict(lambda: {a: 0 for a in self.arms})
        self.rewards: dict[str, dict[str, int]] = defaultdict(lambda: {a: 0 for a in self.arms})

    def _ctx(self, context: str | None) -> str:
        return context if (self.contextual and context) else GLOBAL_CONTEXT

    def select_arm(self, context: str | None = None) -> str:
        raise NotImplementedError

    def update(self, arm: str, reward: int, context: str | None = None) -> None:
        ctx = self._ctx(context)
        self.pulls[ctx][arm] += 1
        self.rewards[ctx][arm] += int(reward)

    def state(self) -> dict:
        return {
            "policy": type(self).__name__,
            "arms": self.arms,
            "contextual": self.contextual,
            "pulls": {c: dict(v) for c, v in self.pulls.items()},
            "rewards": {c: dict(v) for c, v in self.rewards.items()},
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state(), indent=2))

    def load_state(self, state: dict) -> None:
        for ctx, arms in state["pulls"].items():
            self.pulls[ctx].update(arms)
        for ctx, arms in state["rewards"].items():
            self.rewards[ctx].update(arms)


class BaselinePolicy(BanditPolicy):
    """Controle determinístico: sempre recomenda o mesmo braço fixo."""

    def __init__(self, arms: list[str], fixed_arm: str, **kwargs):
        super().__init__(arms, **kwargs)
        assert fixed_arm in self.arms
        self.fixed_arm = fixed_arm

    def select_arm(self, context: str | None = None) -> str:
        return self.fixed_arm

    def state(self) -> dict:
        state = super().state()
        state["fixed_arm"] = self.fixed_arm
        return state


class EpsilonGreedy(BanditPolicy):
    """Joga um braço aleatório com prob. epsilon; senão, o melhor braço empírico."""

    def __init__(self, arms: list[str], epsilon: float = 0.1, **kwargs):
        super().__init__(arms, **kwargs)
        self.epsilon = epsilon

    def select_arm(self, context: str | None = None) -> str:
        ctx = self._ctx(context)
        if self.rng.random() < self.epsilon:
            return str(self.rng.choice(self.arms))
        rates = {
            a: (self.rewards[ctx][a] / self.pulls[ctx][a]) if self.pulls[ctx][a] else 0.0
            for a in self.arms
        }
        best = max(rates.values())
        best_arms = [a for a, r in rates.items() if r == best]
        return str(self.rng.choice(best_arms))

    def state(self) -> dict:
        state = super().state()
        state["epsilon"] = self.epsilon
        return state


class ThompsonSampling(BanditPolicy):
    """Thompson Sampling Beta-Bernoulli.

    Prior: Beta(1, 1) - uniforme e não-informativa. Escolha documentada: sem
    histórico de negócio, assumimos que toda taxa de conversão em [0, 1] é
    igualmente provável; a posterior se concentra rapidamente com as evidências.
    """

    def __init__(self, arms: list[str], prior_alpha: float = 1.0, prior_beta: float = 1.0, **kwargs):
        super().__init__(arms, **kwargs)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def posterior(self, arm: str, context: str | None = None) -> tuple[float, float]:
        ctx = self._ctx(context)
        successes = self.rewards[ctx][arm]
        failures = self.pulls[ctx][arm] - successes
        return self.prior_alpha + successes, self.prior_beta + failures

    def select_arm(self, context: str | None = None) -> str:
        samples = {}
        for arm in self.arms:
            a, b = self.posterior(arm, context)
            samples[arm] = self.rng.beta(a, b)
        return max(samples, key=samples.get)

    def state(self) -> dict:
        s = super().state()
        s["prior_alpha"] = self.prior_alpha
        s["prior_beta"] = self.prior_beta
        return s


# Hiperparâmetros próprios de cada política, persistidos junto ao estado.
POLICY_PARAMS = ("fixed_arm", "epsilon", "prior_alpha", "prior_beta")


def load_policy(path: str | Path) -> BanditPolicy:
    """Reconstrói uma política salva a partir do seu arquivo de estado JSON."""
    state = json.loads(Path(path).read_text())
    cls = {
        "BaselinePolicy": BaselinePolicy,
        "EpsilonGreedy": EpsilonGreedy,
        "ThompsonSampling": ThompsonSampling,
    }[state["policy"]]
    params = {k: state[k] for k in POLICY_PARAMS if k in state}
    policy = cls(state["arms"], contextual=state["contextual"], **params)
    policy.load_state(state)
    return policy


def replay_evaluation(policy: BanditPolicy, df, arm_col="arm", reward_col="converted",
                      context_col="segment") -> dict:
    """Avaliação por replay offline (Li et al., 2011).

    Percorre os eventos do log; sempre que a política escolhe o mesmo braço que
    foi de fato jogado no histórico, a recompensa observada é contabilizada e a
    política é atualizada. Sem propensões conhecidas da política que gerou o log,
    o resultado é uma estimativa comparativa sujeita a viés de seleção.
    """
    matched, conversions = 0, 0
    history = []
    for row in df.itertuples(index=False):
        context = getattr(row, context_col)
        chosen = policy.select_arm(context)
        logged_arm = getattr(row, arm_col)
        if chosen == logged_arm:
            reward = int(getattr(row, reward_col))
            policy.update(chosen, reward, context)
            matched += 1
            conversions += reward
            history.append(conversions / matched)
    return {
        "matched_events": matched,
        "conversions": conversions,
        "conversion_rate": conversions / matched if matched else 0.0,
        "match_rate": matched / len(df) if len(df) else 0.0,
        "history": history,
    }
