"""Contratos de segurança, viés e interpretabilidade da política.

O bandit só pode usar o contexto minimizado (faixa etária × crédito).
Atributos sensíveis do log original não entram na Feature Store nem na API.
"""
from __future__ import annotations

SENSITIVE_COLUMNS = {
    "marital",
    "education",
    "default",
    "duration",
    "pdays",
    "nr.employed",
    "emp.var.rate",
}
ALLOWED_MODEL_FEATURES = {"age", "job", "housing", "loan", "segment", "arm", "converted"}
ALLOWED_API_FIELDS = {"age", "job", "housing", "loan"}


def leaked_sensitive_columns(columns) -> set[str]:
    return set(columns) & SENSITIVE_COLUMNS


def unauthorized_model_columns(columns) -> set[str]:
    return set(columns) - ALLOWED_MODEL_FEATURES


def explain_decision(policy, segment: str, recommended_arm: str, posterior: dict) -> dict:
    """Explicação local da decisão: evidência Beta por braço no segmento."""
    ranked = sorted(
        posterior.items(),
        key=lambda item: item[1]["mean_conversion"],
        reverse=True,
    )
    best_arm, best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else ranked[0]
    return {
        "policy": type(policy).__name__,
        "context_features": ["segment"],
        "segment": segment,
        "recommended_arm": recommended_arm,
        "reason": (
            f"No segmento {segment}, {best_arm} tem média posterior "
            f"{best['mean_conversion']:.4f} contra {runner_up[0]} "
            f"com {runner_up[1]['mean_conversion']:.4f}."
        ),
        "evidence": {
            arm: {
                "alpha": values["alpha"],
                "beta": values["beta"],
                "mean_conversion": values["mean_conversion"],
            }
            for arm, values in posterior.items()
        },
    }


def conversion_gap_by_segment(df, segment_col: str = "segment", reward_col: str = "converted") -> dict:
    """Auditoria de disparidade: max-min da taxa de conversão entre segmentos."""
    rates = df.groupby(segment_col)[reward_col].mean()
    if rates.empty:
        return {"n_segments": 0, "gap": 0.0, "rates": {}}
    return {
        "n_segments": int(rates.size),
        "gap": float(rates.max() - rates.min()),
        "rates": {str(k): float(v) for k, v in rates.items()},
    }
