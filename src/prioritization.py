"""Risk-based module test prioritization."""

import pandas as pd

from prediction import risk_level

COMPLEXITY_FEATURES = ["loc", "v(g)", "N", "V", "branchCount"]


def prioritize_modules(predictions: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank modules using probability as primary signal and normalized complexity."""
    missing = sorted(set(COMPLEXITY_FEATURES) - set(metrics.columns))
    if missing:
        raise ValueError(f"Missing complexity metrics: {', '.join(missing)}")
    complexity = metrics[COMPLEXITY_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    ranges = complexity.max() - complexity.min()
    normalized = (complexity - complexity.min()).div(ranges.replace(0, 1))
    complexity_index = normalized.mean(axis=1)
    ranked = predictions.copy()
    ranked["Complexity Index"] = complexity_index.reindex(ranked.index).fillna(0)
    ranked["Priority Score"] = ranked["Defect Probability"] * (0.5 + 0.5 * ranked["Complexity Index"])
    ranked["Risk Level"] = ranked["Defect Probability"].map(risk_level)
    ranked = ranked.sort_values(["Priority Score", "Defect Probability"], ascending=False)
    ranked["Priority Rank"] = range(1, len(ranked) + 1)
    return ranked.reset_index(drop=True)
