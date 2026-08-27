"""Prediction helpers backed by the persisted trained pipeline."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"


def risk_level(probability: float) -> str:
    if probability < 0.33:
        return "Low"
    if probability < 0.66:
        return "Medium"
    return "High"


def load_bundle(path: Path = MODEL_PATH) -> dict[str, Any]:
    return joblib.load(path)


def predict_modules(module_metrics: pd.DataFrame, model_path: Path = MODEL_PATH) -> pd.DataFrame:
    bundle = load_bundle(model_path)
    expected_features = bundle["features"]
    missing = sorted(set(expected_features) - set(module_metrics.columns))
    if missing:
        raise ValueError(f"Missing required software metrics: {', '.join(missing)}")
    features = module_metrics[expected_features]
    probabilities = bundle["pipeline"].predict_proba(features)[:, 1]
    return pd.DataFrame(
        {
            "Module": module_metrics.index.astype(str),
            "Defect Prediction": ["Defective" if value >= 0.5 else "Non-defective" for value in probabilities],
            "Defect Probability": probabilities,
            "Risk Level": [risk_level(value) for value in probabilities],
        },
        index=module_metrics.index,
    )
