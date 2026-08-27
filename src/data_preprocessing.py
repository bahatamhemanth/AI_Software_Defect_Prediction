"""Leakage-safe loading and preprocessing helpers for NASA PC1."""

from pathlib import Path
from typing import Tuple

import pandas as pd
from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "pc1.arff"
TARGET = "defects"
DROP_FEATURES = ["T"]


def load_raw_dataset(path: Path = RAW_PATH) -> pd.DataFrame:
    records, _ = arff.loadarff(path)
    data = pd.DataFrame(records)
    for column in data.select_dtypes(include="object").columns:
        data[column] = data[column].str.decode("utf-8")
    data[TARGET] = data[TARGET].astype(str).str.lower().map({"true": 1, "false": 0})
    return data


def prepare_model_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    clean_data = data.replace([float("inf"), float("-inf")], pd.NA).drop_duplicates().copy()
    features = [column for column in clean_data.columns if column not in [TARGET, *DROP_FEATURES]]
    return clean_data[features], clean_data[TARGET].astype(int)


def build_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, feature_names)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def preprocessing_summary(data: pd.DataFrame) -> dict:
    duplicate_count = int(data.duplicated().sum())
    return {
        "raw_rows": int(len(data)),
        "duplicate_rows_removed": duplicate_count,
        "rows_after_duplicate_removal": int(len(data) - duplicate_count),
        "raw_features": int(len(data.columns) - 1),
        "dropped_features": DROP_FEATURES,
        "reason_for_dropped_features": "T is perfectly correlated with E in the observed PC1 data.",
        "missing_values": int(data.isna().sum().sum()),
        "infinite_values": int(data.select_dtypes("number").isin([float("inf"), float("-inf")]).sum().sum()),
        "outlier_policy": "Retain plausible extreme software metrics; scaling is performed in the fitted pipeline.",
        "imbalance_policy": "Use balanced class weights on supported estimators; no resampling is applied.",
    }
