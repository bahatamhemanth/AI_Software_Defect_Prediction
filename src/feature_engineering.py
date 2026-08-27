"""Feature-set decisions derived from PC1 EDA."""

from data_preprocessing import DROP_FEATURES


def selected_features(all_features: list[str]) -> list[str]:
    """Return the reproducible predictor set after removing exact redundancy."""
    return [feature for feature in all_features if feature not in DROP_FEATURES]
