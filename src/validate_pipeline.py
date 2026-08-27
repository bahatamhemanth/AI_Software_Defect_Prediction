"""Run end-to-end prediction and prioritization checks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_preprocessing import load_raw_dataset, prepare_model_data  # noqa: E402
from prediction import predict_modules  # noqa: E402
from prioritization import prioritize_modules  # noqa: E402


def main() -> None:
    raw = load_raw_dataset()
    metrics, _ = prepare_model_data(raw)
    sample = metrics.head(1)
    single = predict_modules(sample)
    batch = predict_modules(metrics.head(10))
    ranked = prioritize_modules(batch, metrics.head(10))
    assert len(single) == 1
    assert len(batch) == 10
    assert len(ranked) == 10
    assert ranked["Priority Rank"].tolist() == list(range(1, 11))
    assert ranked["Priority Score"].notna().all()
    print("Single prediction: OK")
    print("Batch prediction: OK")
    print("Prioritization ranking: OK")
    print(single.to_string(index=False))
    print(ranked[["Priority Rank", "Module", "Defect Probability", "Risk Level", "Priority Score"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
