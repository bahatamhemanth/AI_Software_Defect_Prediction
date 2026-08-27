"""Train, compare, and persist defect-prediction pipelines."""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_preprocessing import (  # noqa: E402
    build_preprocessor,
    load_raw_dataset,
    prepare_model_data,
    preprocessing_summary,
)

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"


def make_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=SEED),
    }


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_raw_dataset()
    X, y = prepare_model_data(data)
    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"average_precision": "average_precision", "recall": "recall", "f1": "f1"}
    results = []
    fitted = {}
    for name, estimator in make_models().items():
        pipeline = Pipeline([("preprocessor", build_preprocessor(feature_names)), ("model", estimator)])
        cv_scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        pipeline.fit(X_train, y_train)
        probability = pipeline.predict_proba(X_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        results.append({
            "model": name,
            "cv_average_precision": cv_scores["test_average_precision"].mean(),
            "cv_recall": cv_scores["test_recall"].mean(),
            "cv_f1": cv_scores["test_f1"].mean(),
            "accuracy": accuracy_score(y_test, prediction),
            "precision": precision_score(y_test, prediction, zero_division=0),
            "recall": recall_score(y_test, prediction, zero_division=0),
            "f1": f1_score(y_test, prediction, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probability),
            "pr_auc": average_precision_score(y_test, probability),
        })
        fitted[name] = (pipeline, probability, prediction)
    result_df = pd.DataFrame(results).sort_values(
        ["cv_recall", "cv_average_precision", "cv_f1"], ascending=False
    )
    result_df.to_csv(REPORT_DIR / "model_results.csv", index=False)
    best_name = result_df.iloc[0]["model"]
    best_pipeline, best_probability, best_prediction = fitted[best_name]
    joblib.dump({"pipeline": best_pipeline, "features": feature_names, "model_name": best_name}, MODEL_DIR / "best_model.joblib")
    model = best_pipeline.named_steps["model"]
    transformed_features = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    if hasattr(model, "coef_"):
        importance_values = model.coef_[0]
    elif hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
    else:
        importance_values = np.zeros(len(transformed_features))
    importance = pd.DataFrame({"feature": transformed_features, "importance": importance_values})
    importance["absolute_importance"] = importance["importance"].abs()
    importance.sort_values("absolute_importance", ascending=False).to_csv(REPORT_DIR / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    importance.sort_values("absolute_importance").tail(12).plot.barh(x="feature", y="importance", ax=ax, legend=False, color="#2a6f97")
    ax.set_title(f"Feature Importance: {best_name}")
    ax.set_xlabel("Model coefficient or importance")
    ax.set_ylabel("Software metric")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "feature_importance.png", bbox_inches="tight")
    plt.close(fig)
    with open(REPORT_DIR / "preprocessing_summary.json", "w", encoding="utf-8") as handle:
        json.dump(preprocessing_summary(data), handle, indent=2)
    cm = confusion_matrix(y_test, best_prediction)
    pd.DataFrame(cm, index=["Actual non-defective", "Actual defective"], columns=["Predicted non-defective", "Predicted defective"]).to_csv(REPORT_DIR / "confusion_matrix.csv")
    fpr, tpr, _ = roc_curve(y_test, best_probability)
    precision, recall, _ = precision_recall_curve(y_test, best_probability)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, label=f"{best_name} (AUC={roc_auc_score(y_test, best_probability):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set(title="ROC Curve", xlabel="False positive rate", ylabel="True positive rate")
    axes[0].legend()
    axes[1].plot(recall, precision, label=f"{best_name} (AP={average_precision_score(y_test, best_probability):.3f})")
    axes[1].set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_curves.png", bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    result_df.set_index("model")[["recall", "f1", "pr_auc", "roc_auc"]].plot(kind="bar", ax=ax)
    ax.set_title("Model Comparison on the Held-Out Test Set")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_comparison.png", bbox_inches="tight")
    plt.close(fig)
    report = [
        "# Model Evaluation",
        "",
        "Model selection used five-fold stratified cross-validation on the training set, ordered by defective-class recall, then mean average precision and F1. This reflects the cost of missing a defective module. The held-out test set was used once for final reporting.",
        "",
        f"Selected model: **{best_name}**",
        "",
        result_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "",
        "Class weights were used only on estimators that support them. No resampling was applied. Duplicate rows were removed before the stratified split, and `T` was excluded because EDA found it perfectly correlated with `E`.",
    ]
    (REPORT_DIR / "model_evaluation.md").write_text("\n".join(report), encoding="utf-8")
    print(result_df.to_string(index=False))
    print(f"BEST_MODEL={best_name}")


if __name__ == "__main__":
    main()
