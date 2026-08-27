"""Streamlit demonstration for PC1 defect prediction and test prioritization."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_preprocessing import load_raw_dataset  # noqa: E402
from prediction import MODEL_PATH, predict_modules  # noqa: E402
from prioritization import prioritize_modules  # noqa: E402

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

st.set_page_config(page_title="PC1 Defect Prediction", page_icon="🔎", layout="wide")
st.title("AI-Based Software Defect Prediction")
st.caption("NASA MDP PC1 | Risk-based module testing")

if not MODEL_PATH.exists():
    st.error("No trained model found. Run src/train_models.py first.")
    st.stop()

bundle = __import__("prediction").load_bundle()
raw = load_raw_dataset()
results = pd.read_csv(REPORTS / "model_results.csv")
best_name = bundle["model_name"]

with st.sidebar:
    st.header("Project overview")
    st.write("Predict defect risk from historical software metrics and prioritize modules for testing.")
    st.metric("Modules", f"{len(raw):,}")
    st.metric("Defect rate", f"{raw['defects'].mean() * 100:.2f}%")
    st.metric("Predictor features", len(bundle["features"]))
    st.write(f"Selected model: **{best_name}**")

st.header("Defect prediction and test prioritization")
source = st.file_uploader("Upload a CSV containing PC1 software metrics", type="csv")
if source is not None:
    uploaded = pd.read_csv(source)
    module_names = uploaded.pop("Module") if "Module" in uploaded.columns else pd.Series(uploaded.index.astype(str), name="Module")
    uploaded.index = module_names.astype(str)
    try:
        predictions = predict_modules(uploaded)
        prioritized = prioritize_modules(predictions, uploaded)
        st.subheader("Predictions")
        st.dataframe(predictions.reset_index(drop=True), use_container_width=True)
        st.subheader("Testing priority")
        st.dataframe(prioritized[["Priority Rank", "Module", "Defect Probability", "Risk Level", "Priority Score"]], use_container_width=True)
        st.download_button("Download prioritization CSV", prioritized.to_csv(index=False), "pc1_test_priorities.csv", "text/csv")
    except (ValueError, KeyError) as error:
        st.error(str(error))
else:
    st.info("Upload a CSV to generate predictions. The file must contain the trained PC1 metric columns.")

st.header("Model performance")
st.dataframe(results, use_container_width=True)
col1, col2 = st.columns(2)
with col1:
    st.image(str(FIGURES / "model_curves.png"), caption="ROC and precision-recall curves")
with col2:
    st.image(str(FIGURES / "feature_importance.png"), caption="Selected model feature importance")
