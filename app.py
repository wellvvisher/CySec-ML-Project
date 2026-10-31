import os
import io
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

load_dotenv()
st.set_page_config(page_title="Explainable Malware Detection", page_icon="🛡️", layout="wide")

FEATURES = [
    "file_size", "num_sections", "num_imported_dlls", "num_imported_functions",
    "section_entropy", "has_writable_executable_section", "virtual_raw_size_ratio",
    "unusual_section_name_count", "header_inconsistency_count"
]
LABEL = "label"

@st.cache_data
def load_demo_data():
    path = os.path.join("data", "demo_training_data.csv")
    return pd.read_csv(path)

def build_models():
    return {
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced"))
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced"))
        ])
    }

def train_models(df):
    missing = [c for c in FEATURES + [LABEL] if c not in df.columns]
    if missing:
        raise ValueError("Training CSV is missing columns: " + ", ".join(missing))
    work = df[FEATURES + [LABEL]].copy().dropna(subset=[LABEL])
    if work[LABEL].nunique() < 2:
        raise ValueError("Training data must contain at least two classes in the 'label' column.")
    X = work[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = work[LABEL].astype(str)
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )
    models = build_models()
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, pred, average="weighted", zero_division=0),
            "y_test": y_test,
            "pred": pred,
            "report": classification_report(y_test, pred, zero_division=0, output_dict=True)
        }
    return results, X.columns.tolist()

def gemini_explanation(payload):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None, "Gemini is not configured. Add GEMINI_API_KEY to your .env file to enable AI explanations."
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        prompt = (
            "You are assisting a student malware-analysis project. Explain the following static-feature "
            "machine-learning result in plain language. Do not claim the file is definitely malicious. "
            "Mention uncertainty, possible false positives, and that static features alone are not proof. "
            "Do not invent facts beyond the supplied data. Keep the answer structured with headings.\n\n"
            + json.dumps(payload, indent=2, default=str)
        )
        response = client.models.generate_content(model="gemini-3.8-flash", contents=prompt)
        return response.text, None
    except Exception as exc:
        return None, f"Gemini request failed: {exc}. The ML results are still available."

st.title("🛡️ Explainable Malware Detection System")
st.caption("Project 9 · Static PE-feature classification · ML + optional Google Gemini explanation")
st.warning("Educational prototype only. The bundled dataset is synthetic demonstration data, not a validated malware dataset. Never use this app's prediction as the sole basis for a security decision.")

with st.sidebar:
    st.header("Data & settings")
    st.write("Required numeric feature columns:")
    st.code(", ".join(FEATURES), language="text")
    st.write("Training label column: `label` (for example: `benign`, `suspicious`).")
    uploaded_train = st.file_uploader("Optional: upload labelled training CSV", type=["csv"], key="train_csv")
    st.caption("If no training CSV is uploaded, the bundled synthetic demo dataset is used.")
    train_button = st.button("Train / evaluate models", type="primary", use_container_width=True)

if "trained" not in st.session_state:
    st.session_state.trained = False
if train_button or not st.session_state.trained:
    try:
        training_df = pd.read_csv(uploaded_train) if uploaded_train is not None else load_demo_data()
        results, feature_names = train_models(training_df)
        st.session_state.results = results
        st.session_state.feature_names = feature_names
        st.session_state.training_source = "Uploaded labelled CSV" if uploaded_train is not None else "Bundled synthetic demo dataset"
        st.session_state.trained = True
    except Exception as exc:
        st.error(f"Could not train models: {exc}")
        st.stop()

results = st.session_state.results
st.subheader("1. Model training and evaluation")
st.info(f"Training source: **{st.session_state.training_source}**. The held-out test split is used for these metrics.")
metric_cols = st.columns(4)
best_name = max(results, key=lambda n: results[n]["f1"])
for col, key, label in zip(metric_cols, ["accuracy", "precision", "recall", "f1"], ["Accuracy", "Precision", "Recall", "F1-score"]):
    col.metric(label, f"{results[best_name][key]:.3f}", help=f"{best_name} weighted score on the held-out split")
st.caption(f"Best model by weighted F1 on this split: **{best_name}**. Small or synthetic datasets can produce misleadingly high scores.")
metric_df = pd.DataFrame([
    {"Model": name, "Accuracy": r["accuracy"], "Precision": r["precision"], "Recall": r["recall"], "F1-score": r["f1"]}
    for name, r in results.items()
]).set_index("Model")
st.dataframe(metric_df.style.format("{:.3f}"), use_container_width=True)

selected_model = st.selectbox("Choose model for prediction", list(results.keys()), index=list(results.keys()).index(best_name))
chosen = results[selected_model]["model"]
with st.expander("View confusion matrix and class report"):
    y_test = results[selected_model]["y_test"]
    y_pred = results[selected_model]["pred"]
    labels = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=25, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    st.pyplot(fig)
    st.dataframe(pd.DataFrame(results[selected_model]["report"]).transpose().round(3), use_container_width=True)

st.divider()
st.subheader("2. Upload CSV for analysis")
st.write("Upload one or more rows containing the nine numeric feature columns above. The `label` column is not required for prediction.")
uploaded_predict = st.file_uploader("Choose prediction CSV", type=["csv"], key="predict_csv")
if uploaded_predict is not None:
    try:
        pred_df = pd.read_csv(uploaded_predict)
        missing = [c for c in FEATURES if c not in pred_df.columns]
        if missing:
            st.error("Prediction CSV is missing columns: " + ", ".join(missing))
        else:
            X_pred = pred_df[FEATURES].apply(pd.to_numeric, errors="coerce")
            predictions = chosen.predict(X_pred)
            output = pred_df.copy()
            output["prediction"] = predictions
            if hasattr(chosen, "predict_proba"):
                probs = chosen.predict_proba(X_pred)
                output["model_score_for_predicted_class"] = probs.max(axis=1).round(4)
            st.dataframe(output, use_container_width=True)
            st.download_button("Download predictions CSV", output.to_csv(index=False).encode("utf-8"), "malware_predictions.csv", "text/csv")

            # Transparent model-level feature importance for the selected model.
            fitted_model = chosen.named_steps["model"]
            if hasattr(fitted_model, "feature_importances_"):
                importance = pd.DataFrame({"Feature": FEATURES, "Importance": fitted_model.feature_importances_}).sort_values("Importance", ascending=False)
                st.subheader("3. What influenced the model overall?")
                st.bar_chart(importance.set_index("Feature"))
                st.caption("These are global model feature importances, not proof that any single feature caused an individual prediction.")

            st.subheader("4. Google Gemini explanation (optional)")
            st.write("Gemini explains the model output and supplied feature values; it does not perform the classifier's prediction.")
            row_index = st.selectbox("Select row to explain", list(range(len(output))), format_func=lambda i: f"Row {i + 1}")
            if st.button("Explain selected result with Gemini"):
                payload = {
                    "model": selected_model,
                    "prediction": str(predictions[row_index]),
                    "model_score_for_predicted_class": float(output.iloc[row_index].get("model_score_for_predicted_class", 0)),
                    "features": {k: (None if pd.isna(v) else float(v)) for k, v in X_pred.iloc[row_index].items()},
                    "global_feature_importance": importance.head(5).to_dict(orient="records") if "importance" in locals() else []
                }
                with st.spinner("Requesting explanation from Gemini..."):
                    explanation, error = gemini_explanation(payload)
                if error:
                    st.warning(error)
                if explanation:
                    st.markdown(explanation)
    except Exception as exc:
        st.error(f"Could not process CSV: {exc}")

st.divider()
st.caption("Limitations: this prototype uses static features only; it does not execute or dynamically analyze uploaded executables. Validate on a real, appropriately licensed labelled dataset before making any operational claims.")
