"""Train and evaluate the two baseline models on a labelled feature CSV."""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

FEATURES = [
    "file_size", "num_sections", "num_imported_dlls", "num_imported_functions",
    "section_entropy", "has_writable_executable_section", "virtual_raw_size_ratio",
    "unusual_section_name_count", "header_inconsistency_count"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/demo_training_data.csv", help="CSV with numeric feature columns and a label column")
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    missing = [c for c in FEATURES + ["label"] if c not in df.columns]
    if missing:
        raise SystemExit("Missing required columns: " + ", ".join(missing))
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df["label"].astype(str)
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42, stratify=stratify)
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced")
    }
    for name, estimator in models.items():
        model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print(f"\n{name}")
        print(f"Accuracy:  {accuracy_score(y_test, pred):.3f}")
        print(f"Precision: {precision_score(y_test, pred, average='weighted', zero_division=0):.3f}")
        print(f"Recall:    {recall_score(y_test, pred, average='weighted', zero_division=0):.3f}")
        print(f"F1-score:  {f1_score(y_test, pred, average='weighted', zero_division=0):.3f}")
        print("Confusion matrix (label order:", sorted(y.unique()), ")")
        print(confusion_matrix(y_test, pred, labels=sorted(y.unique())))

if __name__ == "__main__":
    main()
