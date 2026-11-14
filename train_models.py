import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from build_your_own_features_example import build_feature_matrix


ANTICANCER_TRAIN = "participant_datasets/initial/anticancer_train.csv"
HEMOLYTIC_TRAIN = "participant_datasets/initial/hemolytic_train.csv"

def run_pipeline(name: str, path: str) -> None:
    print(f"\n--- {name} ---")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run make_demo_data.py first (organizer step) "
              f"or check you're in the right directory.")
        sys.exit(1)

    df = pd.read_csv(path)
    assert "sequence" in df.columns and "label" in df.columns, f"{path} missing sequence/label columns"
    print(f"Loaded {len(df)} sequences")

    X = build_feature_matrix(df["sequence"])
    y = df["label"].values
    print(f"Feature matrix shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    print(f"Accuracy   : {acc:.4f}")
    print(f"AUC        : {auc:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")

    # The printed scores come from the held-out rows. The saved model is
    # refit on every row so the final test peptides see all of the lab data.
    final_model = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    final_model.fit(X, y)
    joblib.dump(final_model, f"{name}_model.joblib")
    print(f"Saved {name}_model.joblib (fit on all {len(df)} rows)")


def main():
    print("Testing environment on fake/demo data (synthetic - no real signal)...")
    run_pipeline("Anticancer train", ANTICANCER_TRAIN)
    run_pipeline("Hemolytic train", HEMOLYTIC_TRAIN)
    print("\nIf both blocks printed metrics with no errors, your environment is ready.")
    print("Now go work with the real data in participant_datasets/initial/.")


if __name__ == "__main__":
    main()