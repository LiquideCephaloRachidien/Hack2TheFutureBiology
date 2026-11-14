"""
Score peptides with the two saved models and write the submission CSV.

This script does not train. It loads the models written by train_models.py,
turns each sequence into the same features, and writes one row per peptide.
"""

import joblib
import pandas as pd

from build_your_own_features_example import build_feature_matrix

# Swap this for final_test_sequences.csv when that file arrives.
INPUT_PATH = "final_test_sequences.csv"
OUTPUT_PATH = "submission.csv"

ANTICANCER_MODEL = "Anticancer train_model.joblib"
HEMOLYTIC_MODEL = "Hemolytic train_model.joblib"

# A probability at or above this becomes a prediction of 1.
THRESHOLD = 0.5


def verdict(anticancer_prediction, hemolytic_prediction):
    if anticancer_prediction == 1 and hemolytic_prediction == 0:
        return "Selective"
    if anticancer_prediction == 1 and hemolytic_prediction == 1:
        return "Works but toxic"
    if anticancer_prediction == 0 and hemolytic_prediction == 1:
        return "Toxic and doesn't work"
    return "Does nothing"


def main():
    df = pd.read_csv(INPUT_PATH) #final_test_sequences.csv, then run the usual command
    X = build_feature_matrix(df["sequence"])

    anticancer_model = joblib.load(ANTICANCER_MODEL)
    hemolytic_model = joblib.load(HEMOLYTIC_MODEL)

    anticancer_probability = anticancer_model.predict_proba(X)[:, 1]
    hemolytic_probability = hemolytic_model.predict_proba(X)[:, 1]

    anticancer_prediction = (anticancer_probability >= THRESHOLD).astype(int)
    hemolytic_prediction = (hemolytic_probability >= THRESHOLD).astype(int)

    out = pd.DataFrame({
        "sequence": df["sequence"],
        "anticancer_probability": anticancer_probability,
        "anticancer_prediction": anticancer_prediction,
        "hemolytic_probability": hemolytic_probability,
        "hemolytic_prediction": hemolytic_prediction,
    })
    out["verdict"] = [
        verdict(a, h)
        for a, h in zip(out["anticancer_prediction"], out["hemolytic_prediction"])
    ]

    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(out)} rows to {OUTPUT_PATH}")
    print(out["verdict"].value_counts().to_string())


if __name__ == "__main__":
    main()
