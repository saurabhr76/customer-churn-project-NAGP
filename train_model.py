"""
train_model.py
==============
Reproducible, end-to-end training script for the Telco churn model.

Run:  python train_model.py

It performs exactly the same steps as notebook/churn_analysis.ipynb (load ->
clean -> split -> build pipeline -> tune -> evaluate -> save) and writes
`model/churn_model.pkl`, the single artifact the API loads.

The saved artifact is a *bundle* (dict), not a bare estimator:
    pipeline        fitted sklearn Pipeline (feature engineering + preprocessing + tree)
    threshold       decision threshold the API applies to the churn probability
    metrics         hold-out metrics, for the /health and /model-info endpoints
    input_columns   the exact raw fields the API must receive
    trained_at, sklearn_version, target_mapping
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

import churn_features as cf

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.30
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "TelcoCustomerChurn.csv")
MODEL_PATH = os.path.join(HERE, "model", "churn_model.pkl")

# Final hyper-parameters, selected in the notebook by 5-fold GridSearchCV
# (scoring = ROC-AUC, class_weight balanced to handle the 73/27 imbalance).
FINAL_PARAMS = dict(
    criterion="gini",
    max_depth=6,
    min_samples_leaf=80,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

#: Threshold applied to P(churn). 0.50 is the default; the notebook's cost
#: analysis also reports a cheaper operating point. Kept in the artifact so the
#: business can re-tune it without retraining.
DECISION_THRESHOLD = 0.50


def build_pipeline(classifier) -> Pipeline:
    """Feature engineering -> imputation + one-hot -> classifier, as ONE object.

    Everything that learns from data (imputation medians, the one-hot
    vocabulary) lives inside the pipeline, so `fit` only ever sees the training
    fold. That is what makes the whole thing leakage-free and re-usable on
    unseen data with a single `.predict()` call.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), cf.NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cf.CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("features", FunctionTransformer(cf.get_feature_frame, validate=False)),
            ("preprocess", preprocessor),
            ("model", classifier),
        ]
    )


def main() -> None:
    print("1/6  Loading data ...")
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[cf.TARGET])
    y = (df[cf.TARGET] == "Yes").astype(int)
    print(f"     rows={len(df)}  churn rate={y.mean():.2%}")

    print("2/6  Train/test split (70:30, stratified, random_state=42) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"     train={X_train.shape}  test={X_test.shape}")

    print("3/6  Building and fitting pipeline ...")
    pipe = build_pipeline(DecisionTreeClassifier(**FINAL_PARAMS))
    pipe.fit(X_train, y_train)

    print("4/6  Cross-validating on the training set ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"     CV F1  = {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
    print(f"     CV AUC = {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})")

    print("5/6  Evaluating on the hold-out test set ...")
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= DECISION_THRESHOLD).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "cv_f1_mean": round(float(cv_f1.mean()), 4),
        "cv_roc_auc_mean": round(float(cv_auc.mean()), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    print(json.dumps({k: v for k, v in metrics.items()}, indent=2))
    print(classification_report(y_test, pred, target_names=["No Churn", "Churn"]))

    print("6/6  Saving artifact ...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    bundle = {
        "pipeline": pipe,
        "threshold": DECISION_THRESHOLD,
        "metrics": metrics,
        "input_columns": cf.MODEL_INPUT_COLUMNS,
        "target_mapping": {0: "No", 1: "Yes"},
        "model_params": FINAL_PARAMS,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": __import__("sklearn").__version__,
        "model_version": "1.0.0",
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)
    size_kb = os.path.getsize(MODEL_PATH) / 1024
    print(f"     saved -> {MODEL_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
