from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _build_pipeline(feature_cols: list[str]) -> Pipeline:
    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    pre = ColumnTransformer(
        transformers=[("num", numeric, feature_cols)],
        remainder="drop",
    )

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")

    return Pipeline(steps=[("preprocess", pre), ("clf", clf)])


def train_and_score(
    df: pd.DataFrame,
    target_col: str,
    id_col: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    feature_cols = [c for c in df.columns if c not in {target_col, id_col}]

    X = df[feature_cols]
    y = df[target_col].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = _build_pipeline(feature_cols)
    model.fit(X_train, y_train)

    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba_test)),
        "accuracy": float(accuracy_score(y_test, pred_test)),
        "precision": float(precision_score(y_test, pred_test, zero_division=0)),
        "recall": float(recall_score(y_test, pred_test, zero_division=0)),
        "f1": float(f1_score(y_test, pred_test, zero_division=0)),
    }

    churn_score = model.predict_proba(X)[:, 1]
    scored = df[[id_col, target_col]].copy()
    scored["churn_score"] = churn_score

    return scored, metrics


def save_metrics(metrics: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
