from __future__ import annotations

import json
from typing import Tuple

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def evaluate(y_true, y_pred, y_prob) -> dict:
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def train_and_score(
    df: pd.DataFrame,
    target_col: str,
    id_col: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, dict]:
    X = df.drop(columns=[target_col, id_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000)),
        ]
    )
    lr.fit(X_train, y_train)

    lr_prob = lr.predict_proba(X_test)[:, 1]
    lr_pred = (lr_prob > 0.5).astype(int)
    lr_metrics = evaluate(y_test, lr_pred, lr_prob)

    cb = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        verbose=0,
        random_state=random_state,
    )
    cb.fit(X_train, y_train)

    cb_prob = cb.predict_proba(X_test)[:, 1]
    cb_pred = (cb_prob > 0.5).astype(int)
    cb_metrics = evaluate(y_test, cb_pred, cb_prob)

    print("\n=== MODEL COMPARISON ===")
    print("Logistic Regression:", lr_metrics)
    print("CatBoost:", cb_metrics)

    if cb_metrics["roc_auc"] > lr_metrics["roc_auc"]:
        print("👉 Using CatBoost")
        final_prob = cb.predict_proba(X)[:, 1]
        final_metrics = cb_metrics
        final_metrics["selected_model"] = "CatBoost"
    else:
        print("👉 Using Logistic Regression")
        final_prob = lr.predict_proba(X)[:, 1]
        final_metrics = lr_metrics
        final_metrics["selected_model"] = "Logistic Regression"

    scores = pd.DataFrame(
        {
            id_col: df[id_col].values,
            "churn_score": final_prob,
        }
    )

    return scores, final_metrics


def save_metrics(metrics: dict, path) -> None:
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)