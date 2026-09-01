from __future__ import annotations

import json
from typing import Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
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


def get_feature_importance(
    model,
    X: pd.DataFrame,
) -> pd.DataFrame:

    if hasattr(model, "named_steps"):
        model = model.named_steps["model"]

    if hasattr(model, "coef_"):
        importance = model.coef_[0]

    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_

    else:
        return pd.DataFrame()

    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": importance,
            }
        )
        .sort_values(
            by="importance",
            ascending=False,
        )
    )


def get_logistic_contributions(
    model: Pipeline,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate per-customer feature contributions for Logistic Regression.

    Contribution:
        standardized_feature_value * model_coefficient

    Positive contribution -> increases churn risk.
    Negative contribution -> decreases churn risk.
    """

    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["model"]

    X_scaled = scaler.transform(X)

    coefficients = classifier.coef_[0]

    contributions = X_scaled * coefficients

    return pd.DataFrame(
        contributions,
        columns=X.columns,
        index=X.index,
    )


def get_catboost_contributions(
    model: CatBoostClassifier,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate SHAP contributions for CatBoost.

    Positive SHAP value -> increases churn risk.
    Negative SHAP value -> decreases churn risk.
    """

    pool = Pool(X)

    shap_values = model.get_feature_importance(
        pool,
        type="ShapValues",
    )

    # Last SHAP column = expected/base value.
    contributions = shap_values[:, :-1]

    return pd.DataFrame(
        contributions,
        columns=X.columns,
        index=X.index,
    )


def build_customer_explanations(
    model,
    X: pd.DataFrame,
    customer_ids: pd.Series,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Build local model explanations for every customer.

    Returns:
    - CustomerID
    - top_risk_drivers
    - top_protective_drivers
    - model_explanation
    """

    if hasattr(model, "named_steps"):
        contributions = get_logistic_contributions(
            model,
            X,
        )

    elif isinstance(model, CatBoostClassifier):
        contributions = get_catboost_contributions(
            model,
            X,
        )

    else:
        return pd.DataFrame(
            {
                "CustomerID": customer_ids.values,
                "top_risk_drivers": "",
                "top_protective_drivers": "",
                "model_explanation": (
                    "Model explanation is not available."
                ),
            }
        )

    explanations = []

    for position, (_, row) in enumerate(
        contributions.iterrows()
    ):
        positive = (
            row[row > 0]
            .sort_values(ascending=False)
            .head(top_n)
        )

        negative = (
            row[row < 0]
            .sort_values(ascending=True)
            .head(top_n)
        )

        risk_drivers = positive.index.tolist()
        protective_drivers = negative.index.tolist()

        risk_text = (
            ", ".join(risk_drivers)
            if risk_drivers
            else "no strong positive risk drivers"
        )

        protective_text = (
            ", ".join(protective_drivers)
            if protective_drivers
            else "no strong protective drivers"
        )

        customer_id = customer_ids.iloc[position]

        explanation = (
            f"Main churn-risk drivers: {risk_text}. "
            f"Main factors reducing churn risk: "
            f"{protective_text}."
        )

        explanations.append(
            {
                "CustomerID": customer_id,
                "top_risk_drivers": risk_text,
                "top_protective_drivers": protective_text,
                "model_explanation": explanation,
            }
        )

    return pd.DataFrame(explanations)


def train_and_score(
    df: pd.DataFrame,
    target_col: str,
    id_col: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:

    X = df.drop(
        columns=[
            target_col,
            id_col,
        ]
    )

    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # ---------------------------------------------------------------
    # Logistic Regression
    # ---------------------------------------------------------------

    lr = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=5000
                ),
            ),
        ]
    )

    lr.fit(
        X_train,
        y_train,
    )

    lr_prob = lr.predict_proba(
        X_test
    )[:, 1]

    lr_pred = (
        lr_prob > 0.5
    ).astype(int)

    lr_metrics = evaluate(
        y_test,
        lr_pred,
        lr_prob,
    )

    # ---------------------------------------------------------------
    # CatBoost
    # ---------------------------------------------------------------

    cb = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        verbose=0,
        random_state=random_state,
    )

    cb.fit(
        X_train,
        y_train,
    )

    cb_prob = cb.predict_proba(
        X_test
    )[:, 1]

    cb_pred = (
        cb_prob > 0.5
    ).astype(int)

    cb_metrics = evaluate(
        y_test,
        cb_pred,
        cb_prob,
    )

    # ---------------------------------------------------------------
    # Model comparison
    # ---------------------------------------------------------------

    print("\n=== MODEL COMPARISON ===")

    print(
        "Logistic Regression:",
        lr_metrics,
    )

    print(
        "CatBoost:",
        cb_metrics,
    )

    # ---------------------------------------------------------------
    # Select best model
    # ---------------------------------------------------------------

    if cb_metrics["roc_auc"] > lr_metrics["roc_auc"]:

        print("👉 Using CatBoost")

        final_model = cb

        final_prob = cb.predict_proba(
            X
        )[:, 1]

        final_metrics = cb_metrics.copy()

        final_metrics[
            "selected_model"
        ] = "CatBoost"

    else:

        print("👉 Using Logistic Regression")

        final_model = lr

        final_prob = lr.predict_proba(
            X
        )[:, 1]

        final_metrics = lr_metrics.copy()

        final_metrics[
            "selected_model"
        ] = "Logistic Regression"

    # ---------------------------------------------------------------
    # Global feature importance
    # ---------------------------------------------------------------

    feature_importance = get_feature_importance(
        final_model,
        X,
    )

    if not feature_importance.empty:

        feature_importance.to_csv(
            "outputs/feature_importance.csv",
            index=False,
        )

    # ---------------------------------------------------------------
    # Churn scores
    # ---------------------------------------------------------------

    scores = pd.DataFrame(
        {
            id_col: df[id_col].values,
            "churn_score": final_prob,
        }
    )

    # ---------------------------------------------------------------
    # Local customer explanations
    # ---------------------------------------------------------------

    explanations = build_customer_explanations(
        model=final_model,
        X=X,
        customer_ids=df[id_col].reset_index(
            drop=True
        ),
        top_n=3,
    )

    explanations.to_csv(
        "outputs/model_explanations.csv",
        index=False,
    )

    return (
        scores,
        final_metrics,
        explanations,
    )


def save_metrics(
    metrics: dict,
    path,
) -> None:

    with open(
        path,
        "w",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )