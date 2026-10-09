"""Isolation Forest anomaly detection for tabular datasets.

Anomaly scores use scikit-learn's ``decision_function`` convention: lower
scores are more unusual, and negative scores are typically model anomalies.
An anomaly is a model signal, not confirmation that a record is incorrect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

DEFAULT_CONTAMINATION = 0.05
MINIMUM_SAMPLES = 5
RANDOM_STATE = 42


def _empty_result(message: str, features: list[str] | None = None) -> dict:
    return {
        "status": "unavailable",
        "message": message,
        "row_results": pd.DataFrame(columns=["row_index", "prediction", "anomaly_score"]),
        "total_anomalies": 0,
        "anomaly_percentage": 0.0,
        "numerical_features_used": features or [],
    }


def detect_anomalies(
    dataframe: pd.DataFrame,
    contamination: float = DEFAULT_CONTAMINATION,
) -> dict:
    """Detect unusual rows using numerical features without modifying ``dataframe``.

    Boolean and all-missing columns are excluded. Infinite values are treated as
    missing and median-imputed alongside ordinary missing values. Constant
    features are excluded because they do not help isolate observations.
    """
    if not 0.01 <= contamination <= 0.20:
        raise ValueError("contamination must be between 0.01 and 0.20.")
    if dataframe.empty or len(dataframe) < MINIMUM_SAMPLES:
        return _empty_result(
            f"At least {MINIMUM_SAMPLES} rows are needed for anomaly detection."
        )

    feature_names = [
        column for column in dataframe.columns
        if is_numeric_dtype(dataframe[column]) and not is_bool_dtype(dataframe[column])
    ]
    if not feature_names:
        return _empty_result("No suitable numerical features are available.")

    # Work on a copy so neither replacement nor imputation changes uploaded data.
    features = dataframe[feature_names].replace([np.inf, -np.inf], np.nan).copy()
    feature_names = [column for column in feature_names if not features[column].isna().all()]
    if not feature_names:
        return _empty_result("All numerical features are missing or infinite.")
    features = features[feature_names]

    imputed = SimpleImputer(strategy="median").fit_transform(features)
    non_constant = np.nanstd(imputed, axis=0) > 0
    feature_names = [name for name, usable in zip(feature_names, non_constant) if usable]
    if not feature_names:
        return _empty_result("Numerical features are constant after imputation.")
    model_input = imputed[:, non_constant]

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=RANDOM_STATE,
    )
    raw_predictions = model.fit_predict(model_input)
    scores = model.decision_function(model_input)
    predictions = np.where(raw_predictions == -1, "anomaly", "normal")
    row_results = pd.DataFrame(
        {
            "row_index": dataframe.index.to_list(),
            "prediction": predictions,
            "anomaly_score": scores,
        }
    )
    total_anomalies = int((raw_predictions == -1).sum())
    return {
        "status": "success",
        "message": "Lower anomaly scores indicate more unusual records; anomalies are model signals, not confirmed errors.",
        "row_results": row_results,
        "total_anomalies": total_anomalies,
        "anomaly_percentage": round(total_anomalies / len(dataframe) * 100, 2),
        "numerical_features_used": feature_names,
    }
