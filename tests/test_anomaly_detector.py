import numpy as np
import pandas as pd

from src.anomaly_detector import detect_anomalies


def test_detects_normal_numerical_data():
    result = detect_anomalies(pd.DataFrame({"x": range(20), "y": range(20, 40)}))
    assert result["status"] == "success"
    assert result["numerical_features_used"] == ["x", "y"]
    assert len(result["row_results"]) == 20
    assert set(result["row_results"]["prediction"]) <= {"normal", "anomaly"}


def test_synthetic_outlier_is_flagged_with_deterministic_configuration():
    dataframe = pd.DataFrame({"x": [0.0] * 30 + [1000.0], "y": [0.0] * 30 + [1000.0]})
    result = detect_anomalies(dataframe, contamination=0.05)
    flagged_indices = result["row_results"].query("prediction == 'anomaly'")["row_index"].tolist()
    assert 30 in flagged_indices


def test_imputes_missing_numerical_values():
    dataframe = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0, 5.0], "y": [5, 4, 3, 2, 1]})
    result = detect_anomalies(dataframe)
    assert result["status"] == "success"
    assert result["row_results"]["anomaly_score"].notna().all()


def test_non_numerical_data_is_unavailable():
    result = detect_anomalies(pd.DataFrame({"city": ["Delhi"] * 5, "active": [True] * 5}))
    assert result["status"] == "unavailable"
    assert result["total_anomalies"] == 0


def test_boolean_and_all_missing_features_are_excluded():
    dataframe = pd.DataFrame({"value": [1, 2, 3, 4, 5], "flag": [True, False, True, False, True], "empty": [None] * 5})
    result = detect_anomalies(dataframe)
    assert result["status"] == "success"
    assert result["numerical_features_used"] == ["value"]


def test_empty_and_very_small_datasets_are_unavailable():
    assert detect_anomalies(pd.DataFrame({"x": []}))["status"] == "unavailable"
    assert detect_anomalies(pd.DataFrame({"x": [1, 2, 3, 4]}))["status"] == "unavailable"


def test_detection_is_reproducible_and_preserves_original_data():
    dataframe = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, 100.0], "constant": [7] * 5})
    original = dataframe.copy(deep=True)
    first = detect_anomalies(dataframe)
    second = detect_anomalies(dataframe)
    pd.testing.assert_frame_equal(dataframe, original)
    pd.testing.assert_frame_equal(first["row_results"], second["row_results"])
    assert first["numerical_features_used"] == ["x"]
