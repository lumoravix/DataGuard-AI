import pandas as pd

from src.profiler import profile_dataframe
from src.quality_score import calculate_quality_score
from src.report_generator import generate_html_report


def test_report_escapes_unsafe_text_and_works_without_optional_results():
    dataframe = pd.DataFrame({"<script>alert(1)</script>": ["value"]})
    report = generate_html_report(profile_dataframe(dataframe), calculate_quality_score(dataframe), [])
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in report
    assert "Anomaly detection was not enabled" in report
    assert "No cleaned preview" in report


def test_report_includes_anomaly_and_cleaning_summaries():
    dataframe = pd.DataFrame({"score": [1, 2]})
    profile = profile_dataframe(dataframe)
    quality = calculate_quality_score(dataframe)
    anomaly = {"status": "success", "total_anomalies": 1, "anomaly_percentage": 50.0, "numerical_features_used": ["score"]}
    cleaning = {"before_missing": 1, "after_missing": 0, "before_duplicates": 1, "after_duplicates": 0, "rows_removed": 1, "operations": ["Filled values"], "notes": []}
    report = generate_html_report(profile, quality, [], anomaly, cleaning)
    assert "Detected 1 model anomaly" in report
    assert "Before/after missing values: 1/0" in report
