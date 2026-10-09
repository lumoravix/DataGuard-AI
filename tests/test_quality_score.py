import pandas as pd

from src.quality_score import calculate_quality_score


def test_clean_dataset_scores_100_with_redistributed_weights():
    result = calculate_quality_score(pd.DataFrame({"name": ["Ada", "Lin"], "score": [1, 2]}))
    assert result["overall_score"] == 100
    assert result["components"]["validity"] is None
    assert result["effective_weights"] == {"completeness": 57.14, "uniqueness": 42.86, "validity": 0.0}


def test_missing_values_reduce_completeness():
    result = calculate_quality_score(pd.DataFrame({"name": ["Ada", None], "score": [1, 2]}))
    assert result["components"]["completeness"] == 75.0
    assert result["overall_score"] == 85.72


def test_duplicate_rows_reduce_uniqueness():
    result = calculate_quality_score(pd.DataFrame({"name": ["Ada", "Ada"], "score": [1, 1]}))
    assert result["components"]["uniqueness"] == 50.0


def test_configured_invalid_values_reduce_validity():
    result = calculate_quality_score(
        pd.DataFrame({"score": [5, 20]}), numeric_ranges={"score": {"min": 0, "max": 10}}
    )
    assert result["components"]["validity"] == 50.0
    assert result["overall_score"] == 85.0
    assert result["effective_weights"] == {"completeness": 40.0, "uniqueness": 30.0, "validity": 30.0}


def test_missing_and_duplicate_issues_do_not_reduce_validity_again():
    dataframe = pd.DataFrame({"score": [5, 5, None, None]})
    result = calculate_quality_score(dataframe, numeric_ranges={"score": {"min": 0, "max": 10}})
    assert result["components"]["validity"] == 100.0


def test_no_configured_rules_does_not_evaluate_validity():
    result = calculate_quality_score(pd.DataFrame({"value": [1, 2]}))
    assert result["components"]["validity"] is None
    assert result["effective_weights"]["validity"] == 0.0


def test_empty_and_all_missing_datasets_are_unavailable():
    assert calculate_quality_score(pd.DataFrame(columns=["value"]))["status"] == "unavailable"
    assert calculate_quality_score(pd.DataFrame({"value": [None, None]}))["status"] == "unavailable"


def test_scores_are_bounded():
    result = calculate_quality_score(
        pd.DataFrame({"value": [None, 100, 100]}), numeric_ranges={"value": {"max": 0}}
    )
    assert 0 <= result["overall_score"] <= 100
