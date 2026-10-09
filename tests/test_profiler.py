import pandas as pd

from src.profiler import missing_values_table, profile_dataframe


def test_profiles_dataset_with_missing_values_duplicates_and_numbers():
    dataframe = pd.DataFrame({"name": ["Ada", "Ada", None], "score": [10, 10, None]})
    profile = profile_dataframe(dataframe)
    assert profile["rows"] == 3
    assert profile["columns"] == 2
    assert profile["column_names"] == ["name", "score"]
    assert profile["missing_values"] == {"name": 1, "score": 1}
    assert profile["missing_percentages"] == {"name": 33.33, "score": 33.33}
    assert profile["duplicate_rows"] == 1
    assert "score" in profile["numerical_summary"].index


def test_profiles_no_numerical_columns_or_missing_values():
    profile = profile_dataframe(pd.DataFrame({"city": ["Delhi", "Pune"]}))
    assert profile["numerical_summary"].empty
    assert profile["missing_values"] == {"city": 0}
    assert missing_values_table(profile)["Missing values"].sum() == 0


def test_profiles_empty_dataframe_without_dividing_by_zero():
    profile = profile_dataframe(pd.DataFrame(columns=["value"]))
    assert profile["rows"] == 0
    assert profile["missing_percentages"] == {"value": 0.0}
