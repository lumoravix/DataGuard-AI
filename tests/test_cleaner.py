import pandas as pd

from src.cleaner import clean_dataframe, dataframe_to_csv_bytes


def test_mean_and_median_imputation():
    dataframe = pd.DataFrame({"value": [1.0, None, 5.0]})
    assert clean_dataframe(dataframe, numerical_missing="mean")["cleaned_dataframe"].loc[1, "value"] == 3.0
    assert clean_dataframe(dataframe, numerical_missing="median")["cleaned_dataframe"].loc[1, "value"] == 3.0


def test_mode_imputation_and_all_missing_column():
    dataframe = pd.DataFrame({"city": ["Pune", None, "Pune"], "empty": [None, None, None]})
    result = clean_dataframe(dataframe, categorical_missing="mode")
    assert result["cleaned_dataframe"].loc[1, "city"] == "Pune"
    assert result["cleaned_dataframe"]["empty"].isna().all()
    assert "all values are missing" in result["notes"][0]


def test_duplicate_removal_and_combined_operations_preserve_original():
    dataframe = pd.DataFrame({"value": [1.0, None, 1.0], "city": ["A", "B", "A"]})
    original = dataframe.copy(deep=True)
    result = clean_dataframe(dataframe, numerical_missing="mean", remove_duplicates=True)
    pd.testing.assert_frame_equal(dataframe, original)
    assert result["rows_removed"] == 1
    assert len(result["cleaned_dataframe"]) == 2


def test_empty_and_no_cleaning_selected():
    empty = pd.DataFrame(columns=["value"])
    assert clean_dataframe(empty)["cleaned_dataframe"].empty
    dataframe = pd.DataFrame({"value": [None, 2]})
    result = clean_dataframe(dataframe)
    assert result["cleaned_dataframe"].isna().sum().sum() == 1


def test_csv_output_integrity():
    dataframe = pd.DataFrame({"name": ["Ada"], "score": [10]})
    restored = pd.read_csv(__import__("io").BytesIO(dataframe_to_csv_bytes(dataframe)))
    pd.testing.assert_frame_equal(restored, dataframe)
