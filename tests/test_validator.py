import pandas as pd

from src.validator import validate_dataframe


def issues_by_category(issues, category):
    return [issue for issue in issues if issue["issue_category"] == category]


def test_normal_data_has_no_issues():
    dataframe = pd.DataFrame({"city": ["Delhi", "Pune"], "score": [10, 20]})
    assert validate_dataframe(dataframe) == []


def test_detects_null_and_whitespace_missing_values():
    issues = validate_dataframe(pd.DataFrame({"name": [None, "   ", "Ada"]}))
    missing = issues_by_category(issues, "Missing values")[0]
    assert missing["column"] == "name"
    assert missing["affected_rows"] == 2
    assert missing["affected_row_indices"] == [0, 1]


def test_detects_all_members_of_duplicate_rows():
    issues = validate_dataframe(pd.DataFrame({"city": ["Pune", "Pune", "Delhi"]}))
    duplicate = issues_by_category(issues, "Duplicate rows")[0]
    assert duplicate["affected_row_indices"] == [0, 1]


def test_detects_values_outside_configured_range_only():
    issues = validate_dataframe(
        pd.DataFrame({"amount": [-1, 5, 11]}),
        numeric_ranges={"amount": {"min": 0, "max": 10}},
    )
    assert issues_by_category(issues, "Invalid numerical range")[0]["affected_row_indices"] == [0, 2]
    assert validate_dataframe(pd.DataFrame({"amount": [-1]})) == []


def test_detects_expected_type_mismatch():
    issues = validate_dataframe(pd.DataFrame({"age": ["young"]}), expected_types={"age": "number"})
    assert issues_by_category(issues, "Invalid data type")[0]["column"] == "age"


def test_mixed_object_values_are_not_accepted_as_text():
    issues = validate_dataframe(pd.DataFrame({"mixed": ["text", 3]}), expected_types={"mixed": "text"})
    assert issues_by_category(issues, "Invalid data type")[0]["column"] == "mixed"


def test_detects_spaces_and_case_variations():
    issues = validate_dataframe(pd.DataFrame({"department": [" Sales", "sales", "SALES"]}))
    formatting = issues_by_category(issues, "Inconsistent text formatting")
    assert len(formatting) == 2
    assert formatting[0]["affected_row_indices"] == [0]
    assert formatting[1]["affected_row_indices"] == [0, 1, 2]


def test_empty_dataset_returns_no_issues():
    assert validate_dataframe(pd.DataFrame(columns=["name", "score"])) == []
