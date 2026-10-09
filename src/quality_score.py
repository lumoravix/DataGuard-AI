"""Transparent, non-mutating data-quality scoring."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.validator import validate_dataframe

BASE_WEIGHTS = {"completeness": 40.0, "uniqueness": 30.0, "validity": 30.0}


def _unavailable(message: str) -> dict:
    return {
        "status": "unavailable",
        "overall_score": None,
        "components": {"completeness": None, "uniqueness": None, "validity": None},
        "effective_weights": {},
        "explanations": [message],
        "validity_issue_explanations": [],
    }


def _missing_cells(dataframe: pd.DataFrame) -> int:
    """Count null and whitespace-only string cells without changing the data."""
    total = 0
    for column in dataframe.columns:
        series = dataframe[column]
        whitespace = series.astype("string").str.fullmatch(r"\s*").fillna(False)
        total += int((series.isna() | whitespace).sum())
    return total


def calculate_quality_score(
    dataframe: pd.DataFrame,
    numeric_ranges: Mapping[str, Mapping[str, float | int | None]] | None = None,
    expected_types: Mapping[str, str] | None = None,
    validation_results: list[dict] | None = None,
) -> dict:
    """Calculate quality components and a weighted score from 0 to 100.

    Completeness counts non-missing cells (including non-blank text). Uniqueness
    uses ``DataFrame.duplicated()`` so only additional duplicate rows reduce the
    score. Validity is evaluated solely from configured numerical-range and
    expected-type rules; generic missing/duplicate issues are not counted again.
    """
    numeric_ranges = numeric_ranges or {}
    expected_types = expected_types or {}
    if dataframe.empty or len(dataframe.columns) == 0:
        return _unavailable("A dataset with at least one row and column is required to calculate a quality score.")

    cell_count = len(dataframe) * len(dataframe.columns)
    missing_count = _missing_cells(dataframe)
    if missing_count == cell_count:
        return _unavailable("All dataset values are missing, so a meaningful quality score cannot be calculated.")

    completeness = round((cell_count - missing_count) / cell_count * 100, 2)
    additional_duplicates = int(dataframe.duplicated().sum())
    uniqueness = round((len(dataframe) - additional_duplicates) / len(dataframe) * 100, 2)

    applicable_rules: list[tuple[str, str]] = []
    for column in numeric_ranges:
        if column in dataframe.columns and is_numeric_dtype(dataframe[column]):
            applicable_rules.append(("Invalid numerical range", column))
    for column in expected_types:
        if column in dataframe.columns:
            applicable_rules.append(("Invalid data type", column))

    if validation_results is None:
        validation_results = validate_dataframe(dataframe, numeric_ranges, expected_types)

    issue_lookup = {
        (issue["issue_category"], issue["column"]): issue
        for issue in validation_results
    }
    validity_explanations: list[str] = []
    if applicable_rules:
        total_rule_checks = len(dataframe) * len(applicable_rules)
        failed_checks = 0
        for category, column in applicable_rules:
            issue = issue_lookup.get((category, column))
            if issue:
                # One row can fail a rule once, even if a future validator emits
                # repeated detail for that same rule.
                affected = len(set(issue["affected_row_indices"]))
                failed_checks += min(affected, len(dataframe))
                validity_explanations.append(
                    f"{column}: {affected} row(s) failed the configured {category.lower()} rule."
                )
        validity = round((total_rule_checks - failed_checks) / total_rule_checks * 100, 2)
        effective_weights = BASE_WEIGHTS.copy()
        validity_note = "Validity uses configured numerical range and expected-type rules."
    else:
        validity = None
        redistributed_total = BASE_WEIGHTS["completeness"] + BASE_WEIGHTS["uniqueness"]
        effective_weights = {
            "completeness": round(BASE_WEIGHTS["completeness"] / redistributed_total * 100, 2),
            "uniqueness": round(BASE_WEIGHTS["uniqueness"] / redistributed_total * 100, 2),
            "validity": 0.0,
        }
        validity_note = "Validity was not evaluated because no applicable range or type rules were configured; its weight was redistributed proportionally."

    overall_score = round(
        completeness * effective_weights["completeness"] / 100
        + uniqueness * effective_weights["uniqueness"] / 100
        + (validity or 0) * effective_weights["validity"] / 100,
        2,
    )
    return {
        "status": "success",
        "overall_score": max(0.0, min(100.0, overall_score)),
        "components": {"completeness": completeness, "uniqueness": uniqueness, "validity": validity},
        "effective_weights": effective_weights,
        "explanations": [
            f"Completeness: {cell_count - missing_count} of {cell_count} cells are populated.",
            f"Uniqueness: {len(dataframe) - additional_duplicates} of {len(dataframe)} rows are not additional duplicates.",
            validity_note,
        ],
        "validity_issue_explanations": validity_explanations,
    }
