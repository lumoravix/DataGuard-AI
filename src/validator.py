"""Generic, non-mutating data-quality validation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_string_dtype,
)


def _issue(category: str, column: str | None, indices: list[Any], explanation: str) -> dict:
    """Build a consistent validation-result record."""
    return {
        "issue_category": category,
        "column": column,
        "affected_rows": len(indices),
        "affected_row_indices": indices,
        "explanation": explanation,
    }


def _missing_mask(series: pd.Series) -> pd.Series:
    """Treat nulls and whitespace-only text as missing without altering data."""
    whitespace_only = series.astype("string").str.fullmatch(r"\s*").fillna(False)
    return series.isna() | whitespace_only


def _matches_expected_type(series: pd.Series, expected_type: str) -> bool:
    normalized_type = expected_type.strip().lower()
    if normalized_type in {"number", "numeric"}:
        return is_numeric_dtype(series)
    if normalized_type in {"text", "string"}:
        # ``object`` dtype may contain arbitrary Python values, so verify the
        # actual non-null values instead of treating every object column as text.
        non_null = series.dropna()
        return is_string_dtype(series) and non_null.map(lambda value: isinstance(value, str)).all()
    if normalized_type in {"datetime", "date"}:
        return is_datetime64_any_dtype(series)
    if normalized_type in {"boolean", "bool"}:
        return is_bool_dtype(series)
    return str(series.dtype).lower() == normalized_type


def validate_dataframe(
    dataframe: pd.DataFrame,
    numeric_ranges: Mapping[str, Mapping[str, float | int | None]] | None = None,
    expected_types: Mapping[str, str] | None = None,
) -> list[dict]:
    """Validate arbitrary tabular data and return structured issue records.

    ``numeric_ranges`` maps column names to optional ``min`` and ``max`` values.
    Numeric values are only considered invalid when an explicit range is supplied.
    ``expected_types`` accepts friendly values such as ``number``, ``text``,
    ``datetime``, or ``boolean`` (or an exact pandas dtype string).
    """
    numeric_ranges = numeric_ranges or {}
    expected_types = expected_types or {}
    issues: list[dict] = []

    for column in dataframe.columns:
        missing_indices = dataframe.index[_missing_mask(dataframe[column])].tolist()
        if missing_indices:
            issues.append(_issue(
                "Missing values", column, missing_indices,
                "Contains null or whitespace-only values.",
            ))

    duplicate_indices = dataframe.index[dataframe.duplicated(keep=False)].tolist()
    if duplicate_indices:
        issues.append(_issue(
            "Duplicate rows", None, duplicate_indices,
            "Rows are exact duplicates of at least one other row.",
        ))

    for column, bounds in numeric_ranges.items():
        if column not in dataframe.columns or not is_numeric_dtype(dataframe[column]):
            continue
        minimum, maximum = bounds.get("min"), bounds.get("max")
        invalid_mask = pd.Series(False, index=dataframe.index)
        if minimum is not None:
            invalid_mask |= dataframe[column] < minimum
        if maximum is not None:
            invalid_mask |= dataframe[column] > maximum
        indices = dataframe.index[invalid_mask.fillna(False)].tolist()
        if indices:
            description = f"Value falls outside the configured range ({minimum} to {maximum})."
            issues.append(_issue("Invalid numerical range", column, indices, description))

    for column, expected_type in expected_types.items():
        if column in dataframe.columns and not _matches_expected_type(dataframe[column], expected_type):
            issues.append(_issue(
                "Invalid data type", column, dataframe.index.tolist(),
                f"Expected {expected_type}; found {dataframe[column].dtype}.",
            ))

    for column in dataframe.select_dtypes(include=["object", "string"]).columns:
        series = dataframe[column].dropna()
        strings = series[series.map(lambda value: isinstance(value, str))]
        space_indices = strings.index[strings.str.contains(r"^\s|\s$", regex=True)].tolist()
        if space_indices:
            issues.append(_issue(
                "Inconsistent text formatting", column, space_indices,
                "Contains leading or trailing spaces.",
            ))

        normalized = strings.str.strip().str.casefold()
        # Different casing is present when a normalized value maps to >1 originals.
        case_indices = strings.index[strings.groupby(normalized).transform("nunique") > 1].tolist()
        if case_indices:
            issues.append(_issue(
                "Inconsistent text formatting", column, case_indices,
                "Contains values that differ only by letter case.",
            ))

    return issues
