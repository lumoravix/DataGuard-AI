"""Opt-in, non-mutating tabular data cleaning helpers."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype


def _missing_mask(series: pd.Series) -> pd.Series:
    """Recognize nulls and whitespace-only text as missing values."""
    return series.isna() | series.astype("string").str.fullmatch(r"\s*").fillna(False)


def clean_dataframe(
    dataframe: pd.DataFrame,
    numerical_missing: str = "unchanged",
    categorical_missing: str = "unchanged",
    remove_duplicates: bool = False,
) -> dict:
    """Return a cleaned copy and an auditable summary of selected operations.

    No type inference or coercion occurs: numeric-looking text and identifier
    columns remain untouched. All filling is explicitly selected by the caller.
    """
    if numerical_missing not in {"unchanged", "mean", "median"}:
        raise ValueError("numerical_missing must be unchanged, mean, or median.")
    if categorical_missing not in {"unchanged", "mode"}:
        raise ValueError("categorical_missing must be unchanged or mode.")

    cleaned = dataframe.copy(deep=True)
    before_missing = int(sum(_missing_mask(cleaned[column]).sum() for column in cleaned.columns))
    before_duplicates = int(cleaned.duplicated().sum())
    operations: list[str] = []
    notes: list[str] = []

    for column in cleaned.columns:
        series = cleaned[column]
        missing = _missing_mask(series)
        if not missing.any():
            continue
        if is_numeric_dtype(series):
            if numerical_missing == "unchanged":
                continue
            non_missing = series[~missing]
            if non_missing.empty:
                notes.append(f"{column}: left unchanged because all values are missing.")
                continue
            fill_value = non_missing.mean() if numerical_missing == "mean" else non_missing.median()
            try:
                cleaned.loc[missing, column] = fill_value
            except (TypeError, ValueError):
                # Nullable integer columns cannot represent a fractional mean/median.
                cleaned[column] = series.astype("Float64").mask(missing, fill_value)
                notes.append(f"{column}: converted to a nullable float type to store the selected fill value.")
            operations.append(f"{column}: filled missing numerical values with the {numerical_missing}.")
        elif categorical_missing == "mode":
            non_missing = series[~missing]
            if non_missing.empty:
                notes.append(f"{column}: left unchanged because all values are missing.")
                continue
            fill_value = non_missing.mode(dropna=True).iloc[0]
            cleaned.loc[missing, column] = fill_value
            operations.append(f"{column}: filled missing categorical values with the mode.")

    if remove_duplicates:
        cleaned = cleaned.drop_duplicates(keep="first").copy()
        removed = len(dataframe) - len(cleaned)
        operations.append(f"Removed {removed} additional duplicate row(s), keeping the first occurrence.")
    else:
        removed = 0

    after_missing = int(sum(_missing_mask(cleaned[column]).sum() for column in cleaned.columns))
    after_duplicates = int(cleaned.duplicated().sum())
    data_type_changes = [
        f"{column}: {dataframe[column].dtype} → {cleaned[column].dtype}"
        for column in dataframe.columns
        if dataframe[column].dtype != cleaned[column].dtype
    ]
    if not operations:
        operations.append("No cleaning operations were selected or applicable.")
    return {
        "cleaned_dataframe": cleaned,
        "before_missing": before_missing,
        "after_missing": after_missing,
        "before_duplicates": before_duplicates,
        "after_duplicates": after_duplicates,
        "rows_removed": removed,
        "operations": operations,
        "notes": notes,
        "data_type_changes": data_type_changes,
    }


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Serialize a cleaned dataset without its pandas index."""
    return dataframe.to_csv(index=False).encode("utf-8")
