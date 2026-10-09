"""Dataset profiling helpers."""

import pandas as pd


def profile_dataframe(dataframe: pd.DataFrame) -> dict:
    """Return core structural and data-quality facts for a DataFrame."""
    missing_values = dataframe.isna().sum()
    row_count = len(dataframe)
    missing_percentages = (
        (missing_values / row_count * 100).round(2)
        if row_count
        else pd.Series(0.0, index=dataframe.columns)
    )
    numeric_columns = dataframe.select_dtypes(include="number")
    numerical_summary = (
        numeric_columns.describe().transpose().round(2)
        if len(numeric_columns.columns) > 0
        else pd.DataFrame()
    )
    return {
        "rows": row_count,
        "columns": len(dataframe.columns),
        "column_names": dataframe.columns.tolist(),
        "data_types": dataframe.dtypes.astype(str).to_dict(),
        "missing_values": missing_values.to_dict(),
        "missing_percentages": missing_percentages.to_dict(),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "numerical_summary": numerical_summary,
    }


def missing_values_table(profile: dict) -> pd.DataFrame:
    """Format a profile's missing-value data for display and charting."""
    return pd.DataFrame(
        {
            "Column": list(profile["missing_values"].keys()),
            "Missing values": list(profile["missing_values"].values()),
            "Missing (%)": list(profile["missing_percentages"].values()),
        }
    )
