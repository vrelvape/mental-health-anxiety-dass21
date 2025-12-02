"""
Descriptive statistics helpers for the mental health survey project.

These functions are used to:
- summarize numeric variables (mean, std, quartiles, missing),
- inspect the distribution of categorical variables.

They are generic and can be reused on any pandas DataFrame.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd


def descriptive_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for all numeric columns in the DataFrame.

    Returns a table with:
    - count, mean, std, min, quartiles, max (via DataFrame.describe),
    - an extra 'missing' column with the number of NaN values per variable.
    """
    numeric_df = df.select_dtypes(include=["number"])
    summary = numeric_df.describe().T
    summary["missing"] = numeric_df.isna().sum()
    return summary


def descriptive_categorical_counts(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Compute value counts for all categorical or object-type columns.

    Returns
    -------
    dict
        A dictionary mapping each column name to its value_counts() Series,
        including missing values (dropna=False).
    """
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    results: Dict[str, pd.Series] = {}

    for col in categorical_cols:
        results[col] = df[col].value_counts(dropna=False)

    return results


def full_descriptive_summary(
    df: pd.DataFrame,
    categorical_normalize: bool = False,
) -> Dict[str, object]:
    """
    Compute a full descriptive summary of a DataFrame.

    Includes:
    - numeric_summary: DataFrame of descriptive stats for numeric columns,
    - categorical_counts: dict of value counts for categorical columns,
    - optionally categorical_proportions: dict of normalized value counts.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    categorical_normalize : bool, default False
        If True, also return relative frequencies for categorical variables.

    Returns
    -------
    dict
        A dictionary with at least:
        - "numeric_summary": pd.DataFrame
        - "categorical_counts": Dict[str, pd.Series]
        and, if `categorical_normalize=True`:
        - "categorical_proportions": Dict[str, pd.Series]
    """
    numeric_summary = descriptive_numeric_summary(df)
    categorical_counts = descriptive_categorical_counts(df)

    if categorical_normalize:
        categorical_proportions: Dict[str, pd.Series] = {
            col: counts / counts.sum()
            for col, counts in categorical_counts.items()
        }
        return {
            "numeric_summary": numeric_summary,
            "categorical_counts": categorical_counts,
            "categorical_proportions": categorical_proportions,
        }

    return {
        "numeric_summary": numeric_summary,
        "categorical_counts": categorical_counts,
    }