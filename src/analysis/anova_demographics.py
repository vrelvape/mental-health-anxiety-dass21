"""
One-way ANOVA helpers for demographic predictors.

This module contains small utilities to:
- create grouped versions of key demographic variables (religion, age),
- filter out explicit "No response" categories,
- run one-way ANOVAs for DASS-Anxiety across demographic factors.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from scipy import stats


def filter_no_response(df: pd.DataFrame, factor: str) -> pd.DataFrame:
    """
    Return a copy of df filtered to exclude 'No response' and missing values
    in the given factor column.
    """
    if factor not in df.columns:
        raise ValueError(f"Column '{factor}' not found in DataFrame.")

    filtered = df.copy()

    # Exclude explicit "No response"
    filtered = filtered[~filtered[factor].isin(["No response"])]

    # Exclude missing values in the factor
    filtered = filtered[filtered[factor].notna()].copy()

    if pd.api.types.is_categorical_dtype(filtered[factor]):
        filtered[factor] = filtered[factor].cat.remove_unused_categories()

    return filtered


def add_religion_grouped_column(
    df: pd.DataFrame,
    source_col: str = "religion_clean",
    new_col: str = "religion_grouped",
) -> pd.DataFrame:
    """
    Add a grouped religion column to the DataFrame.

    Groups:
    - Religious: Muslim, Christian (all), Hindu, Buddhist, Jewish, Sikh
    - Non-religious: Agnostic, Atheist
    - Unspecified: Other
    - 'No response' and unknown values are mapped to NaN and removed later.
    """
    df = df.copy()

    religious = [
        "Muslim",
        "Christian (Catholic)",
        "Christian (Protestant)",
        "Christian (Other)",
        "Christian (Mormon)",
        "Hindu",
        "Buddhist",
        "Jewish",
        "Sikh",
    ]
    non_religious = ["Agnostic", "Atheist"]
    other_unspecified = ["Other"]

    def regroup_religion(x: Optional[str]) -> Optional[str]:
        if x in religious:
            return "Religious"
        elif x in non_religious:
            return "Non-religious"
        elif x in other_unspecified:
            return "Unspecified"
        elif x == "No response":
            return None
        else:
            return None

    if source_col not in df.columns:
        raise ValueError(f"Column '{source_col}' not found in DataFrame.")

    df[new_col] = df[source_col].map(regroup_religion)
    df[new_col] = df[new_col].astype("category")

    return df


def add_age_group_column(
    df: pd.DataFrame,
    source_col: str = "age_clean",
    new_col: str = "age_group",
) -> pd.DataFrame:
    """
    Add an age_group categorical column using the bins and labels
    from the original notebook.
    """
    df = df.copy()

    if source_col not in df.columns:
        raise ValueError(f"Column '{source_col}' not found in DataFrame.")

    bins = [12, 17, 24, 34, 49, 100]
    labels = [
        "Adolescents(13–17)",
        "Young Adults(18–24)",
        "Emerging Adults(25–34)",
        "Middle-aged Adults(35–49)",
        "Seniors(50+)",
    ]

    df[new_col] = pd.cut(
        df[source_col],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True,
        ordered=True,
    )

    df[new_col] = df[new_col].astype("category")

    return df


def run_one_way_anova(
    df: pd.DataFrame,
    dependent: str,
    factor: str,
    exclude_no_response: bool = True,
) -> Dict[str, object]:
    """
    Run a one-way ANOVA for a dependent variable across groups
    defined by a categorical factor.
    """
    if dependent not in df.columns:
        raise ValueError(f"Column '{dependent}' not found in DataFrame.")

    if exclude_no_response:
        df = filter_no_response(df, factor)
    else:
        if factor not in df.columns:
            raise ValueError(f"Column '{factor}' not found in DataFrame.")
        df = df.copy()

    grouped = df.groupby(factor, observed=True)[dependent]

    # Extract groups as numpy arrays
    groups = [group.values for _, group in grouped]

    # Need at least two groups with data to run ANOVA
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        raise ValueError(
            f"Not enough non-empty groups to run ANOVA for factor '{factor}'."
        )

    f_stat, p_value = stats.f_oneway(*groups)
    group_stats = grouped.agg(["mean", "std", "count"]).round(2)

    return {
        "factor": factor,
        "dependent": dependent,
        "f_stat": float(f_stat),
        "p_value": float(p_value),
        "group_stats": group_stats,
    }


def run_demographic_anova_for_dassanxiety(
    analysis_df: pd.DataFrame,
    factors: List[str],
    exclude_no_response: bool = True,
) -> Dict[str, Dict[str, object]]:
    """
    Run one-way ANOVAs for 'dassanxiety' across multiple demographic factors.

    If 'religion_grouped' or 'age_group' are requested and missing,
    the corresponding columns are created.
    """
    df = analysis_df.copy()

    if "religion_grouped" in factors and "religion_grouped" not in df.columns:
        df = add_religion_grouped_column(df)

    if "age_group" in factors and "age_group" not in df.columns:
        df = add_age_group_column(df)

    results: Dict[str, Dict[str, object]] = {}

    for factor in factors:
        results[factor] = run_one_way_anova(
            df=df,
            dependent="dassanxiety",
            factor=factor,
            exclude_no_response=exclude_no_response,
        )

    return results