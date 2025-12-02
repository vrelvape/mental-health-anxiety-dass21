"""
ANOVA helpers for personality (TIPI) predictors.

This module:
- builds quantile-based groups for each TIPI trait (e.g. Low / Medium / High),
- runs one-way ANOVAs on DASS-Anxiety across those groups.
"""

from typing import Dict, List

import pandas as pd

from .anova_demographics import run_one_way_anova


def add_tipi_group_columns(
    df: pd.DataFrame,
    traits: List[str],
    n_groups: int = 3,
) -> pd.DataFrame:
    """
    Create grouped versions of TIPI trait scores using quantile-based bins.

    For each trait, a new categorical column is created with suffix '_group'.
    By default, traits are split into three groups:
    - Low
    - Medium
    - High
    """
    df = df.copy()

    if n_groups == 3:
        labels = ["Low", "Medium", "High"]
    elif n_groups == 4:
        labels = ["Low", "Medium-low", "Medium-high", "High"]
    else:
        labels = [f"G{i+1}" for i in range(n_groups)]

    for trait in traits:
        if trait not in df.columns:
            # Trait not available in this dataset
            continue

        valid = df[trait].dropna()

        # If there are too few distinct values, quantile cuts are not meaningful
        if valid.nunique() < n_groups:
            continue

        try:
            df[f"{trait}_group"] = pd.qcut(
                df[trait],
                q=n_groups,
                labels=labels,
                duplicates="drop",
            ).astype("category")
        except ValueError:
            # If qcut still fails (e.g. highly skewed data), skip this trait
            continue

    return df


def run_personality_anova_for_dassanxiety(
    analysis_df: pd.DataFrame,
    traits: List[str],
    n_groups: int = 3,
) -> Dict[str, Dict[str, object]]:
    """
    Run one-way ANOVAs for 'dassanxiety' across grouped TIPI traits.

    For each TIPI trait, a grouped version (e.g. 'Extraversion_group')
    is created using quantile-based bins, then a one-way ANOVA is run
    on 'dassanxiety' across these groups.
    """
    df = add_tipi_group_columns(analysis_df, traits=traits, n_groups=n_groups)

    results: Dict[str, Dict[str, object]] = {}

    for trait in traits:
        factor_name = f"{trait}_group"
        if factor_name not in df.columns:
            # The grouping could not be created (too few distinct values, etc.)
            continue

        res = run_one_way_anova(
            df=df,
            dependent="dassanxiety",
            factor=factor_name,
            exclude_no_response=True,
        )
        results[trait] = res

    return results