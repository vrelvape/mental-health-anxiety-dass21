"""
OLS regression models for DASS-Anxiety.

This module defines three related linear models:

1. A psychological baseline:
   dassanxiety ~ dassstress + dassdepression

2. A model adding core demographics:
   dassanxiety ~ psychological predictors + gender, education, urbanicity,
                 marital status, religion, race

3. A full model adding TIPI personality traits on top of the previous one.
"""

from __future__ import annotations

from typing import List

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pandas.api.types import CategoricalDtype

from .anova_demographics import add_religion_grouped_column


# Categorical predictors we want to clean before fitting regression models
CATEGORICAL_FOR_REGRESSION: List[str] = [
    "gender_clean",
    "education_clean",
    "urban_clean",
    "married_clean",
    "religion_grouped",
    "race_clean",
]


def ensure_religion_grouped(
    df: pd.DataFrame,
    source_col: str = "religion_clean",
    new_col: str = "religion_grouped",
) -> pd.DataFrame:
    """
    Make sure a grouped religion column is present in the DataFrame.

    If ``new_col`` already exists, the DataFrame is returned unchanged.
    Otherwise, it is created from ``source_col`` using the helper defined in
    ``anova_demographics``.
    """
    if new_col in df.columns:
        return df
    return add_religion_grouped_column(df, source_col=source_col, new_col=new_col)


def filter_no_response_for_regression(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with explicit 'No response' values in key categorical columns.

    This keeps the design matrix for regression clean and avoids dummy
    variables for missing or undefined categories.
    """
    df = df.copy()

    for col in CATEGORICAL_FOR_REGRESSION:
        if col not in df.columns:
            continue

        df = df[df[col] != "No response"]

        if isinstance(df[col].dtype, CategoricalDtype):
            df[col] = df[col].cat.remove_unused_categories()

    return df


def fit_psychological_anxiety_model(
    analysis_standardized_df: pd.DataFrame,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit a baseline model with only psychological predictors:

        dassanxiety ~ dassstress + dassdepression

    This uses the standardized analysis DataFrame and returns a fitted
    statsmodels OLS result object.
    """
    df = analysis_standardized_df.copy()

    # Keep only the variables we need and drop rows with missing values
    model_df = df[["dassanxiety", "dassstress", "dassdepression"]].dropna().copy()

    y = model_df["dassanxiety"]
    X = sm.add_constant(model_df[["dassstress", "dassdepression"]])

    model = sm.OLS(y, X).fit()
    return model


def fit_demographic_psych_anxiety_model(
    analysis_standardized_df: pd.DataFrame,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit a model combining psychological and demographic predictors:

        dassanxiety ~ dassstress + dassdepression
                      + gender_clean + education_clean + urban_clean
                      + married_clean + religion_grouped + race_clean

    This function:
    - ensures the ``religion_grouped`` column exists,
    - removes 'No response' categories in the key demographics,
    - fits an OLS model using a formula interface.
    """
    df = ensure_religion_grouped(analysis_standardized_df.copy())
    df = filter_no_response_for_regression(df)

    formula = (
        "dassanxiety ~ dassstress + dassdepression + "
        "gender_clean + education_clean + urban_clean + "
        "married_clean + religion_grouped + race_clean"
    )

    model = smf.ols(formula=formula, data=df, missing="drop").fit()
    return model


def fit_full_anxiety_model_with_personality(
    analysis_standardized_df: pd.DataFrame,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit the full regression model including personality traits:

        dassanxiety ~ dassstress + dassdepression
                      + gender_clean + education_clean + urban_clean
                      + married_clean + religion_grouped + race_clean
                      + Extraversion + Agreeableness + Conscientiousness
                      + Emotional Stability + Openness

    The ``Emotional Stability`` column has a space in its name, so it is
    referenced inside the formula using statsmodels' Q() helper.
    """
    df = ensure_religion_grouped(analysis_standardized_df.copy())
    df = filter_no_response_for_regression(df)

    formula = (
        "dassanxiety ~ dassstress + dassdepression + "
        "gender_clean + education_clean + urban_clean + "
        "married_clean + religion_grouped + race_clean + "
        "Extraversion + Agreeableness + Conscientiousness + "
        "Q('Emotional Stability') + Openness"
    )

    model = smf.ols(formula=formula, data=df, missing="drop").fit()
    return model