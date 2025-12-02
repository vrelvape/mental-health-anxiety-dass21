"""
Forest plots for regression coefficients from the anxiety models.

This module builds a tidy coefficient summary from a statsmodels
regression and generates simple forest plots with 95% confidence intervals.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.regressions import fit_full_anxiety_model_with_personality
from src.config import FIGURES_DIR, ensure_directories_exist


def build_coef_summary(model) -> pd.DataFrame:
    """
    Build a coefficient summary table from a fitted regression model.

    Parameters
    ----------
    model :
        Fitted statsmodels regression result with .params and .conf_int().

    Returns
    -------
    pd.DataFrame
        DataFrame with index = predictor names and columns
        ['coef', 'ci_lower', 'ci_upper'], excluding intercept terms.
    """
    params = model.params
    conf_int = model.conf_int()

    summary_df = pd.concat(
        [
            params.rename("coef"),
            conf_int.rename(columns={0: "ci_lower", 1: "ci_upper"}),
        ],
        axis=1,
    )

    # Drop intercept terms if present
    summary_df = summary_df.drop(index=["const", "Intercept"], errors="ignore")
    return summary_df


def plot_forest(
    summary_df: pd.DataFrame,
    predictors: list[str],
    title: str,
    filename: str,
) -> None:
    """
    Create a forest plot (coefficients + 95% CI) for selected predictors.

    Parameters
    ----------
    summary_df :
        Output of build_coef_summary; index = predictor names.
    predictors :
        List of predictor names (index labels) to include in the plot.
    title :
        Plot title.
    filename :
        Output filename (PNG) inside FIGURES_DIR.
    """
    subset = summary_df.loc[[p for p in predictors if p in summary_df.index]].copy()

    if subset.empty:
        print(f"No predictors found for plot: {title}")
        return

    subset = subset.sort_values("coef")

    ensure_directories_exist()

    plt.figure(figsize=(8, max(4, 0.5 * len(subset))))
    ax = plt.gca()

    x = subset["coef"]
    y = subset.index
    lower_err = x - subset["ci_lower"]
    upper_err = subset["ci_upper"] - x

    ax.errorbar(
        x=x,
        y=y,
        xerr=[lower_err, upper_err],
        fmt="o",
        capsize=3,
    )

    ax.axvline(0, color="gray", linestyle="--", linewidth=1)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Coefficient (with 95% CI)", fontsize=12)
    ax.set_ylabel("Predictor", fontsize=12)

    plt.tight_layout()
    output_path = FIGURES_DIR / filename
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved forest plot: {output_path}")


def generate_forest_plots(analysis_df: pd.DataFrame) -> None:
    """
    Fit the full anxiety model (psychological + demographics + TIPI),
    build a coefficient summary, and generate two forest plots:

    1) Psychological and personality predictors
    2) Demographic predictors

    Parameters
    ----------
    analysis_df :
        Standardized analysis DataFrame containing DASS variables,
        demographic factors, and TIPI traits.
    """
    ensure_directories_exist()

    model = fit_full_anxiety_model_with_personality(analysis_df)
    summary_df = build_coef_summary(model)

    psych_predictors = [
        "dassstress",
        "dassdepression",
        "Extraversion",
        "Agreeableness",
        "Conscientiousness",
        "Q('Emotional Stability')",
        "Openness",
    ]
    psych_set = set(psych_predictors)

    demographic_predictors = [name for name in summary_df.index if name not in psych_set]

    plot_forest(
        summary_df,
        predictors=psych_predictors,
        title="Regression coefficients: psychological and personality predictors",
        filename="forest_psych_personality_predictors.png",
    )

    plot_forest(
        summary_df,
        predictors=demographic_predictors,
        title="Regression coefficients: demographic predictors",
        filename="forest_demographic_predictors.png",
    )