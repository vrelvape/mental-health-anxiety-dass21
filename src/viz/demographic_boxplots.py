"""
Boxplots of DASS-Anxiety scores across demographic groups.

This module generates annotated boxplots for several cleaned demographic
variables (gender, education, urbanicity, race, religion, age groups, etc.).
"""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.api.types import CategoricalDtype

from src.analysis.anova_demographics import add_religion_grouped_column
from src.config import FIGURES_DIR, ensure_directories_exist


def make_age_group_column(
    df: pd.DataFrame,
    source_col: str = "age_clean",
) -> pd.DataFrame:
    """
    Create an age_group categorical column based on numeric ages.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a numeric age column.
    source_col : str
        Name of the numeric age column.

    Returns
    -------
    pd.DataFrame
        Copy of df with a new 'age_group' categorical column.
    """
    bins = [12, 17, 24, 34, 49, 100]
    labels = [
        "Adolescents(13–17)",
        "Young Adults(18–24)",
        "Emerging Adults(25–34)",
        "Middle-aged Adults(35–49)",
        "Seniors(50+)",
    ]

    out = df.copy()

    if source_col in out.columns:
        out["age_group"] = pd.cut(
            out[source_col],
            bins=bins,
            labels=labels,
            right=True,
            include_lowest=True,
        )

    return out


def boxplot_anxiety_by_factor(
    df: pd.DataFrame,
    factor: str,
    output_name: str,
    y_col: str = "dassanxiety",
    drop_values: Optional[List[str]] = None,
) -> None:
    """
    Create an annotated boxplot of anxiety scores by a demographic factor
    and save it in FIGURES_DIR.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing the demographic factor and anxiety scores.
    factor : str
        Name of the demographic variable (e.g. "gender_clean").
    output_name : str
        Output filename for the figure (PNG).
    y_col : str
        Name of the numeric dependent variable.
    drop_values : list of str or None
        Values that should be excluded before plotting.
    """
    plot_df = df.copy()

    if drop_values is not None:
        plot_df = plot_df[~plot_df[factor].isin(drop_values)].copy()

    if isinstance(plot_df[factor].dtype, CategoricalDtype):
        plot_df[factor] = plot_df[factor].cat.remove_unused_categories()

    plot_df = plot_df.dropna(subset=[factor, y_col])
    if plot_df.empty:
        print(f"Skipping {factor}: no data available.")
        return

    # Descriptive stats per group (min, Q1, median, Q3, max, count)
    desc = plot_df.groupby(factor, observed=False)[y_col].describe()
    order = list(desc.index)

    palette = sns.color_palette("Set2", n_colors=len(order))

    plt.figure(figsize=(9, 6))
    ax = sns.boxplot(
        data=plot_df,
        x=factor,
        y=y_col,
        order=order,
        palette=palette,
        hue=factor,
        legend=False,
    )

    ax.set_ylim(10, 60)
    ax.tick_params(axis="x", labelsize=8)

    # Map group labels to x positions on the axis
    positions = {label.get_text(): i for i, label in enumerate(ax.get_xticklabels())}

    # Add annotations (min, Q1, median, Q3, max, n)
    for group, stats_row in desc.iterrows():
        if group not in positions:
            continue

        x = positions[group]
        min_val, q1, median, q3, max_val = (
            stats_row["min"],
            stats_row["25%"],
            stats_row["50%"],
            stats_row["75%"],
            stats_row["max"],
        )
        count = int(stats_row["count"])

        ax.text(x, min_val - 1, f"min={min_val:.0f}", ha="center", va="top", fontsize=8)
        ax.text(x, max_val + 0.5, f"max={max_val:.0f}", ha="center", va="bottom", fontsize=8)

        ax.text(x - 0.22, q1 - 0.5, f"Q1={q1:.0f}", ha="right", va="top", fontsize=8)
        ax.text(x - 0.22, q3 + 0.5, f"Q3={q3:.0f}", ha="right", va="bottom", fontsize=8)

        label_y = median + (q3 - median) * 0.4
        ax.text(
            x,
            label_y,
            f"n={count:,}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
        )

    ax.set_title(f"DASS-Anxiety by {factor}", fontsize=14)
    ax.set_xlabel(factor, fontsize=12)
    ax.set_ylabel("DASS-Anxiety score", fontsize=12)

    plt.tight_layout()
    ensure_directories_exist()
    output_path = FIGURES_DIR / output_name
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved boxplot for {factor} to: {output_path}")


def generate_demographic_boxplots(analysis_df: pd.DataFrame) -> None:
    """
    Generate all demographic boxplots for DASS-Anxiety.

    The function assumes that analysis_df contains:
    - 'dassanxiety'
    - cleaned demographic variables (gender_clean, education_clean, etc.)
    """
    ensure_directories_exist()

    df = analysis_df.copy()

    # Add grouped religion column and age groups if needed
    df = add_religion_grouped_column(
        df,
        source_col="religion_clean",
        new_col="religion_grouped",
    )
    df = make_age_group_column(df, source_col="age_clean")

    factors = [
        ("gender_clean", "boxplot_dassanxiety_by_gender.png"),
        ("orientation_clean", "boxplot_dassanxiety_by_orientation.png"),
        ("married_clean", "boxplot_dassanxiety_by_married.png"),
        ("urban_clean", "boxplot_dassanxiety_by_urban.png"),
        ("race_clean", "boxplot_dassanxiety_by_race.png"),
        ("education_clean", "boxplot_dassanxiety_by_education.png"),
        ("religion_grouped", "boxplot_dassanxiety_by_religion_grouped.png"),
        ("age_group", "boxplot_dassanxiety_by_age_group.png"),
    ]

    for factor, filename in factors:
        if factor not in df.columns:
            print(f"Skipping {factor}: column missing.")
            continue

        drop_vals = ["No response"]
        if factor == "religion_grouped":
            drop_vals = ["No response", "Exclude"]

        boxplot_anxiety_by_factor(
            df,
            factor=factor,
            output_name=filename,
            y_col="dassanxiety",
            drop_values=drop_vals,
        )


if __name__ == "__main__":
    from src.data_loading import load_data
    from src.cleaning import full_cleaning_pipeline

    raw = load_data()
    cleaned = full_cleaning_pipeline(raw)
    analysis_std = cleaned.analysis_standardized_df

    generate_demographic_boxplots(analysis_std)