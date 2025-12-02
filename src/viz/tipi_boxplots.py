"""
TIPI (Ten Item Personality Inventory) boxplots for DASS-Anxiety.

This module:
- groups each TIPI trait into Low / Medium / High categories using tertiles,
- produces annotated boxplots of DASS-Anxiety scores for each trait.
"""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.api.types import CategoricalDtype

from src.config import FIGURES_DIR, ensure_directories_exist


def make_tipi_group_column(
    df: pd.DataFrame,
    trait: str,
    new_col: str,
    labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Create an ordered categorical column that groups a TIPI trait
    into Low / Medium / High using tertiles.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the TIPI trait.
    trait : str
        Name of the TIPI trait column (e.g. "Extraversion").
    new_col : str
        Name of the new categorical grouping column.
    labels : list of str or None
        Category labels in increasing order. Defaults to ["Low", "Medium", "High"].

    Returns
    -------
    pd.DataFrame
        Copy of df with the new ordered categorical column.
    """
    if labels is None:
        labels = ["Low", "Medium", "High"]

    out = df.copy()

    if trait not in out.columns:
        raise KeyError(f"{trait} not found in DataFrame columns.")

    valid = out[trait].dropna()
    if valid.empty:
        raise ValueError(f"No valid values for trait {trait}.")

    q1 = valid.quantile(1 / 3)
    q2 = valid.quantile(2 / 3)

    def group_score(x: float) -> str:
        if pd.isna(x):
            return pd.NA
        if x <= q1:
            return labels[0]
        if x <= q2:
            return labels[1]
        return labels[2]

    out[new_col] = out[trait].apply(group_score)

    cat_type = CategoricalDtype(categories=labels, ordered=True)
    out[new_col] = out[new_col].astype(cat_type)

    return out


def plot_annotated_boxplot(
    df: pd.DataFrame,
    factor_name: str,
    output_name: str,
    y_col: str = "dassanxiety",
) -> None:
    """
    Create an annotated boxplot of y_col by factor_name with summary stats
    and save it into FIGURES_DIR.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing factor_name and y_col.
    factor_name : str
        Name of the grouping column (e.g. "Extraversion_group").
    output_name : str
        Output filename for the figure (PNG).
    y_col : str
        Name of the numeric dependent variable.
    """
    plot_df = df.copy()

    plot_df = plot_df.dropna(subset=[factor_name, y_col])
    if plot_df.empty:
        print(f"No data to plot for {factor_name}.")
        return

    desc = plot_df.groupby(factor_name, observed=False)[y_col].describe()
    order = list(desc.index)

    plt.figure(figsize=(9, 6))
    # Simple three-color palette for Low / Medium / High
    palette = {"Low": "#9ecae1", "Medium": "#fdd0a2", "High": "#fb6a4a"}

    ax = sns.boxplot(
        data=plot_df,
        x=factor_name,
        y=y_col,
        order=order,
        hue=factor_name,
        palette=palette,
        legend=False,
    )

    ax.set_ylim(10, 60)
    ax.tick_params(axis="x", labelrotation=0, labelsize=9)

    # Map factor levels to x-positions on the axis
    positions = {label.get_text(): i for i, label in enumerate(ax.get_xticklabels())}

    group_counts = (
        plot_df.groupby(factor_name, observed=False)[y_col].count().reindex(order)
    )

    # Add annotations: min, Q1, median, Q3, max, and n
    for group, stats_row in desc.iterrows():
        if group not in positions:
            continue

        x = positions[group]
        min_val = stats_row["min"]
        q1 = stats_row["25%"]
        median = stats_row["50%"]
        q3 = stats_row["75%"]
        max_val = stats_row["max"]
        count = int(group_counts[group])

        ax.text(
            x,
            min_val - 1,
            f"min={min_val:.0f}",
            ha="center",
            va="top",
            fontsize=8,
        )
        ax.text(
            x,
            max_val + 0.5,
            f"max={max_val:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

        ax.text(
            x - 0.18,
            q1 - 0.5,
            f"Q1={q1:.0f}",
            ha="right",
            va="top",
            fontsize=8,
        )
        ax.text(
            x - 0.18,
            q3 + 0.5,
            f"Q3={q3:.0f}",
            ha="right",
            va="bottom",
            fontsize=8,
        )

        n_y = median + (q3 - median) * 0.4
        ax.text(
            x,
            n_y,
            f"n={count:,}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
        )

    ax.set_title(f"DASS-Anxiety by {factor_name}", fontsize=14)
    ax.set_xlabel(factor_name.replace("_", " "), fontsize=12)
    ax.set_ylabel("DASS-Anxiety score", fontsize=12)

    plt.tight_layout()
    ensure_directories_exist()
    output_path = FIGURES_DIR / output_name
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved annotated boxplot for {factor_name} to: {output_path}")


def generate_tipi_boxplots(analysis_df: pd.DataFrame) -> None:
    """
    Generate annotated boxplots of DASS-Anxiety for all TIPI traits.

    Parameters
    ----------
    analysis_df : pd.DataFrame
        Standardized analysis DataFrame containing TIPI trait columns
        and 'dassanxiety'.
    """
    ensure_directories_exist()

    trait_to_group_col = {
        "Extraversion": "Extraversion_group",
        "Agreeableness": "Agreeableness_group",
        "Conscientiousness": "Conscientiousness_group",
        "Emotional Stability": "Emotional_Stability_group",
        "Openness": "Openness_group",
    }

    df = analysis_df.copy()

    # Create grouped versions of each TIPI trait
    for trait, group_col in trait_to_group_col.items():
        df = make_tipi_group_column(df, trait=trait, new_col=group_col)

    # Plot boxplots for each grouped trait
    for trait, group_col in trait_to_group_col.items():
        filename = f"boxplot_dassanxiety_by_{group_col.lower()}_annotated.png"
        plot_annotated_boxplot(
            df=df,
            factor_name=group_col,
            output_name=filename,
            y_col="dassanxiety",
        )


if __name__ == "__main__":
    from src.data_loading import load_data
    from src.cleaning import full_cleaning_pipeline

    raw = load_data()
    cleaned = full_cleaning_pipeline(raw)
    analysis_std = cleaned.analysis_standardized_df

    generate_tipi_boxplots(analysis_std)