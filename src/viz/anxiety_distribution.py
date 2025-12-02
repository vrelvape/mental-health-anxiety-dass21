"""
Visualisations and summary tables for the distribution of DASS-Anxiety scores.
"""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns

from src.data_loading import load_data
from src.cleaning import full_cleaning_pipeline
from src.config import FIGURES_DIR, TABLES_DIR, ensure_directories_exist


# Severity categories and colour bands used in the plots
SEVERITY_RANGES = [
    (0, 14, "forestgreen", "Normal"),
    (14, 18, "lightgreen", "Mild"),
    (18, 26, "lemonchiffon", "Moderate"),
    (26, 34, "lightpink", "Severe"),
    (34, 56, "lightcoral", "Extremely Severe"),
]

SEVERITY_BINS = [0, 14, 18, 26, 34, 56]
SEVERITY_LABELS = ["Normal", "Mild", "Moderate", "Severe", "Extremely Severe"]


def load_anxiety_scores() -> Tuple[pd.Series, pd.DataFrame]:
    """
    Load the raw data, run the cleaning pipeline and return:

    - the DASS-Anxiety scores as a Series,
    - the full standardized analysis DataFrame.
    """
    raw_df = load_data()
    cleaned = full_cleaning_pipeline(raw_df)
    analysis_std = cleaned.analysis_standardized_df

    return analysis_std["dassanxiety"].copy(), analysis_std


#  Core plotting functions

def plot_anxiety_histogram(scores: pd.Series) -> None:
    """
    Create and save a histogram of anxiety scores with severity bands
    and mean/median lines.
    """
    ensure_directories_exist()
    sns.set(style="whitegrid")

    mean_val = scores.mean()
    median_val = scores.median()

    fig, ax = plt.subplots(figsize=(14, 6))

    # Coloured severity bands in the background
    for start, end, color, label in SEVERITY_RANGES:
        ax.axvspan(start, end, color=color, alpha=0.3, label=label, zorder=0)

    sns.histplot(
        scores,
        bins=43,
        kde=True,
        color="cornflowerblue",
        edgecolor="black",
        ax=ax,
        zorder=2,
    )

    ax.axvline(
        mean_val,
        color="yellow",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_val:.2f}",
        zorder=3,
    )
    ax.axvline(
        median_val,
        color="orangered",
        linestyle="--",
        linewidth=2,
        label=f"Median = {median_val:.2f}",
        zorder=3,
    )

    ax.set_title("Distribution of Anxiety Scores (DASS-Anxiety)", fontsize=14)
    ax.set_xlabel("Anxiety score", fontsize=12)
    ax.set_ylabel("Number of participants", fontsize=12)

    legend = ax.legend(
        loc="upper right",
        title="Legend",
        facecolor="white",
        framealpha=1,
    )
    legend.get_frame().set_edgecolor("gray")

    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_path = FIGURES_DIR / "hist_dassanxiety_with_severity_bands.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved anxiety histogram to: {out_path}")


def plot_anxiety_boxplot(scores: pd.Series) -> None:
    """
    Create and save a boxplot of anxiety scores with severity bands
    and annotated quantiles.
    """
    ensure_directories_exist()
    sns.set(style="whitegrid")

    mean_val = scores.mean()
    median_val = scores.median()
    q1 = scores.quantile(0.25)
    q3 = scores.quantile(0.75)
    min_val = scores.min()
    max_val = scores.max()

    fig, ax = plt.subplots(figsize=(14, 2.8))

    for start, end, color, _label in SEVERITY_RANGES:
        ax.axvspan(start, end, color=color, alpha=0.3, zorder=0)

    sns.boxplot(
        x=scores,
        ax=ax,
        color="cornflowerblue",
        width=0.4,
        showmeans=False,
        zorder=2,
    )

    ax.axvline(
        mean_val,
        color="yellow",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_val:.2f}",
        zorder=3,
    )
    ax.axvline(
        median_val,
        color="orangered",
        linestyle="--",
        linewidth=2,
        label=f"Median = {median_val:.2f}",
        zorder=3,
    )

    # Annotate min, Q1, median, Q3, max
    for val in [min_val, q1, median_val, q3, max_val]:
        ax.text(
            val,
            0.15,
            f"{val:.2f}",
            ha="right",
            va="baseline",
            fontsize=9,
            color="black",
            zorder=4,
        )

    handles = [
        mpatches.Patch(color="forestgreen", alpha=0.3, label="Normal"),
        mpatches.Patch(color="lightgreen", alpha=0.3, label="Mild"),
        mpatches.Patch(color="lemonchiffon", alpha=0.3, label="Moderate"),
        mpatches.Patch(color="lightpink", alpha=0.3, label="Severe"),
        mpatches.Patch(color="lightcoral", alpha=0.3, label="Extremely Severe"),
        plt.Line2D(
            [], [], color="yellow", linestyle="--", linewidth=2, label=f"Mean = {mean_val:.2f}"
        ),
        plt.Line2D(
            [], [], color="orangered", linestyle="--", linewidth=2, label=f"Median = {median_val:.2f}"
        ),
    ]

    legend = ax.legend(
        handles=handles,
        loc="upper left",
        title="Legend",
        facecolor="white",
        framealpha=1,
    )
    legend.get_frame().set_edgecolor("gray")

    ax.set_title("Boxplot of Anxiety Scores (DASS-Anxiety)", fontsize=14)
    ax.set_xlabel("Anxiety score", fontsize=12)
    ax.set_yticks([])

    plt.tight_layout()
    out_path = FIGURES_DIR / "boxplot_dassanxiety_with_severity_bands.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved anxiety boxplot to: {out_path}")

#  Severity tables & crosstabs

def create_severity_table(
    analysis_df: pd.DataFrame,
    save_csv: bool = True,
) -> pd.DataFrame:
    """
    Compute severity-category counts and percentages and (optionally)
    save them as a CSV file.
    """
    df = analysis_df.copy()

    df["anxiety_severity"] = pd.cut(
        df["dassanxiety"],
        bins=SEVERITY_BINS,
        labels=SEVERITY_LABELS,
        right=True,
        include_lowest=True,
    )

    counts = df["anxiety_severity"].value_counts().sort_index()
    perc = (counts / len(df) * 100).round(2)

    summary = pd.DataFrame({"Count": counts, "Percentage (%)": perc})
    summary.index.name = "anxiety_severity"

    if save_csv:
        ensure_directories_exist()
        out_path = TABLES_DIR / "anxiety_severity_summary.csv"
        summary.to_csv(out_path, index=True)
        print(f"Saved anxiety severity table to: {out_path}")

    return summary


def plot_severity_barplot(severity_summary: pd.DataFrame) -> None:
    """
    Create and save a barplot of severity categories using counts
    from the severity summary table.
    """
    ensure_directories_exist()

    plot_df = severity_summary.reset_index()
    first_col = plot_df.columns[0]
    if first_col != "severity":
        plot_df = plot_df.rename(columns={first_col: "severity"})

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        x="severity",
        y="Count",
        data=plot_df,
        hue="severity",
        palette="pastel",
        legend=False,
    )

    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    for i, row in plot_df.iterrows():
        ax.text(
            i,
            row["Count"] + 0.01 * plot_df["Count"].max(),
            f"n={int(row['Count'])}\n({row['Percentage (%)']:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
        )

    ax.set_title("Distribution of Anxiety Severity Categories", fontsize=14)
    ax.set_xlabel("Severity category", fontsize=12)
    ax.set_ylabel("Number of participants", fontsize=12)
    plt.tight_layout()

    out_path = FIGURES_DIR / "barplot_anxiety_severity_categories.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved severity barplot to: {out_path}")


def create_severity_crosstabs(
    analysis_df: pd.DataFrame,
    save_csv: bool = True,
) -> None:
    """
    Create crosstab tables of severity by gender and by age_group and
    optionally save them as CSV files.
    """
    df = analysis_df.copy()

    if "anxiety_severity" not in df.columns:
        df["anxiety_severity"] = pd.cut(
            df["dassanxiety"],
            bins=SEVERITY_BINS,
            labels=SEVERITY_LABELS,
            right=True,
            include_lowest=True,
        )

    if "age_group" not in df.columns and "age_clean" in df.columns:
        bins = [12, 17, 24, 34, 49, 100]
        labels = [
            "Adolescents(13–17)",
            "Young Adults(18–24)",
            "Emerging Adults(25–34)",
            "Middle-aged Adults(35–49)",
            "Seniors(50+)",
        ]
        df["age_group"] = pd.cut(
            df["age_clean"],
            bins=bins,
            labels=labels,
            right=True,
            include_lowest=True,
        )

    gender_counts = pd.crosstab(
        df["gender_clean"],
        df["anxiety_severity"],
        normalize=False,
    )
    gender_percent = (
        pd.crosstab(
            df["gender_clean"],
            df["anxiety_severity"],
            normalize="index",
        )
        * 100
    ).round(2)

    if save_csv:
        ensure_directories_exist()
        gender_counts_path = TABLES_DIR / "anxiety_severity_by_gender_counts.csv"
        gender_percent_path = TABLES_DIR / "anxiety_severity_by_gender_percent.csv"
        gender_counts.to_csv(gender_counts_path)
        gender_percent.to_csv(gender_percent_path)
        print(f"Saved gender x severity counts to: {gender_counts_path}")
        print(f"Saved gender x severity percentages to: {gender_percent_path}")

    if "age_group" in df.columns:
        age_counts = pd.crosstab(
            df["age_group"],
            df["anxiety_severity"],
            normalize=False,
        )
        age_percent = (
            pd.crosstab(
                df["age_group"],
                df["anxiety_severity"],
                normalize="index",
            )
            * 100
        ).round(2)

        if save_csv:
            ensure_directories_exist()
            age_counts_path = TABLES_DIR / "anxiety_severity_by_agegroup_counts.csv"
            age_percent_path = TABLES_DIR / "anxiety_severity_by_agegroup_percent.csv"
            age_counts.to_csv(age_counts_path)
            age_percent.to_csv(age_percent_path)
            print(f"Saved age group x severity counts to: {age_counts_path}")
            print(f"Saved age group x severity percentages to: {age_percent_path}")
    else:
        print("age_group column not available: skipped age x severity crosstabs.")


if __name__ == "__main__":
    # Convenience entry point to regenerate all anxiety distribution outputs
    scores, analysis_std = load_anxiety_scores()

    plot_anxiety_histogram(scores)
    plot_anxiety_boxplot(scores)

    summary = create_severity_table(analysis_std, save_csv=True)
    plot_severity_barplot(summary)
    create_severity_crosstabs(analysis_std, save_csv=True)