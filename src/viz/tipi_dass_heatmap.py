import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.data_loading import load_data
from src.cleaning import full_cleaning_pipeline
from src.analysis.descriptive import full_descriptive_summary
from src.config import FIGURES_DIR, ensure_directories_exist


CORR_COLUMNS = [
    "dassanxiety",
    "dassstress",
    "dassdepression",
    "dasssum",
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Emotional Stability",
    "Openness",
]


def describe_analysis_dataframe(analysis_df: pd.DataFrame) -> None:
    """
    Print basic descriptive statistics for the analysis DataFrame.

    Shows the numeric summary and one example of categorical counts.
    """
    summary = full_descriptive_summary(analysis_df, categorical_normalize=False)

    numeric_summary = summary["numeric_summary"]
    categorical_counts = summary["categorical_counts"]

    print("Numeric summary (first 10 rows):")
    print(numeric_summary.head(10))

    print("\nNumber of categorical variables summarized:", len(categorical_counts))

    if categorical_counts:
        first_cat = list(categorical_counts.keys())[0]
        print("\nExample categorical variable and its counts:")
        print("Variable:", first_cat)
        print(categorical_counts[first_cat])


def plot_tipi_dass_heatmap(analysis_standardized_df: pd.DataFrame) -> None:
    """
    Create and save a correlation heatmap between DASS scores and TIPI traits.
    """
    ensure_directories_exist()

    # Keep only the columns that are actually present
    available_cols = [c for c in CORR_COLUMNS if c in analysis_standardized_df.columns]
    if not available_cols:
        print("No correlation columns available for the heatmap; skipping.")
        return

    corr_df = analysis_standardized_df[available_cols].copy()
    corr_matrix = corr_df.corr()

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        annot_kws={"fontsize": 8},
    )
    plt.title("Correlation heatmap: Big Five traits vs DASS scores", fontsize=14)
    plt.tight_layout()

    output_path = FIGURES_DIR / "heatmap_tipi_dass.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Heatmap saved to: {output_path}")


def main() -> None:
    """
    Load the real dataset, run the cleaning pipeline, print basic descriptives,
    and generate the TIPI–DASS correlation heatmap.
    """
    print("Loading data and running cleaning pipeline...")
    raw_df = load_data()
    cleaned = full_cleaning_pipeline(raw_df)

    analysis_df = cleaned.analysis_df
    analysis_standardized_df = cleaned.analysis_standardized_df

    describe_analysis_dataframe(analysis_df)
    plot_tipi_dass_heatmap(analysis_standardized_df)


if __name__ == "__main__":
    main()