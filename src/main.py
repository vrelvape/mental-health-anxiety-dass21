import logging
from typing import Any

import pandas as pd

from src.config import ensure_directories_exist
from src.data_loading import load_data
from src.cleaning import full_cleaning_pipeline

logger = logging.getLogger(__name__)


def run_export_core_tables() -> None:
    """
    Export all core tables (CSV/LaTeX/PDF) using the dedicated module.
    """
    from src.export_core_tables import export_all_core_tables

    logger.info("Exporting core tables (CSV/LaTeX/PDF)...")
    export_all_core_tables()
    logger.info("Core tables exported.")


def run_anxiety_distribution(analysis_std_df: pd.DataFrame) -> None:
    """
    Generate anxiety distribution figures and severity tables.

    This includes:
    - histogram and boxplot of DASS anxiety scores,
    - severity summary table and barplot,
    - severity-by-gender and severity-by-age crosstabs (figures only).
    """
    from src.viz.anxiety_distribution import (
        plot_anxiety_histogram,
        plot_anxiety_boxplot,
        create_severity_table,
        plot_severity_barplot,
        create_severity_crosstabs,
    )

    logger.info("Generating anxiety distribution figures and tables...")

    scores = analysis_std_df["dassanxiety"]

    # Figures
    plot_anxiety_histogram(scores)
    plot_anxiety_boxplot(scores)

    # Severity tables and barplot (CSV already handled in export_core_tables)
    severity_summary = create_severity_table(analysis_std_df, save_csv=False)
    plot_severity_barplot(severity_summary)
    create_severity_crosstabs(analysis_std_df, save_csv=False)

    logger.info("Anxiety distribution outputs generated.")


def run_demographic_boxplots(analysis_std_df: pd.DataFrame) -> None:
    """
    Generate boxplots of DASS anxiety by demographic factors.
    """
    from src.viz.demographic_boxplots import generate_demographic_boxplots

    logger.info("Generating demographic boxplots...")
    generate_demographic_boxplots(analysis_std_df)
    logger.info("Demographic boxplots generated.")


def run_tipi_boxplots(analysis_std_df: pd.DataFrame) -> None:
    """
    Generate boxplots of DASS anxiety by TIPI personality traits.
    """
    from src.viz.tipi_boxplots import generate_tipi_boxplots

    logger.info("Generating TIPI boxplots...")
    generate_tipi_boxplots(analysis_std_df)
    logger.info("TIPI boxplots generated.")

def run_tipi_dass_heatmap(analysis_std_df) -> None:
    """Generate the TIPI–DASS correlation heatmap."""
    from src.viz.tipi_dass_heatmap import plot_tipi_dass_heatmap

    logging.info("Generating TIPI–DASS correlation heatmap...")
    plot_tipi_dass_heatmap(analysis_std_df)
    logging.info("TIPI–DASS heatmap generated.")

def run_geo_anxiety_outputs(analysis_std_df: pd.DataFrame) -> None:
    """
    Generate geographic figures (world map and continent barplot).

    CSV tables for these summaries are already produced in export_core_tables;
    here we only create the corresponding figures.
    """
    from src.viz.geo_anxiety_plots import generate_geo_anxiety_outputs

    logger.info("Generating geographic anxiety outputs...")
    generate_geo_anxiety_outputs(analysis_std_df, save_csv=False)
    logger.info("Geographic anxiety outputs generated.")


def run_nonresponse_plot(analysis_df: pd.DataFrame) -> None:
    """
    Generate a barplot of non-response rates for selected variables.
    """
    from src.analysis.nonresponse import compute_nonresponse_table, plot_nonresponse_bar

    logger.info("Generating non-response plot...")

    variables_to_check = [
        "gender_clean",
        "education_clean",
        "urban_clean",
        "religion_clean",
        "orientation_clean",
        "race_clean",
        "hand_clean",
        "voted_clean",
        "married_clean",
        "engnat_clean",
        "screensize_clean",
        "uniquenetworklocation_clean",
        "source_clean",
    ]

    nonresponse_table = compute_nonresponse_table(
        analysis_df,
        variables=variables_to_check,
        no_response_label="No response",
    )

    plot_nonresponse_bar(
        nonresponse_table,
        top_n=10,
        filename="nonresponse_rates_top10.png",
    )

    logger.info("Non-response plot generated.")


def run_regression_forest_plots(analysis_std_df: pd.DataFrame) -> None:
    """
    Fit the full anxiety regression model and generate two forest plots:

    - psychological and personality predictors,
    - demographic predictors.
    """
    from src.viz.coef_forest_plots import generate_forest_plots

    logger.info("Fitting full regression model and generating forest plots...")
    generate_forest_plots(analysis_std_df)
    logger.info("Regression forest plots generated.")


def run_random_forest_model(analysis_std_df: pd.DataFrame) -> None:
    """
    Fit the Random Forest anxiety model and generate a feature importance plot.

    The model uses:
    - DASS stress and depression,
    - TIPI traits,
    - demographic predictors (one-hot encoded),
    with inverse-frequency weights for gender imbalance.
    """
    from src.analysis.ml_models import fit_random_forest_anxiety
    from src.viz.ml_feature_importance import plot_rf_feature_importances

    logger.info("Fitting Random Forest model for anxiety...")

    rf_result = fit_random_forest_anxiety(analysis_std_df)

    logger.info(
        "Random Forest performance: R² train = %.3f, R² test = %.3f, "
        "RMSE train = %.3f, RMSE test = %.3f",
        rf_result.r2_train,
        rf_result.r2_test,
        rf_result.rmse_train,
        rf_result.rmse_test,
    )

    plot_rf_feature_importances(
        rf_result.feature_importances,
        top_n=15,
        filename="rf_feature_importances_top15.png",
        title="Random Forest feature importances for DASS-Anxiety",
    )

    logger.info("Random Forest feature importance plot generated.")


def main() -> None:
    """
    Main pipeline entry point.

    This function:
    - prepares directories,
    - loads and cleans the data,
    - exports all core tables,
    - generates all figures and model outputs used in the report.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )

    logger.info("Initialising results directories...")
    ensure_directories_exist()

    logger.info("Loading raw data...")
    raw_df = load_data()

    logger.info("Running full cleaning pipeline...")
    cleaned = full_cleaning_pipeline(raw_df)
    analysis_df = cleaned.analysis_df
    analysis_std_df = cleaned.analysis_standardized_df

    # 1) Tables (CSV/LaTeX/PDF)
    run_export_core_tables()

    # 2) Descriptive visualisations and summaries
    run_anxiety_distribution(analysis_std_df)
    run_demographic_boxplots(analysis_std_df)
    run_tipi_boxplots(analysis_std_df)
    run_tipi_dass_heatmap(analysis_std_df)
    run_geo_anxiety_outputs(analysis_std_df)
    run_nonresponse_plot(analysis_df)

    # 3) Models and forest plots
    run_regression_forest_plots(analysis_std_df)
    run_random_forest_model(analysis_std_df)

    logger.info("All core analyses completed.")


if __name__ == "__main__":
    main()