"""
Visualisation of Random Forest feature importances for DASS-Anxiety.

This module provides a helper function to create a horizontal barplot of
the most important predictors according to a previously fitted
RandomForestRegressor model.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, ensure_directories_exist


def plot_rf_feature_importances(
    importances: pd.DataFrame,
    *,
    top_n: int = 15,
    filename: str = "rf_feature_importances_top15.png",
    title: str = "Random Forest feature importances for DASS anxiety",
) -> None:
    """
    Create and save a horizontal barplot of the top-N Random Forest
    feature importances.

    Parameters
    ----------
    importances :
        DataFrame with columns ['feature', 'importance'], as returned by
        fit_random_forest_anxiety(). The 'importance' values are typically
        normalised to sum to 1, but this is not strictly required here.
    top_n :
        Number of top features to display (by descending importance).
    filename :
        Output PNG filename inside FIGURES_DIR.
    title :
        Title displayed above the plot.
    """
    required_cols = {"feature", "importance"}
    if not required_cols.issubset(importances.columns):
        raise ValueError(
            "importances DataFrame must contain 'feature' and 'importance' columns."
        )

    if importances.empty:
        print("No feature importances to plot.")
        return

    if top_n <= 0:
        print("top_n must be positive; nothing to plot.")
        return

    ensure_directories_exist()

    # Keep only the top-N features by importance
    plot_df = (
        importances.copy()
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance", ascending=True)  # for neat horizontal ordering
    )

    plt.figure(figsize=(10, max(4, 0.4 * len(plot_df))))
    ax = sns.barplot(
        data=plot_df,
        x="importance",
        y="feature",
        orient="h",
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Relative importance", fontsize=12)
    ax.set_ylabel("Predictor", fontsize=12)

    # Annotate each bar with the importance value in %
    max_imp = plot_df["importance"].max()
    for patch in ax.patches:
        width = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        ax.text(
            width + 0.01 * max_imp,
            y,
            f"{width * 100:.1f}%",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved RF feature importance plot to: {out_path}")


if __name__ == "__main__":
    from src.analysis.ml_models import fit_random_forest_anxiety
    from src.cleaning import full_cleaning_pipeline
    from src.data_loading import load_data

    raw = load_data()
    cleaned = full_cleaning_pipeline(raw)
    analysis_std = cleaned.analysis_standardized_df

    rf_res = fit_random_forest_anxiety(analysis_std)

    plot_rf_feature_importances(
        rf_res.feature_importances,
        top_n=15,
        filename="rf_feature_importances_standalone.png",
        title="Standalone RF feature importances",
    )