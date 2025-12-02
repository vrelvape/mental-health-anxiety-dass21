"""
Non-response analysis utilities.

This module provides:
- a helper to compute non-response rates for a set of variables;
- a function to plot the top non-response rates as a bar chart.

It is used to document potential missing-data patterns and survey biases.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, ensure_directories_exist


def compute_nonresponse_table(
    df: pd.DataFrame,
    variables: Sequence[str],
    no_response_label: str = "No response",
) -> pd.DataFrame:
    """
    Compute non-response counts and percentages for a list of variables.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the cleaned analysis data.
    variables : sequence of str
        Names of the columns to inspect for the `no_response_label`.
    no_response_label : str, default "No response"
        Label used in the dataset to indicate a non-response.

    Returns
    -------
    pd.DataFrame
        Table with one row per variable and columns:
        - variable
        - n_total
        - n_no_response
        - pct_no_response  (in percent, float)
    """
    rows: list[dict] = []

    for col in variables:
        if col not in df.columns:
            # Silently skip unknown columns (robust to minor schema changes)
            continue

        col_series = df[col]
        n_total = col_series.notna().sum()
        n_no = (col_series == no_response_label).sum()
        pct_no = (n_no / n_total * 100) if n_total > 0 else 0.0

        rows.append(
            {
                "variable": col,
                "n_total": int(n_total),
                "n_no_response": int(n_no),
                "pct_no_response": float(pct_no),
            }
        )

    table = pd.DataFrame(rows)

    if table.empty:
        return table

    table = table.sort_values("pct_no_response", ascending=False).reset_index(drop=True)
    return table


def plot_nonresponse_bar(
    nonresponse_table: pd.DataFrame,
    top_n: int = 10,
    filename: str = "nonresponse_rates_top10.png",
) -> None:
    """
    Create and save a horizontal bar plot of non-response rates.

    Parameters
    ----------
    nonresponse_table : pd.DataFrame
        Output from `compute_nonresponse_table`.
    top_n : int, default 10
        Number of variables to show (highest non-response rates).
    filename : str, default "nonresponse_rates_top10.png"
        File name for the saved figure in FIGURES_DIR.
    """
    if nonresponse_table.empty:
        # Nothing to plot: avoid creating an empty figure.
        return

    ensure_directories_exist()

    plot_df = nonresponse_table.head(top_n).copy()

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=plot_df,
        x="pct_no_response",
        y="variable",
        orient="h",
    )
    plt.xlabel("Non-response rate (%)", fontsize=12)
    plt.ylabel("Variable", fontsize=12)
    plt.title("Top non-response rates by variable", fontsize=14)
    plt.tight_layout()

    output_path = FIGURES_DIR / filename
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Non-response bar plot saved to: {output_path}")