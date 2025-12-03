import pandas as pd

from src.viz.anxiety_distribution import (
    SEVERITY_LABELS,
    plot_anxiety_histogram,
    plot_anxiety_boxplot,
    create_severity_table,
    plot_severity_barplot,
    create_severity_crosstabs,
)
import src.config as config


def _make_dummy_analysis_df() -> pd.DataFrame:
    """Small dummy DataFrame to exercise the plotting and table functions."""
    return pd.DataFrame(
        {
            "dassanxiety": [5, 10, 15, 20, 27, 33, 40, 45],
            "gender_clean": [
                "Male", "Female", "Female", "Male",
                "Female", "Other", "Male", "Female",
            ],
            "age_clean": [16, 22, 28, 35, 42, 55, 19, 31],
        }
    )


def test_create_severity_table_basic(tmp_path, monkeypatch):
    """
    create_severity_table returns a DataFrame with the expected structure.

    The test uses a temporary tables directory so that no real files
    in results/tables are overwritten.
    """
    # Redirect TABLES_DIR to a temporary directory
    tmp_results = tmp_path / "results"
    tmp_tables = tmp_results / "tables"
    tmp_tables.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "RESULTS_DIR", tmp_results, raising=False)
    monkeypatch.setattr(config, "TABLES_DIR", tmp_tables, raising=False)

    from src.viz import anxiety_distribution as viz_mod
    monkeypatch.setattr(viz_mod, "TABLES_DIR", tmp_tables, raising=False)

    df = _make_dummy_analysis_df()
    summary = create_severity_table(df)

    assert isinstance(summary, pd.DataFrame)
    assert set(summary.columns) == {"Count", "Percentage (%)"}
    assert summary["Count"].sum() == len(df)
    assert set(summary.index).issubset(set(SEVERITY_LABELS))

    # The CSV file must be created in the temporary tables directory
    assert (tmp_tables / "anxiety_severity_summary.csv").exists()


def test_plot_functions_do_not_crash(tmp_path, monkeypatch):
    """
    Integration-style test: plotting functions should run without raising
    exceptions, and should write into a temporary figures/tables folder
    instead of the real results directory.
    """
    # Temporary results directories
    tmp_results = tmp_path / "results"
    tmp_figures = tmp_results / "figures"
    tmp_tables = tmp_results / "tables"
    tmp_figures.mkdir(parents=True, exist_ok=True)
    tmp_tables.mkdir(parents=True, exist_ok=True)

    # Patch paths in config and in the viz module
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_results, raising=False)
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_figures, raising=False)
    monkeypatch.setattr(config, "TABLES_DIR", tmp_tables, raising=False)

    from src.viz import anxiety_distribution as viz_mod
    monkeypatch.setattr(viz_mod, "FIGURES_DIR", tmp_figures, raising=False)
    monkeypatch.setattr(viz_mod, "TABLES_DIR", tmp_tables, raising=False)

    df = _make_dummy_analysis_df()
    scores = df["dassanxiety"]

    # If any of these calls raises, pytest will fail
    plot_anxiety_histogram(scores)
    plot_anxiety_boxplot(scores)
    summary = create_severity_table(df)
    plot_severity_barplot(summary)
    create_severity_crosstabs(df)

    # Check that the expected figures exist in the temporary directory
    assert (tmp_figures / "hist_dassanxiety_with_severity_bands.png").exists()
    assert (tmp_figures / "boxplot_dassanxiety_with_severity_bands.png").exists()
    assert (tmp_figures / "barplot_anxiety_severity_categories.png").exists()