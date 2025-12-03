import pandas as pd
import src.config as config

from src.viz.geo_anxiety_plots import (
    compute_mean_anxiety_by_country,
    compute_continent_stats,
    save_continent_table_and_plot,
    save_top10_mean_anxiety_by_country,
    save_top10_countries_by_participants,
)


def _dummy_geo_df() -> pd.DataFrame:
    """Return a small dummy dataset for testing geographic outputs."""
    return pd.DataFrame(
        {
            "dassanxiety": [20, 25, 30, 35, 40, 28, 32, 22],
            "country_name": [
                "Switzerland",
                "Switzerland",
                "France",
                "France",
                "Germany",
                "Germany",
                "Italy",
                "Spain",
            ],
            "continent": [
                "Europe",
                "Europe",
                "Europe",
                "Europe",
                "Europe",
                "Europe",
                "Europe",
                "Europe",
            ],
        }
    )


def test_compute_mean_anxiety_by_country_has_expected_columns():
    """Mean-anxiety-by-country table should have all expected columns."""
    df = _dummy_geo_df()
    mean_df = compute_mean_anxiety_by_country(df)

    # Required columns present
    assert {"country_name", "mean_anxiety", "iso_alpha_3"}.issubset(mean_df.columns)

    # One row per unique country
    assert len(mean_df) == df["country_name"].nunique()

    # At least some ISO codes should be resolved
    assert mean_df["iso_alpha_3"].notna().any()


def test_geo_tables_are_written_to_temp_directory(tmp_path, monkeypatch):
    """
    Ensure geographic tables and continent barplot are written into
    a temporary directory instead of the real results folder.
    """
    tmp_results = tmp_path / "results"
    tmp_figures = tmp_results / "figures"
    tmp_tables = tmp_results / "tables"
    tmp_figures.mkdir(parents=True, exist_ok=True)
    tmp_tables.mkdir(parents=True, exist_ok=True)

    # Patch config paths
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_results, raising=False)
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_figures, raising=False)
    monkeypatch.setattr(config, "TABLES_DIR", tmp_tables, raising=False)

    # Also patch the module-level variables inside geo_anxiety_plots
    from src.viz import geo_anxiety_plots as viz_mod
    monkeypatch.setattr(viz_mod, "FIGURES_DIR", tmp_figures, raising=False)
    monkeypatch.setattr(viz_mod, "TABLES_DIR", tmp_tables, raising=False)

    df = _dummy_geo_df()

    mean_df = compute_mean_anxiety_by_country(df)
    continent_stats = compute_continent_stats(df)

    # These should all write into the temporary folders
    save_continent_table_and_plot(continent_stats)
    save_top10_mean_anxiety_by_country(mean_df)
    save_top10_countries_by_participants(df)

    # Check that expected files exist in the TEMP tables/figures directory
    assert (tmp_tables / "mean_anxiety_by_continent.csv").exists()
    assert (tmp_tables / "top10_mean_anxiety_by_country.csv").exists()
    assert (tmp_tables / "top10_countries_by_participants.csv").exists()
    assert (tmp_figures / "barplot_anxiety_by_continent.png").exists()