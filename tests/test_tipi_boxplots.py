import pandas as pd
import src.config as config

from src.viz.tipi_boxplots import (
    make_tipi_group_column,
    generate_tipi_boxplots,
)


def _dummy_tipi_df() -> pd.DataFrame:
    """Return a small dummy dataset for testing TIPI boxplots."""
    n = 12
    data = {
        "dassanxiety": [12, 18, 25, 30, 35, 40, 22, 28, 32, 38, 27, 33],
        "Extraversion": [3.0, 4.5, 2.0, 5.0, 4.0, 3.5, 2.5, 4.2, 3.8, 4.9, 2.7, 3.3],
        "Agreeableness": [4.0, 3.8, 4.5, 3.5, 4.2, 3.9, 4.1, 4.3, 3.7, 3.6, 4.4, 3.8],
        "Conscientiousness": [3.5, 4.0, 3.8, 4.5, 3.2, 4.1, 3.9, 4.3, 3.6, 4.4, 3.7, 4.2],
        "Emotional Stability": [2.5, 3.0, 2.8, 3.2, 2.9, 3.1, 2.7, 3.3, 2.6, 3.4, 2.4, 3.0],
        "Openness": [4.5, 4.0, 3.8, 4.2, 3.9, 4.1, 3.7, 4.3, 3.6, 4.4, 3.5, 4.2],
    }
    return pd.DataFrame(data)


def test_make_tipi_group_column_creates_ordered_categories():
    df = _dummy_tipi_df()
    out = make_tipi_group_column(df, trait="Extraversion", new_col="Extraversion_group")

    assert "Extraversion_group" in out.columns
    assert out["Extraversion_group"].dtype.ordered
    assert set(out["Extraversion_group"].dropna().unique()) <= {"Low", "Medium", "High"}


def test_generate_tipi_boxplots_writes_to_temp_directory(tmp_path, monkeypatch):
    """
    Ensure generate_tipi_boxplots runs without error and writes
    figures into a temporary directory rather than the real results folder.
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

    from src.viz import tipi_boxplots as viz_mod
    monkeypatch.setattr(viz_mod, "FIGURES_DIR", tmp_figures, raising=False)

    df = _dummy_tipi_df()

    generate_tipi_boxplots(df)

    # At least one TIPI boxplot should have been created
    generated = list(tmp_figures.glob("boxplot_dassanxiety_by_*_annotated.png"))
    assert generated, "No TIPI boxplot figures were created in the temp directory."