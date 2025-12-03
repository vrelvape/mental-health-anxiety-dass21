import pandas as pd
import src.config as config

from src.viz.demographic_boxplots import (
    make_age_group_column,
    generate_demographic_boxplots,
)


def _dummy_df() -> pd.DataFrame:
    """Return a small dummy dataset for testing demographic boxplots."""
    return pd.DataFrame(
        {
            "dassanxiety": [12, 18, 25, 30, 35, 40, 22, 28],
            "gender_clean": [
                "Male",
                "Female",
                "Female",
                "Male",
                "Female",
                "Other",
                "Male",
                "Female",
            ],
            "orientation_clean": ["Heterosexual"] * 4 + ["Bisexual"] * 4,
            "married_clean": ["Never married"] * 6 + ["Currently married"] * 2,
            "urban_clean": [
                "Urban",
                "Urban",
                "Suburban",
                "Rural",
                "Urban",
                "Rural",
                "Suburban",
                "Urban",
            ],
            "race_clean": ["White", "White", "Asian", "Black", "White", "Other", "Black", "Asian"],
            "education_clean": [
                "High school",
                "University degree",
                "University degree",
                "Graduate degree",
                "High school",
                "Graduate degree",
                "High school",
                "University degree",
            ],
            "religion_clean": [
                "Agnostic",
                "Christian (Catholic)",
                "Atheist",
                "Muslim",
                "Other",
                "Christian (Protestant)",
                "Agnostic",
                "Atheist",
            ],
            "age_clean": [17, 19, 23, 31, 40, 52, 27, 35],
        }
    )


def test_make_age_group_column_creates_valid_categories():
    """make_age_group_column should add a non-null age_group column."""
    df = _dummy_df()
    out = make_age_group_column(df)

    assert "age_group" in out.columns
    assert out["age_group"].notna().all()


def test_generate_demographic_boxplots_uses_temp_directory(tmp_path, monkeypatch):
    """
    Ensure demographic boxplots are written into a temporary directory
    rather than the real results folder.
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

    from src.viz import demographic_boxplots as viz_mod
    monkeypatch.setattr(viz_mod, "FIGURES_DIR", tmp_figures, raising=False)

    df = make_age_group_column(_dummy_df())

    generate_demographic_boxplots(df)

    expected = tmp_figures / "boxplot_dassanxiety_by_gender.png"
    assert expected.exists()