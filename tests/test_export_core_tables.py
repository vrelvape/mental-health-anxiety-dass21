import pandas as pd
import src.export_core_tables as export_mod


class FakeCleaned:
    """Minimal fake object mimicking the output of full_cleaning_pipeline()."""

    def __init__(self) -> None:
        # Large tables
        self.clean_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        self.clean_valid_df = pd.DataFrame({"x": [5, 6], "y": [7, 8]})

        # analysis_df (used for ANOVA: demographics + TIPI)
        self.analysis_df = pd.DataFrame(
            {
                "dassanxiety": [10.0, 20.0, 30.0],
                "gender_clean": ["M", "F", "F"],
                "age_group": ["young", "adult", "old"],
                "orientation_clean": ["het", "bi", "het"],
                "married_clean": ["no", "yes", "no"],
                "urban_clean": ["city", "town", "city"],
                "race_clean": ["A", "B", "A"],
                "education_clean": ["high", "uni", "uni"],
                "religion_grouped": ["A", "A", "B"],
                # TIPI traits (needed by personality ANOVA)
                "Extraversion": [3.0, 4.0, 5.0],
                "Agreeableness": [4.0, 3.5, 4.5],
                "Conscientiousness": [3.5, 4.0, 4.2],
                "Emotional Stability": [2.5, 3.0, 3.5],
                "Openness": [4.5, 4.0, 3.8],
            }
        )

        # analysis_standardized_df (used for severity + geo + ML models)
        self.analysis_standardized_df = pd.DataFrame(
            {
                "dassanxiety": [5, 15, 28, 40, 50],
                "age_clean": [20, 22, 30, 45, 60],
                "gender_clean": ["M", "F", "F", "F", "M"],
                "continent": ["Europe", "Asia", "Europe", "Africa", "Europe"],
                "country_name": ["A", "B", "A", "C", "A"],
            }
        )


def test_export_all_core_tables(tmp_path, monkeypatch):
    """
    High-level test: export_all_core_tables should run end-to-end
    and create the expected CSV/LaTeX/PDF files in temporary directories.
    """
    fake_cleaned = FakeCleaned()

    # 1) Mock data loading (we do not use the real raw dataset here)
    monkeypatch.setattr(
        export_mod,
        "load_data",
        lambda: pd.DataFrame({"dummy": [1, 2, 3]}),
    )

    # 2) Mock cleaning pipeline to return our fake cleaned object
    monkeypatch.setattr(
        export_mod,
        "full_cleaning_pipeline",
        lambda df: fake_cleaned,
    )

    # 3) Redirect output folders (CSV/LaTeX/PDF) to tmp_path
    csv_dir = tmp_path / "csv"
    latex_dir = tmp_path / "latex"
    pdf_dir = tmp_path / "pdf"

    monkeypatch.setattr(export_mod, "CSV_DIR", csv_dir, raising=False)
    monkeypatch.setattr(export_mod, "LATEX_DIR", latex_dir, raising=False)
    monkeypatch.setattr(export_mod, "PDF_DIR", pdf_dir, raising=False)

    # 4) Run the export
    export_mod.export_all_core_tables()

    # 5) Directory checks
    assert csv_dir.exists()
    assert latex_dir.exists()
    assert pdf_dir.exists()

    # Large tables → CSV only
    large_tables = [
        "clean_df",
        "clean_valid_df",
        "analysis_df",
        "analysis_standardized_df",
    ]
    for name in large_tables:
        assert (csv_dir / f"{name}.csv").exists()

    # Summary tables → 3 formats
    expected_multi = [
        "anxiety_severity_summary",
        "anxiety_severity_by_gender_counts",
        "anxiety_severity_by_gender_percent",
        "anxiety_severity_by_agegroup_counts",
        "anxiety_severity_by_agegroup_percent",
        "mean_anxiety_by_continent",
        "top10_mean_anxiety_by_country",
        "top10_countries_by_participants",
        "anova_demographics_summary",
        "anova_personality_summary",
        "rf_feature_importances",
        "model_comparison",
    ]

    for name in expected_multi:
        assert (csv_dir / f"{name}.csv").exists()
        assert (latex_dir / f"{name}.tex").exists()
        assert (pdf_dir / f"{name}.pdf").exists()