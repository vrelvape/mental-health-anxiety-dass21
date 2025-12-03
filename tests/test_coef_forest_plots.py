import numpy as np
import pandas as pd
import src.config as config

from src.viz.coef_forest_plots import (
    build_coef_summary,
    plot_forest,
    generate_forest_plots,
)


class DummyModel:
    """Minimal dummy model to test build_coef_summary."""

    def __init__(self):
        self.params = pd.Series(
            [1.0, 0.5, -0.2],
            index=["const", "x1", "x2"],
        )
        # Conf intervals: each row = [lower, upper]
        self._conf = pd.DataFrame(
            [[0.8, 1.2], [0.3, 0.7], [-0.4, 0.0]],
            index=["const", "x1", "x2"],
        )

    def conf_int(self) -> pd.DataFrame:
        return self._conf


def _dummy_regression_df(n: int = 80) -> pd.DataFrame:
    """
    Synthetic dataset for testing generate_forest_plots indirectly
    through the regression model.
    """
    rng = np.random.default_rng(42)

    # DASS
    stress = rng.normal(0, 1, size=n)
    depression = rng.normal(0, 1, size=n)
    anxiety = 0.3 * stress + 0.5 * depression + rng.normal(0, 0.8, size=n)

    # Demographics
    gender = rng.choice(["Female", "Male"], size=n)
    education = rng.choice(["High school", "University degree"], size=n)
    urban = rng.choice(["Urban", "Rural"], size=n)
    married = rng.choice(["Never married", "Currently married"], size=n)
    religion = rng.choice(["Christian", "Agnostic/Atheist"], size=n)
    race = rng.choice(["White", "Asian", "Black"], size=n)

    # TIPI
    extraversion = rng.normal(0, 1, size=n)
    agreeableness = rng.normal(0, 1, size=n)
    conscientiousness = rng.normal(0, 1, size=n)
    emotional_stability = rng.normal(0, 1, size=n)
    openness = rng.normal(0, 1, size=n)

    return pd.DataFrame(
        {
            "dassanxiety": anxiety,
            "dassstress": stress,
            "dassdepression": depression,
            "gender_clean": gender,
            "education_clean": education,
            "urban_clean": urban,
            "married_clean": married,
            "religion_grouped": religion,
            "race_clean": race,
            "Extraversion": extraversion,
            "Agreeableness": agreeableness,
            "Conscientiousness": conscientiousness,
            "Emotional Stability": emotional_stability,
            "Openness": openness,
        }
    )


def test_build_coef_summary_drops_intercept_and_has_ci_columns():
    model = DummyModel()
    summary = build_coef_summary(model)

    # Intercept should be dropped
    assert "const" not in summary.index

    # Columns must be present
    assert set(summary.columns) == {"coef", "ci_lower", "ci_upper"}

    # We expect 2 remaining predictors: x1 and x2
    assert set(summary.index) == {"x1", "x2"}


def test_plot_forest_creates_file_in_temp_directory(tmp_path, monkeypatch):
    """
    plot_forest should generate a PNG file in a temporary figures directory.
    """
    tmp_results = tmp_path / "results"
    tmp_figures = tmp_results / "figures"
    tmp_figures.mkdir(parents=True, exist_ok=True)

    # Patch config paths
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_results, raising=False)
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_figures, raising=False)

    from src.viz import coef_forest_plots as viz_mod
    monkeypatch.setattr(viz_mod, "FIGURES_DIR", tmp_figures, raising=False)

    # Simple summary_df for test
    summary_df = pd.DataFrame(
        {
            "coef": [0.5, -0.3],
            "ci_lower": [0.2, -0.6],
            "ci_upper": [0.8, 0.0],
        },
        index=["x1", "x2"],
    )

    filename = "test_forest_plot.png"
    plot_forest(
        summary_df=summary_df,
        predictors=["x1", "x2"],
        title="Test Forest Plot",
        filename=filename,
    )

    assert (tmp_figures / filename).exists()


def test_generate_forest_plots_creates_two_files(tmp_path, monkeypatch):
    """
    Integration-style test: generate_forest_plots should run on a synthetic
    dataset and produce at least the two expected forest plot PNG files
    in a temporary directory.
    """
    tmp_results = tmp_path / "results"
    tmp_figures = tmp_results / "figures"
    tmp_figures.mkdir(parents=True, exist_ok=True)

    # Patch config paths
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_results, raising=False)
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_figures, raising=False)

    from src.viz import coef_forest_plots as viz_mod
    monkeypatch.setattr(viz_mod, "FIGURES_DIR", tmp_figures, raising=False)

    df = _dummy_regression_df()

    generate_forest_plots(df)

    expected_files = [
        "forest_psych_personality_predictors.png",
        "forest_demographic_predictors.png",
    ]
    for fname in expected_files:
        assert (tmp_figures / fname).exists()
