import numpy as np
import pandas as pd

from src.analysis.regressions import (
    fit_psychological_anxiety_model,
    fit_demographic_psych_anxiety_model,
    fit_full_anxiety_model_with_personality,
)


def _dummy_regression_df(n: int = 120) -> pd.DataFrame:
    """
    Synthetic dataset for testing regression models.

    It contains:
    - DASS scales: dassanxiety, dassstress, dassdepression
    - Demographic factors: gender_clean, education_clean, urban_clean,
      married_clean, religion_grouped, race_clean
    - TIPI traits: Extraversion, Agreeableness, Conscientiousness,
      Emotional Stability, Openness
    """
    rng = np.random.default_rng(2025)

    # DASS variables
    dassstress = rng.normal(0, 1, size=n)
    dassdepression = rng.normal(0, 1, size=n)

    # True underlying relationship for anxiety
    base = 0.3 * dassstress + 0.5 * dassdepression
    noise = rng.normal(0, 0.8, size=n)
    dassanxiety = base + noise

    # Demographic factors (categorical)
    gender = rng.choice(["Female", "Male"], size=n, p=[0.6, 0.4])
    education = rng.choice(
        ["High school", "University degree", "Graduate degree"],
        size=n,
        p=[0.3, 0.5, 0.2],
    )
    urban = rng.choice(["Urban", "Rural"], size=n, p=[0.7, 0.3])
    married = rng.choice(["Never married", "Currently married"], size=n, p=[0.7, 0.3])
    religion_grouped = rng.choice(
        ["Christian", "Agnostic/Atheist", "Other"],
        size=n,
        p=[0.4, 0.4, 0.2],
    )
    race = rng.choice(["White", "Asian", "Black", "Other"], size=n)

    # TIPI traits (continuous)
    extraversion = rng.normal(0, 1, size=n)
    agreeableness = rng.normal(0, 1, size=n)
    conscientiousness = rng.normal(0, 1, size=n)
    emotional_stability = rng.normal(0, 1, size=n)
    openness = rng.normal(0, 1, size=n)

    return pd.DataFrame(
        {
            "dassanxiety": dassanxiety,
            "dassstress": dassstress,
            "dassdepression": dassdepression,
            "gender_clean": gender,
            "education_clean": education,
            "urban_clean": urban,
            "married_clean": married,
            "religion_grouped": religion_grouped,
            "race_clean": race,
            "Extraversion": extraversion,
            "Agreeableness": agreeableness,
            "Conscientiousness": conscientiousness,
            "Emotional Stability": emotional_stability,
            "Openness": openness,
        }
    )


def test_fit_psychological_anxiety_model_basic():
    """
    fit_psychological_anxiety_model should fit an OLS model of
    dassanxiety ~ dassstress + dassdepression and return a statsmodels
    results object with the expected parameters.
    """
    df = _dummy_regression_df()

    model = fit_psychological_anxiety_model(df)

    # The model should have parameter estimates for stress and depression
    param_names = list(model.params.index)
    assert "dassstress" in param_names
    assert "dassdepression" in param_names

    # Number of observations should match the input data (no unexpected row drops)
    assert model.nobs == len(df)


def test_fit_demographic_psych_anxiety_model_runs_and_has_dass_terms():
    """
    fit_demographic_psych_anxiety_model should run without error on
    a realistic dataset and keep the psychological predictors in the model.
    """
    df = _dummy_regression_df()

    model = fit_demographic_psych_anxiety_model(df)

    param_names = list(model.params.index)
    # We still expect the psychological predictors to be present
    assert "dassstress" in param_names
    assert "dassdepression" in param_names

    # There should be more parameters than the simple psychological model
    assert len(model.params) > 3  # at least const + 2 DASS + some categorical dummies
    assert model.nobs == len(df)


def test_fit_full_anxiety_model_with_personality_includes_tipi_traits():
    """
    fit_full_anxiety_model_with_personality should include the five TIPI traits
    in addition to psychological and demographic predictors.
    """
    df = _dummy_regression_df()

    model = fit_full_anxiety_model_with_personality(df)

    param_names = list(model.params.index)

    # Check that the TIPI traits appear as predictors (exact naming may depend on the formula)
    expected_tipi_terms = [
        "Extraversion",
        "Agreeableness",
        "Conscientiousness",
        "Emotional Stability",
        "Openness",
    ]

    # We allow partial matches in case the formula expands them (e.g. standardization)
    for trait in expected_tipi_terms:
        assert any(trait in name for name in param_names), f"TIPI trait {trait} not found in model params."

    # Sanity check: model has a reasonable number of parameters
    assert len(model.params) > 5
    assert model.nobs == len(df)
