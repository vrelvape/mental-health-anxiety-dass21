import numpy as np
import pandas as pd

from src.analysis.anova_demographics import run_demographic_anova_for_dassanxiety


def _dummy_demographic_df() -> pd.DataFrame:
    """Create a small synthetic dataset to test demographic ANOVA on anxiety."""
    rng = np.random.default_rng(42)

    n = 60
    gender = np.where(rng.random(n) < 0.6, "Female", "Male")
    orientation = np.where(rng.random(n) < 0.8, "Heterosexual", "Bisexual")
    married = np.where(rng.random(n) < 0.7, "Never married", "Currently married")
    urban = np.where(rng.random(n) < 0.5, "Urban", "Rural")
    race = rng.choice(["White", "Asian", "Black"], size=n)
    education = rng.choice(["High school", "University", "Graduate"], size=n)
    religion_grouped = rng.choice(
        ["Christian", "Agnostic/Atheist", "Other"],
        size=n,
    )
    age_group = rng.choice(
        [
            "Adolescents(13–17)",
            "Young Adults(18–24)",
            "Emerging Adults(25–34)",
            "Middle-aged Adults(35–49)",
            "Seniors(50+)",
        ],
        size=n,
    )

    # Make dassanxiety depend a bit on gender and urban vs rural,
    # so it is not just pure noise.
    base = 20 + (gender == "Female") * 3 + (urban == "Urban") * 2
    noise = rng.normal(0, 5, size=n)
    dassanxiety = base + noise

    return pd.DataFrame(
        {
            "dassanxiety": dassanxiety,
            "gender_clean": gender,
            "orientation_clean": orientation,
            "married_clean": married,
            "urban_clean": urban,
            "race_clean": race,
            "education_clean": education,
            "religion_grouped": religion_grouped,
            "age_group": age_group,
        }
    )


def test_run_demographic_anova_for_dassanxiety_structure() -> None:
    """
    Check that run_demographic_anova_for_dassanxiety returns:
    - one entry per factor, and
    - for each factor: f_stat, p_value, and a group_stats DataFrame.
    """
    df = _dummy_demographic_df()

    factors = [
        "gender_clean",
        "orientation_clean",
        "married_clean",
        "urban_clean",
        "race_clean",
        "education_clean",
        "religion_grouped",
        "age_group",
    ]

    results = run_demographic_anova_for_dassanxiety(df, factors)

    # One result per factor
    assert isinstance(results, dict)
    assert set(results.keys()) == set(factors)

    for factor, res in results.items():
        # Required keys
        assert "f_stat" in res
        assert "p_value" in res
        assert "group_stats" in res

        # Types
        assert isinstance(res["f_stat"], (float, np.floating))
        assert isinstance(res["p_value"], (float, np.floating))
        assert isinstance(res["group_stats"], pd.DataFrame)

        group_stats = res["group_stats"]

        # Index should match the groups of the factor in the data
        expected_groups = df[factor].dropna().unique()
        assert set(group_stats.index) == set(expected_groups)

        # We expect the usual describe() columns
        for col in ["mean", "std", "count"]:
            assert col in group_stats.columns

        # For ANOVA there must be at least two groups
        assert len(group_stats) >= 2