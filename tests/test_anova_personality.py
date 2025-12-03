import numpy as np
import pandas as pd

from src.analysis.anova_personality import run_personality_anova_for_dassanxiety


def _dummy_personality_df() -> pd.DataFrame:
    """
    Small synthetic dataset for testing personality ANOVA on DASS anxiety.

    It contains:
    - a continuous outcome: 'dassanxiety'
    - the five TIPI traits as continuous predictors
    """
    rng = np.random.default_rng(123)

    n = 80

    extraversion = rng.normal(3.5, 0.8, size=n)
    agreeableness = rng.normal(4.0, 0.6, size=n)
    conscientiousness = rng.normal(3.8, 0.7, size=n)
    emotional_stability = rng.normal(3.0, 0.7, size=n)
    openness = rng.normal(4.0, 0.5, size=n)

    # Simple pattern: higher emotional stability → lower anxiety, plus noise
    base = 30 - 2.0 * emotional_stability
    noise = rng.normal(0, 4, size=n)
    dassanxiety = base + noise

    return pd.DataFrame(
        {
            "dassanxiety": dassanxiety,
            "Extraversion": extraversion,
            "Agreeableness": agreeableness,
            "Conscientiousness": conscientiousness,
            "Emotional Stability": emotional_stability,
            "Openness": openness,
        }
    )


def test_run_personality_anova_for_dassanxiety_structure():
    """
    The personality ANOVA function should return a dict with one entry per
    TIPI trait, each containing at least: f_stat, p_value and group_stats.
    """
    df = _dummy_personality_df()

    tipi_traits = [
        "Extraversion",
        "Agreeableness",
        "Conscientiousness",
        "Emotional Stability",
        "Openness",
    ]

    results = run_personality_anova_for_dassanxiety(
        df,
        traits=tipi_traits,
        n_groups=3,
    )

    assert isinstance(results, dict)
    assert set(results.keys()) == set(tipi_traits)

    for trait, res in results.items():
        # Required keys
        assert "f_stat" in res
        assert "p_value" in res
        assert "group_stats" in res

        # Types
        assert isinstance(res["f_stat"], (float, np.floating))
        assert isinstance(res["p_value"], (float, np.floating))
        assert isinstance(res["group_stats"], pd.DataFrame)

        # We expect 2–3 groups (Low/Medium/High) depending on binning
        assert 2 <= len(res["group_stats"]) <= 3