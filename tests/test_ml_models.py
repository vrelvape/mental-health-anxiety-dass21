import numpy as np
import pandas as pd

from src.analysis.ml_models import (
    prepare_rf_data,
    fit_random_forest_anxiety,
    NUMERIC_FEATURES_DEFAULT,
    CATEGORICAL_FEATURES_DEFAULT,
)


def _dummy_rf_df(n: int = 300) -> pd.DataFrame:
    """
    Build a synthetic dataset for testing Random Forest preparation
    and fitting.

    The target 'dassanxiety' has a clear linear signal from
    'dassstress' and 'dassdepression', plus noise.
    """
    rng = np.random.default_rng(123)

    # Psychological variables
    dassstress = rng.normal(0, 1, size=n)
    dassdepression = rng.normal(0, 1, size=n)

    # TIPI traits
    extraversion = rng.normal(0, 1, size=n)
    agreeableness = rng.normal(0, 1, size=n)
    conscientiousness = rng.normal(0, 1, size=n)
    emotional_stability = rng.normal(0, 1, size=n)
    openness = rng.normal(0, 1, size=n)

    # Demographic categorical predictors
    gender = rng.choice(["Female", "Male"], size=n, p=[0.6, 0.4])
    orientation = rng.choice(["Heterosexual", "Bisexual", "Other"], size=n)
    married = rng.choice(["Never married", "Currently married"], size=n)
    urban = rng.choice(["Urban", "Rural"], size=n, p=[0.7, 0.3])
    race = rng.choice(["White", "Asian", "Black", "Other"], size=n)
    education = rng.choice(
        ["High school", "University degree", "Graduate degree"],
        size=n,
    )
    religion = rng.choice(
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

    # Construct anxiety with a clear signal from stress + depression
    noise = rng.normal(0, 0.8, size=n)
    dassanxiety = 0.6 * dassstress + 0.9 * dassdepression + noise

    return pd.DataFrame(
        {
            "dassanxiety": dassanxiety,
            "dassstress": dassstress,
            "dassdepression": dassdepression,
            "Extraversion": extraversion,
            "Agreeableness": agreeableness,
            "Conscientiousness": conscientiousness,
            "Emotional Stability": emotional_stability,
            "Openness": openness,
            "gender_clean": gender,
            "orientation_clean": orientation,
            "married_clean": married,
            "urban_clean": urban,
            "race_clean": race,
            "education_clean": education,
            "religion_grouped": religion,
            "age_group": age_group,
        }
    )


def test_prepare_rf_data_returns_valid_X_y():
    """
    prepare_rf_data should return a non-empty numeric feature matrix X
    and a target vector y with matching length, and no missing values.
    """
    df = _dummy_rf_df()
    X, y = prepare_rf_data(
        df,
        numeric_features=NUMERIC_FEATURES_DEFAULT,
        categorical_features=CATEGORICAL_FEATURES_DEFAULT,
    )

    # Basic checks
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert X.shape[1] > 0
    assert not X.isna().any().any()
    assert not y.isna().any()


def test_fit_random_forest_anxiety_produces_importances_and_metrics():
    """
    fit_random_forest_anxiety should fit a model with:
    - a non-empty, normalized feature_importances table
    - valid float metrics (R² and RMSE)
    - positive test R² on synthetic data with strong signal.
    """
    df = _dummy_rf_df()
    result = fit_random_forest_anxiety(
        df,
        numeric_features=NUMERIC_FEATURES_DEFAULT,
        categorical_features=CATEGORICAL_FEATURES_DEFAULT,
        random_state=42,
        n_estimators=100,
        max_depth=None,
        test_size=0.25,
    )

    # Feature importance DataFrame
    fi = result.feature_importances
    assert isinstance(fi, pd.DataFrame)
    assert not fi.empty
    assert set(fi.columns) == {"feature", "importance"}
    assert (fi["importance"] >= 0).all()
    # Should be normalized to 1 (within tolerance)
    assert abs(fi["importance"].sum() - 1.0) < 1e-6

    # Basic metric sanity checks
    assert isinstance(result.r2_train, float)
    assert isinstance(result.r2_test, float)
    assert isinstance(result.rmse_train, float)
    assert isinstance(result.rmse_test, float)

    # With strong signal in the synthetic data, test R² should be > 0
    assert result.r2_test > 0.0