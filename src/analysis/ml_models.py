from __future__ import annotations

"""
Supervised models to predict continuous DASS-Anxiety scores.

All models use the same feature set (DASS Stress/Depression, TIPI traits
and demographic variables) so that their performance can be compared
fairly using R² and RMSE on a held-out test set.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Configuration and simple containers

TARGET_COL = "dassanxiety"

NUMERIC_FEATURES_DEFAULT: List[str] = [
    "dassstress",
    "dassdepression",
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Emotional Stability",
    "Openness",
]

CATEGORICAL_FEATURES_DEFAULT: List[str] = [
    "gender_clean",
    "orientation_clean",
    "married_clean",
    "urban_clean",
    "race_clean",
    "education_clean",
    "religion_grouped",
    "age_group",
]


@dataclass
class RandomForestAnxietyResult:
    """Container for the Random Forest anxiety model result."""

    model: RandomForestRegressor
    feature_importances: pd.DataFrame
    r2_train: float
    r2_test: float
    rmse_train: float
    rmse_test: float


@dataclass
class ModelPerf:
    """
    Simple container for performance metrics of a single model.

    Used to store and compare the behaviour of different algorithms
    trained on the same task.
    """

    name: str
    r2_train: float
    r2_test: float
    rmse_train: float
    rmse_test: float


# Preprocessing helpers (shared by all models)

def prepare_rf_data(
    df: pd.DataFrame,
    numeric_features: Sequence[str] = NUMERIC_FEATURES_DEFAULT,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES_DEFAULT,
    target_col: str = TARGET_COL,
    no_response_label: str | None = "No response",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build the feature matrix X and target vector y for all anxiety models.

    Steps:
    - Select available numeric and categorical predictors.
    - Optionally drop rows with explicit "No response" labels.
    - One-hot encode all categorical predictors.
    - Drop rows containing missing values in any used column.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # Keep only features that actually exist
    available_numeric = [c for c in numeric_features if c in df.columns]
    available_categorical = [c for c in categorical_features if c in df.columns]

    if not available_numeric and not available_categorical:
        raise ValueError("No valid predictor columns found.")

    cols_to_use = available_numeric + available_categorical + [target_col]
    sub = df[cols_to_use].copy()

    # Remove explicit "No response" from categoricals
    if no_response_label is not None:
        for col in available_categorical:
            if col in sub.columns:
                sub = sub[sub[col] != no_response_label]

    # Drop remaining missing values
    sub = sub.dropna()

    # Numerical part
    if available_numeric:
        X_num = sub[available_numeric].astype(float)
    else:
        X_num = pd.DataFrame(index=sub.index)

    # Categorical part: one-hot encoding
    if available_categorical:
        X_cat = pd.get_dummies(
            sub[available_categorical].astype("category"),
            drop_first=True,
        )
    else:
        X_cat = pd.DataFrame(index=sub.index)

    X = pd.concat([X_num, X_cat], axis=1)
    y = sub[target_col].astype(float)

    if X.empty:
        raise ValueError("After preprocessing, feature matrix X is empty.")

    return X, y


def _compute_gender_sample_weight(
    df: pd.DataFrame,
    y_index: pd.Index,
    gender_col: str = "gender_clean",
) -> pd.Series:
    """
    Compute inverse-frequency sample weights to reduce gender imbalance.

    If the gender column is missing or empty, return uniform weights = 1.0.
    """
    if gender_col not in df.columns:
        return pd.Series(1.0, index=y_index)

    gender_aligned = df.loc[y_index, gender_col]
    gender_counts = gender_aligned.value_counts(dropna=True)

    if gender_counts.empty:
        return pd.Series(1.0, index=y_index)

    gender_freq = gender_counts / gender_counts.sum()
    inv_freq = 1.0 / gender_freq

    weights = gender_aligned.map(inv_freq)
    weights = pd.to_numeric(weights, errors="coerce").fillna(1.0)

    return weights


# Random Forest model

def fit_random_forest_anxiety(
    df: pd.DataFrame,
    numeric_features: Sequence[str] = NUMERIC_FEATURES_DEFAULT,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES_DEFAULT,
    target_col: str = TARGET_COL,
    *,
    random_state: int = 42,
    n_estimators: int = 500,
    max_depth: int | None = None,
    test_size: float = 0.2,
    no_response_label: str | None = "No response",
) -> RandomForestAnxietyResult:
    """
    Fit a RandomForestRegressor to predict continuous DASS-Anxiety scores.

    The model uses:
    - DASS Stress / Depression,
    - TIPI traits,
    - demographic predictors (one-hot encoded),
    and is trained with inverse-frequency gender sample weights.
    """
    X, y = prepare_rf_data(
        df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_col=target_col,
        no_response_label=no_response_label,
    )

    # Gender-based sample weights (inverse frequency)
    sample_weight = _compute_gender_sample_weight(df, y.index)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    sample_weight_train = sample_weight.loc[y_train.index]

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train, sample_weight=sample_weight_train)

    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))

    # Feature importances
    importances = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": rf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    total = importances["importance"].sum()
    if total > 0:
        importances["importance"] = importances["importance"] / total

    importances = importances.reset_index(drop=True)

    return RandomForestAnxietyResult(
        model=rf,
        feature_importances=importances,
        r2_train=float(r2_train),
        r2_test=float(r2_test),
        rmse_train=rmse_train,
        rmse_test=rmse_test,
    )


# Linear Regression baseline

def fit_linear_anxiety_model(
    df: pd.DataFrame,
    numeric_features: Sequence[str] = NUMERIC_FEATURES_DEFAULT,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES_DEFAULT,
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ModelPerf:
    """
    Baseline linear regression model for continuous DASS-Anxiety.

    Uses the same preprocessed features as the Random Forest model,
    but without sample weighting.
    """
    X, y = prepare_rf_data(
        df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_col=target_col,
        no_response_label="No response",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))

    return ModelPerf(
        name="Linear Regression",
        r2_train=float(r2_train),
        r2_test=float(r2_test),
        rmse_train=rmse_train,
        rmse_test=rmse_test,
    )


# Gradient Boosting model

def fit_gb_anxiety_model(
    df: pd.DataFrame,
    numeric_features: Sequence[str] = NUMERIC_FEATURES_DEFAULT,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES_DEFAULT,
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ModelPerf:
    """
    Gradient Boosting regressor for continuous DASS-Anxiety.

    Uses the same features and train/test split strategy as the
    linear and Random Forest models, which makes the comparison
    of R² and RMSE more meaningful.
    """
    X, y = prepare_rf_data(
        df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_col=target_col,
        no_response_label="No response",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    gb = GradientBoostingRegressor(random_state=random_state)
    gb.fit(X_train, y_train)

    y_pred_train = gb.predict(X_train)
    y_pred_test = gb.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))

    return ModelPerf(
        name="Gradient Boosting",
        r2_train=float(r2_train),
        r2_test=float(r2_test),
        rmse_train=rmse_train,
        rmse_test=rmse_test,
    )


# Model comparison helper

def compare_anxiety_models(
    df: pd.DataFrame,
    numeric_features: Sequence[str] = NUMERIC_FEATURES_DEFAULT,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES_DEFAULT,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """
    Fit the three models on the same task and return a comparison table.

    The returned DataFrame contains one row per model with:
    - model name,
    - R² on train and test sets,
    - RMSE on train and test sets.
    """
    perfs: list[dict] = []

    # 1) Linear Regression (baseline)
    lin_perf = fit_linear_anxiety_model(
        df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_col=target_col,
    )
    perfs.append(lin_perf.__dict__)

    # 2) Random Forest (with gender weighting)
    rf_result = fit_random_forest_anxiety(
        df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_col=target_col,
    )
    perfs.append(
        {
            "name": "Random Forest",
            "r2_train": rf_result.r2_train,
            "r2_test": rf_result.r2_test,
            "rmse_train": rf_result.rmse_train,
            "rmse_test": rf_result.rmse_test,
        }
    )

    # 3) Gradient Boosting
    gb_perf = fit_gb_anxiety_model(
        df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_col=target_col,
    )
    perfs.append(gb_perf.__dict__)

    perf_df = pd.DataFrame(perfs)
    return perf_df