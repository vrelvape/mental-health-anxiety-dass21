import pandas as pd

from src.data_loading import load_data
from src.cleaning import full_cleaning_pipeline


def test_full_cleaning_pipeline_basic_structure():
    """
    Integration test for the full cleaning pipeline on the real dataset.

    This test checks that:
    - the raw data can be loaded
    - the pipeline runs without errors
    - the returned object has the expected attributes
    - the main DataFrames are non-empty and consistent
    """
    raw_df = load_data()
    assert isinstance(raw_df, pd.DataFrame)
    assert not raw_df.empty

    cleaned = full_cleaning_pipeline(raw_df)

    # The cleaned object should expose the expected attributes
    for attr in [
        "clean_df",
        "clean_valid_df",
        "analysis_df",
        "analysis_standardized_df",
    ]:
        assert hasattr(cleaned, attr), f"Missing attribute '{attr}' on cleaned object."

    clean_df = cleaned.clean_df
    clean_valid_df = cleaned.clean_valid_df
    analysis_df = cleaned.analysis_df
    analysis_std = cleaned.analysis_standardized_df

    # All should be DataFrames and non-empty
    for name, df in [
        ("clean_df", clean_df),
        ("clean_valid_df", clean_valid_df),
        ("analysis_df", analysis_df),
        ("analysis_standardized_df", analysis_std),
    ]:
        assert isinstance(df, pd.DataFrame), f"{name} is not a DataFrame."
        assert not df.empty, f"{name} is empty."

    # Analysis standardized df should have the same number of rows as analysis df
    assert len(analysis_std) == len(analysis_df)

    # Check that some key columns are present in the analysis DataFrames
    expected_cols = [
        "dassanxiety",
        "dassstress",
        "dassdepression",
        "gender_clean",
        "age_clean",
    ]
    missing_in_analysis = [c for c in expected_cols if c not in analysis_df.columns]
    missing_in_std = [c for c in expected_cols if c not in analysis_std.columns]

    assert not missing_in_analysis, f"Missing columns in analysis_df: {missing_in_analysis}"
    assert not missing_in_std, f"Missing columns in analysis_standardized_df: {missing_in_std}"
