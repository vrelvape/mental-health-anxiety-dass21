import pandas as pd

from src.data_loading import load_data


def test_load_data_returns_non_empty_dataframe():
    """
    Integration test: load_data should return a non-empty DataFrame
    containing the main raw columns from the mental health survey.
    """
    df = load_data()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    # Core columns expected in the raw dataset
    expected_cols = [
        "dassanxiety",
        "dassstress",
        "dassdepression",
        "gender",
        "age",
    ]
    missing = [c for c in expected_cols if c not in df.columns]

    # For this project we expect all of them to be present
    assert not missing, f"Missing expected raw columns: {missing}"