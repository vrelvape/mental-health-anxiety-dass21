import pandas as pd
import src.config as config

from src.analysis.nonresponse import compute_nonresponse_table, plot_nonresponse_bar


def _dummy_nonresponse_df() -> pd.DataFrame:
    """
    Small synthetic dataset for testing non-response analysis.

    We include some "No response" values across several variables
    to exercise the non-response logic.
    """
    return pd.DataFrame(
        {
            "gender_clean": ["Male", "Female", "No response", "Female"],
            "education_clean": [
                "High school",
                "University",
                "No response",
                "Graduate",
            ],
            "urban_clean": ["Urban", "Rural", "No response", "Urban"],
            "religion_clean": ["Agnostic", "No response", "Christian", "Atheist"],
            "orientation_clean": ["Heterosexual", "No response", "Bisexual", "Heterosexual"],
            "race_clean": ["White", "No response", "Asian", "Black"],
            "hand_clean": ["Right", "Left", "No response", "Right"],
            "voted_clean": ["Yes", "No response", "No", "Yes"],
            "married_clean": ["Never married", "No response", "Currently married", "Never married"],
            "engnat_clean": ["Yes", "No response", "No", "Yes"],
            "screensize_clean": ["Laptop", "Desktop", "No response", "Phone"],
            "uniquenetworklocation_clean": ["Home", "Work", "No response", "Other"],
            "source_clean": ["Student", "No response", "Online", "Student"],
        }
    )


def test_compute_nonresponse_table_basic_structure():
    """
    compute_nonresponse_table should return a DataFrame containing one row
    per requested variable, with at least one numeric column for counts/percentages.
    """
    df = _dummy_nonresponse_df()

    variables_to_check = [
        "gender_clean",
        "education_clean",
        "urban_clean",
        "religion_clean",
        "orientation_clean",
        "race_clean",
        "hand_clean",
        "voted_clean",
        "married_clean",
        "engnat_clean",
        "screensize_clean",
        "uniquenetworklocation_clean",
        "source_clean",
    ]

    table = compute_nonresponse_table(
        df,
        variables=variables_to_check,
        no_response_label="No response",
    )

    assert isinstance(table, pd.DataFrame)
    assert not table.empty

    # We expect at most one row per variable
    assert len(table) <= len(variables_to_check)

    # Check that all rows correspond to one of the requested variables
    if "variable" in table.columns:
        vars_in_table = set(table["variable"])
    else:
        vars_in_table = set(table.index)

    assert vars_in_table.issubset(set(variables_to_check))

    # There should be at least one numeric column (counts/percentages)
    numeric_cols = table.select_dtypes(include="number").columns
    assert len(numeric_cols) >= 1


def test_plot_nonresponse_bar_writes_to_temp_directory(tmp_path, monkeypatch):
    """
    plot_nonresponse_bar should create a figure file in a temporary
    figures directory, not in the real project results folder.
    """
    tmp_results = tmp_path / "results"
    tmp_figures = tmp_results / "figures"
    tmp_results.mkdir(parents=True, exist_ok=True)
    tmp_figures.mkdir(parents=True, exist_ok=True)

    # Patch config paths
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_results, raising=False)
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_figures, raising=False)

    from src.analysis import nonresponse as nonresp_mod
    monkeypatch.setattr(nonresp_mod, "FIGURES_DIR", tmp_figures, raising=False)

    df = _dummy_nonresponse_df()

    variables_to_check = [
        "gender_clean",
        "education_clean",
        "urban_clean",
        "religion_clean",
        "orientation_clean",
        "race_clean",
        "hand_clean",
        "voted_clean",
        "married_clean",
        "engnat_clean",
        "screensize_clean",
        "uniquenetworklocation_clean",
        "source_clean",
    ]

    table = compute_nonresponse_table(
        df,
        variables=variables_to_check,
        no_response_label="No response",
    )

    filename = "nonresponse_rates_top10_test.png"
    plot_nonresponse_bar(
        table,
        top_n=10,
        filename=filename,
    )

    assert (tmp_figures / filename).exists()
