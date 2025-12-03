import numpy as np
import pandas as pd
import src.config as config

from src.viz import tipi_dass_heatmap as viz_mod


def _dummy_analysis_std_df(n: int = 50) -> pd.DataFrame:
    """Return a small dummy standardized-like DataFrame for the TIPI–DASS heatmap."""
    rng = np.random.default_rng(42)

    return pd.DataFrame(
        {
            "dassanxiety": rng.normal(0, 1, size=n),
            "dassstress": rng.normal(0, 1, size=n),
            "dassdepression": rng.normal(0, 1, size=n),
            "dasssum": rng.normal(0, 1, size=n),
            "Extraversion": rng.normal(0, 1, size=n),
            "Agreeableness": rng.normal(0, 1, size=n),
            "Conscientiousness": rng.normal(0, 1, size=n),
            "Emotional Stability": rng.normal(0, 1, size=n),
            "Openness": rng.normal(0, 1, size=n),
        }
    )


def test_plot_tipi_dass_heatmap_writes_to_temp_directory(tmp_path, monkeypatch):
    """
    plot_tipi_dass_heatmap should create a PNG file in a temporary figures directory.
    """
    # Temporary results/figures dirs
    tmp_results = tmp_path / "results"
    tmp_figures = tmp_results / "figures"
    tmp_figures.mkdir(parents=True, exist_ok=True)

    # Patch config paths
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_results, raising=False)
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_figures, raising=False)

    # Patch FIGURES_DIR inside the visualization module as well
    monkeypatch.setattr(viz_mod, "FIGURES_DIR", tmp_figures, raising=False)

    df = _dummy_analysis_std_df()

    # Call the function: if it crashes, pytest will fail
    viz_mod.plot_tipi_dass_heatmap(df)

    expected_file = tmp_figures / "heatmap_tipi_dass.png"
    assert expected_file.exists()