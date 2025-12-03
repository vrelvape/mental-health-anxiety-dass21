import pandas as pd
import src.config as config

from src.viz.ml_feature_importance import plot_rf_feature_importances


def test_plot_rf_feature_importances_creates_file(tmp_path, monkeypatch):
    """
    plot_rf_feature_importances should generate a PNG file
    in a temporary figures directory.
    """
    # Fake feature importances DataFrame
    importances = pd.DataFrame(
        {
            "feature": ["dassstress", "dassdepression", "Extraversion"],
            "importance": [0.5, 0.3, 0.2],
        }
    )

    # Temporary results/figures directory
    tmp_results = tmp_path / "results"
    tmp_figures = tmp_results / "figures"
    tmp_figures.mkdir(parents=True, exist_ok=True)

    # Redirect FIGURES_DIR and RESULTS_DIR to the temporary paths
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_results, raising=False)
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_figures, raising=False)

    # Ensure the viz module uses the patched FIGURES_DIR
    from src.viz import ml_feature_importance as viz_mod
    monkeypatch.setattr(viz_mod, "FIGURES_DIR", tmp_figures, raising=False)

    filename = "rf_feature_importances_test.png"

    plot_rf_feature_importances(
        importances=importances,
        top_n=3,
        filename=filename,
        title="Test RF Feature Importances",
    )

    # The PNG file should now exist in the temporary figures directory
    assert (tmp_figures / filename).exists()