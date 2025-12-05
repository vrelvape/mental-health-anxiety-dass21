from pathlib import Path

import src.config as config


def test_ensure_directories_exist_creates_directories(tmp_path, monkeypatch):
    """
    ensure_directories_exist should create the results/figures/tables directories.

    This test redirects all paths to a temporary location so that the real
    project directories are not touched.
    """
    # Prepare temporary directories
    tmp_results = tmp_path / "results"
    tmp_figures = tmp_results / "figures"
    tmp_tables = tmp_results / "tables"
    tmp_docs = tmp_path / "docs"

    # Patch paths in config if they exist
    if hasattr(config, "RESULTS_DIR"):
        monkeypatch.setattr(config, "RESULTS_DIR", tmp_results, raising=False)
    if hasattr(config, "FIGURES_DIR"):
        monkeypatch.setattr(config, "FIGURES_DIR", tmp_figures, raising=False)
    if hasattr(config, "TABLES_DIR"):
        monkeypatch.setattr(config, "TABLES_DIR", tmp_tables, raising=False)
    if hasattr(config, "DOCS_DIR"):
        monkeypatch.setattr(config, "DOCS_DIR", tmp_docs, raising=False)

    # Call the function under test
    config.ensure_directories_exist()

    # Assertions: directories should exist in the temp location
    assert tmp_results.exists() and tmp_results.is_dir()
    if hasattr(config, "FIGURES_DIR"):
        assert tmp_figures.exists() and tmp_figures.is_dir()
    if hasattr(config, "TABLES_DIR"):
        assert tmp_tables.exists() and tmp_tables.is_dir()
    if hasattr(config, "DOCS_DIR"):
        assert tmp_docs.exists() and tmp_docs.is_dir()