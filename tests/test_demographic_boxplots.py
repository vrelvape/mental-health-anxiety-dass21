def test_generate_demographic_boxplots_uses_temp_directory(tmp_path, monkeypatch):
    """
    Ensure demographic boxplots are written into a temporary directory
    rather than the real results folder.
    """
    tmp_results = tmp_path / "results"
    tmp_figures = tmp_results / "figures"
    tmp_tables = tmp_results / "tables"
    tmp_figures.mkdir(parents=True, exist_ok=True)
    tmp_tables.mkdir(parents=True, exist_ok=True)

    # Patch config paths
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_results, raising=False)
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_figures, raising=False)
    monkeypatch.setattr(config, "TABLES_DIR", tmp_tables, raising=False)

    from src.viz import demographic_boxplots as viz_mod
    monkeypatch.setattr(viz_mod, "FIGURES_DIR", tmp_figures, raising=False)

    df = make_age_group_column(_dummy_df())

    generate_demographic_boxplots(df)

    expected = tmp_figures / "boxplot_dassanxiety_by_gender.png"
    assert expected.exists()