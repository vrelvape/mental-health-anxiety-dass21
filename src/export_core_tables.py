"""
Export core tables for the DASS-21 anxiety project.

This module:
- charge les données brutes,
- lance le pipeline de nettoyage complet,
- construit les tables principales (sévérité, géographie, ANOVA, ML),
- exporte chaque table en CSV, LaTeX et PDF.

Il est appelé depuis src.main via export_all_core_tables().
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data_loading import load_data
from src.cleaning import full_cleaning_pipeline
from src.config import TABLES_DIR, ensure_directories_exist
from src.analysis.anova_demographics import run_demographic_anova_for_dassanxiety
from src.analysis.anova_personality import run_personality_anova_for_dassanxiety
from src.analysis.ml_models import fit_random_forest_anxiety, compare_anxiety_models


# Sous-dossiers pour les différents formats de tables
CSV_DIR = TABLES_DIR / "csv"
LATEX_DIR = TABLES_DIR / "latex"
PDF_DIR = TABLES_DIR / "pdf"


def init_table_dirs() -> None:
    """
    Crée la structure de répertoires utilisée pour les exports de tables.

    - results/tables/
    - results/tables/csv
    - results/tables/latex
    - results/tables/pdf
    """
    ensure_directories_exist()
    for d in (CSV_DIR, LATEX_DIR, PDF_DIR):
        d.mkdir(parents=True, exist_ok=True)


def export_df_to_pdf(df: pd.DataFrame, path: Path, title: str | None = None) -> None:
    """
    Rendre un DataFrame (petit / moyen) en PDF via matplotlib.

    Le texte dans les cellules est "wrapé" pour éviter que les labels
    trop longs ne débordent, afin de garder une table lisible.
    """
    import textwrap

    def wrap_cell_text(x, max_width: int = 18) -> str:
        if isinstance(x, str):
            return "\n".join(textwrap.wrap(x, max_width))
        return x

    df_to_show = df.copy().applymap(lambda x: wrap_cell_text(x, max_width=18))

    # Arrondir les colonnes numériques pour un affichage plus propre
    for col in df_to_show.columns:
        if pd.api.types.is_numeric_dtype(df_to_show[col]):
            df_to_show[col] = df_to_show[col].round(3)

    n_rows, n_cols = df_to_show.shape

    # Heuristique simple pour la taille de la figure
    fig_width = max(7, 0.9 * n_cols)
    fig_height = max(4, 0.5 * n_rows)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    # Wrap des en-têtes de colonnes
    col_labels = [
        wrap_cell_text(c, max_width=18) if isinstance(c, str) else c
        for c in df_to_show.columns
    ]

    table = ax.table(
        cellText=df_to_show.values,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.4)
    table.auto_set_column_width(list(range(n_cols)))

    # Style de la ligne d'en-tête et bordures
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eaeaea")

    if title:
        ax.set_title(title, fontsize=11, pad=8)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def export_table_all_formats(
    df: pd.DataFrame,
    base_name: str,
    index: bool = False,
    title_for_pdf: str | None = None,
) -> None:
    """
    Exporter un DataFrame en CSV, LaTeX et PDF.

    - CSV : results/tables/csv/{base_name}.csv
    - LaTeX : results/tables/latex/{base_name}.tex
    - PDF : results/tables/pdf/{base_name}.pdf
    """
    csv_path = CSV_DIR / f"{base_name}.csv"
    latex_path = LATEX_DIR / f"{base_name}.tex"
    pdf_path = PDF_DIR / f"{base_name}.pdf"

    # CSV + LaTeX
    df.to_csv(csv_path, index=index)
    latex_str = df.to_latex(index=index)
    latex_path.write_text(latex_str, encoding="utf-8")

    # Pour le PDF on reset l'index pour obtenir une table propre
    df_for_pdf = df.reset_index() if index else df.reset_index(drop=True)
    export_df_to_pdf(df_for_pdf, pdf_path, title=title_for_pdf or base_name)

    print(f"Saved table in all formats for: {base_name}")


def export_large_table_csv_only(df: pd.DataFrame, base_name: str, index: bool = False) -> None:
    """
    Exporter uniquement un CSV pour les très grandes tables
    (clean_df, analysis_df, etc.).

    Ces tables sont utiles pour inspection, mais peu adaptées pour LaTeX/PDF.
    """
    csv_path = CSV_DIR / f"{base_name}.csv"
    df.to_csv(csv_path, index=index)
    print(f"Saved LARGE table (CSV only) for: {base_name}")


def add_age_group_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajouter une colonne catégorielle age_group dérivée de age_clean.

    Bins :
        13–17, 18–24, 25–34, 35–49, 50+.
    """
    bins = [12, 17, 24, 34, 49, 100]
    labels = [
        "Adolescents(13–17)",
        "Young Adults(18–24)",
        "Emerging Adults(25–34)",
        "Middle-aged Adults(35–49)",
        "Seniors(50+)",
    ]

    out = df.copy()
    if "age_clean" in out.columns:
        out["age_group"] = pd.cut(
            out["age_clean"],
            bins=bins,
            labels=labels,
            right=True,
            include_lowest=True,
        )
    return out


def build_anxiety_severity_tables(analysis_std: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Construire toutes les tables liées aux catégories de sévérité d'anxiété :

    - distribution globale des catégories de sévérité,
    - distribution de sévérité par genre,
    - distribution de sévérité par tranche d'âge.
    """
    df = add_age_group_column(analysis_std).copy()

    bins = [0, 14, 18, 26, 34, 56]
    labels = ["Normal", "Mild", "Moderate", "Severe", "Extremely Severe"]

    df["anxiety_severity"] = pd.cut(
        df["dassanxiety"],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True,
    )

    tables: dict[str, pd.DataFrame] = {}

    # Résumé global
    severity_counts = df["anxiety_severity"].value_counts().sort_index()
    severity_percent = (severity_counts / len(df) * 100).round(2)
    tables["anxiety_severity_summary"] = pd.DataFrame(
        {"Count": severity_counts, "Percentage (%)": severity_percent}
    )

    # Par genre
    if "gender_clean" in df.columns:
        by_gender_counts = (
            df.groupby(["gender_clean", "anxiety_severity"], observed=False)
            .size()
            .unstack()
        )
        by_gender_percent = (
            by_gender_counts.apply(lambda x: x / x.sum() * 100, axis=1).round(2)
        )
        tables["anxiety_severity_by_gender_counts"] = by_gender_counts
        tables["anxiety_severity_by_gender_percent"] = by_gender_percent

    # Par tranche d'âge
    if "age_group" in df.columns:
        by_age_counts = (
            df.groupby(["age_group", "anxiety_severity"], observed=False)
            .size()
            .unstack()
        )
        by_age_percent = (
            by_age_counts.apply(lambda x: x / x.sum() * 100, axis=1).round(2)
        )
        tables["anxiety_severity_by_agegroup_counts"] = by_age_counts
        tables["anxiety_severity_by_agegroup_percent"] = by_age_percent

    return tables


def build_geo_tables(analysis_std: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Construire les tables géographiques :

    - moyenne d'anxiété par continent,
    - top 10 pays par anxiété moyenne,
    - top 10 pays par taille d'échantillon.
    """
    tables: dict[str, pd.DataFrame] = {}

    # Moyenne par continent
    if "continent" in analysis_std.columns:
        mean_by_continent = (
            analysis_std.groupby("continent", observed=True)["dassanxiety"]
            .mean()
            .reset_index()
            .rename(columns={"dassanxiety": "mean_anxiety"})
            .sort_values("mean_anxiety", ascending=False)
        )
        tables["mean_anxiety_by_continent"] = mean_by_continent

    # Tables par pays (top 10)
    if "country_name" in analysis_std.columns:
        mean_by_country = (
            analysis_std.groupby("country_name", observed=False)["dassanxiety"]
            .mean()
            .reset_index()
            .rename(columns={"dassanxiety": "mean_anxiety"})
            .sort_values("mean_anxiety", ascending=False)
        )
        tables["top10_mean_anxiety_by_country"] = mean_by_country.head(10)

        counts = analysis_std["country_name"].value_counts().reset_index()
        counts.columns = ["country_name", "n_participants"]
        tables["top10_countries_by_participants"] = counts.head(10)

    return tables


def build_anova_tables(analysis_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Construire les tables de résumés ANOVA pour :

    - prédicteurs démographiques (genre, orientation, âge, etc.),
    - traits de personnalité TIPI.

    Chaque ligne contient :
    - F statistic,
    - p-value.
    """
    tables: dict[str, pd.DataFrame] = {}

    demographic_factors = [
        "gender_clean",
        "orientation_clean",
        "married_clean",
        "urban_clean",
        "race_clean",
        "education_clean",
        "religion_grouped",
        "age_group",
    ]

    tipi_traits = [
        "Extraversion",
        "Agreeableness",
        "Conscientiousness",
        "Emotional Stability",
        "Openness",
    ]

    # ANOVA démographiques
    demo_results = run_demographic_anova_for_dassanxiety(analysis_df, demographic_factors)
    demo_rows = [
        {"factor": factor, "F": res["f_stat"], "p_value": res["p_value"]}
        for factor, res in demo_results.items()
    ]
    if demo_rows:
        tables["anova_demographics_summary"] = pd.DataFrame(demo_rows).set_index("factor")

    # ANOVA personnalité
    perso_results = run_personality_anova_for_dassanxiety(
        analysis_df,
        traits=tipi_traits,
        n_groups=3,
    )
    perso_rows = [
        {"trait": trait, "F": res["f_stat"], "p_value": res["p_value"]}
        for trait, res in perso_results.items()
    ]
    if perso_rows:
        tables["anova_personality_summary"] = pd.DataFrame(perso_rows).set_index("trait")

    return tables


def export_all_core_tables() -> None:
    """
    Fonction haut niveau qui orchestre tous les exports de tables.

    Étapes :
    1. Charger les données brutes et lancer le pipeline de nettoyage.
    2. Exporter les grandes tables (clean_df, analysis_df, ...) en CSV uniquement.
    3. Construire et exporter :
       - tables de sévérité,
       - tables géographiques,
       - résumés ANOVA,
       - importances du Random Forest,
       - comparaison des trois modèles de régression.
    """
    init_table_dirs()

    raw_df = load_data()
    cleaned = full_cleaning_pipeline(raw_df)

    # 1) Tables de nettoyage → CSV uniquement
    export_large_table_csv_only(cleaned.clean_df, "clean_df", index=False)
    export_large_table_csv_only(cleaned.clean_valid_df, "clean_valid_df", index=False)
    export_large_table_csv_only(cleaned.analysis_df, "analysis_df", index=False)
    export_large_table_csv_only(
        cleaned.analysis_standardized_df,
        "analysis_standardized_df",
        index=False,
    )

    analysis_std = cleaned.analysis_standardized_df
    analysis_df = cleaned.analysis_df

    # 2) Tables de sévérité → 3 formats
    severity_tables = build_anxiety_severity_tables(analysis_std)
    for name, df in severity_tables.items():
        export_table_all_formats(df, name, index=True)

    # 3) Tables géographiques → 3 formats
    geo_tables = build_geo_tables(analysis_std)
    for name, df in geo_tables.items():
        export_table_all_formats(df, name, index=False)

    # 4) Tables ANOVA → 3 formats
    anova_tables = build_anova_tables(analysis_df)
    for name, df in anova_tables.items():
        export_table_all_formats(df, name, index=True)

    # 5) Importances du Random Forest → 3 formats
    rf_result = fit_random_forest_anxiety(analysis_std)
    rf_importances = rf_result.feature_importances
    export_table_all_formats(
        rf_importances,
        base_name="rf_feature_importances",
        index=False,
        title_for_pdf="Random Forest feature importances for DASS anxiety",
    )

    # 6) Comparaison des trois modèles → 3 formats
    model_comp_df = compare_anxiety_models(analysis_std)
    export_table_all_formats(
        model_comp_df,
        base_name="model_comparison",
        index=False,
        title_for_pdf="Comparison of regression models for DASS anxiety",
    )

    print("\nAll tables exported to:", TABLES_DIR)


def main() -> None:
    """Point d'entrée en ligne de commande pour exporter toutes les tables."""
    export_all_core_tables()


if __name__ == "__main__":
    main()