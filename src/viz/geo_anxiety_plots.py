import geopandas as gpd
import pycountry
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR, TABLES_DIR, WORLD_MAP_PATH, ensure_directories_exist


def get_iso_alpha_3(name: str) -> str | None:
    """
    Return ISO alpha-3 country code for a given country name.

    Parameters
    ----------
    name : str
        Country name as found in the survey data.

    Returns
    -------
    str or None
        ISO alpha-3 code if found, otherwise None.
    """
    try:
        return pycountry.countries.lookup(name).alpha_3
    except Exception:
        return None


def compute_mean_anxiety_by_country(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean anxiety scores per country and attach ISO alpha-3 codes.

    Parameters
    ----------
    analysis_df : pd.DataFrame
        Analysis DataFrame containing 'country_name' and 'dassanxiety'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'country_name', 'mean_anxiety', 'iso_alpha_3'.
    """
    group = (
        analysis_df.groupby("country_name", observed=False)["dassanxiety"]
        .mean()
        .reset_index()
    )
    group.columns = ["country_name", "mean_anxiety"]

    group["iso_alpha_3"] = group["country_name"].apply(get_iso_alpha_3)
    return group


def plot_world_anxiety_map(
    mean_anxiety_by_country: pd.DataFrame,
    world_gdf: gpd.GeoDataFrame | None = None,
) -> None:
    """
    Plot a world choropleth of mean anxiety by country, using a local GeoJSON.

    The base map is loaded from WORLD_MAP_PATH (committed in the repository),
    so no internet connection is required at runtime.
    """
    ensure_directories_exist()

    if world_gdf is None:
        try:
            world_gdf = gpd.read_file(WORLD_MAP_PATH)
        except Exception as exc:
            print(f"Could not load local world map (skipping choropleth): {exc}")
            return

    world_gdf = world_gdf.copy()

    # Try several common ISO-3 column names, including this file's:
    possible_iso_cols = [
        "iso_a3",
        "ISO_A3",
        "iso3",
        "ISO3",
        "ISO3166-1-Alpha-3",  
        "id",
    ]
    code_col = None
    for col in possible_iso_cols:
        if col in world_gdf.columns:
            code_col = col
            break

    if code_col is None:
        print(
            "World GeoDataFrame has no suitable ISO-3 column "
            f"(available columns: {list(world_gdf.columns)}); skipping map."
        )
        return

    merged = world_gdf.merge(
        mean_anxiety_by_country,
        how="left",
        left_on=code_col,
        right_on="iso_alpha_3",
    )

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    merged.plot(
        column="mean_anxiety",
        cmap="Reds",
        linewidth=0.5,
        ax=ax,
        edgecolor="0.8",
        legend=True,
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )

    ax.set_title("Mean Anxiety Score by Country", fontsize=16)
    ax.axis("off")
    plt.tight_layout()

    out_path = FIGURES_DIR / "worldmap_anxiety_by_country.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved world map to: {out_path}")

def compute_continent_stats(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean anxiety and sample size by continent.

    Parameters
    ----------
    analysis_df : pd.DataFrame
        Analysis DataFrame containing 'continent' and 'dassanxiety'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['continent', 'mean', 'count'] sorted by mean.
    """
    stats = (
        analysis_df.groupby("continent", observed=True)["dassanxiety"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    return stats


def save_continent_table_and_plot(
    continent_stats: pd.DataFrame,
    save_csv: bool = True,
) -> None:
    """
    Optionally save a CSV table and always save a barplot
    of mean anxiety by continent.

    Parameters
    ----------
    continent_stats : pd.DataFrame
        Output of compute_continent_stats.
    save_csv : bool
        If True, save the table as CSV in TABLES_DIR.
    """
    ensure_directories_exist()

    if save_csv:
        table_path = TABLES_DIR / "mean_anxiety_by_continent.csv"
        continent_stats.to_csv(table_path, index=False)
        print(f"Saved mean anxiety by continent table to: {table_path}")

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x="continent",
        y="mean",
        data=continent_stats,
        hue="continent",
        palette="pastel",
        legend=False,
    )

    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    for i, row in continent_stats.iterrows():
        ax.text(
            i,
            row["mean"] + 0.8,
            f"n={int(row['count'])}",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

    ax.set_title("Average Anxiety Score (DASS) by Continent", fontsize=14)
    ax.set_xlabel("Continent", fontsize=12)
    ax.set_ylabel("Mean Anxiety Score", fontsize=12)
    ax.set_ylim(0, continent_stats["mean"].max() + 5)
    plt.tight_layout()

    plot_path = FIGURES_DIR / "barplot_anxiety_by_continent.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved continent barplot to: {plot_path}")


def save_top10_mean_anxiety_by_country(
    mean_anxiety_by_country: pd.DataFrame,
) -> None:
    """
    Save CSV for the top 10 countries by highest mean anxiety.

    Parameters
    ----------
    mean_anxiety_by_country : pd.DataFrame
        DataFrame with 'country_name' and 'mean_anxiety'.
    """
    ensure_directories_exist()

    top10 = (
        mean_anxiety_by_country.sort_values("mean_anxiety", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    out_path = TABLES_DIR / "top10_mean_anxiety_by_country.csv"
    top10.to_csv(out_path, index=False)
    print(f"Saved top10 mean anxiety by country to: {out_path}")


def save_top10_countries_by_participants(analysis_df: pd.DataFrame) -> None:
    """
    Save CSV with the top 10 countries by number of participants.

    Parameters
    ----------
    analysis_df : pd.DataFrame
        Analysis DataFrame containing 'country_name'.
    """
    ensure_directories_exist()

    counts = analysis_df["country_name"].value_counts().reset_index()
    counts.columns = ["country_name", "n_participants"]

    top10_counts = counts.head(10)

    out_path = TABLES_DIR / "top10_countries_by_participants.csv"
    top10_counts.to_csv(out_path, index=False)
    print(f"Saved top10 countries by participants to: {out_path}")


def generate_geo_anxiety_outputs(
    analysis_df: pd.DataFrame,
    save_csv: bool = True,
) -> None:
    """
    High-level wrapper to generate all geographic outputs:
    - world map of mean anxiety by country
    - continent barplot (+ optional CSV)
    - optional top10 countries by mean anxiety
    - optional top10 countries by number of participants

    Parameters
    ----------
    analysis_df : pd.DataFrame
        Standardized analysis DataFrame with at least 'country_name',
        'continent' and 'dassanxiety'.
    save_csv : bool
        If True, also save CSV tables in TABLES_DIR.
    """
    ensure_directories_exist()

    mean_by_country = compute_mean_anxiety_by_country(analysis_df)
    plot_world_anxiety_map(mean_by_country)

    continent_stats = compute_continent_stats(analysis_df)
    save_continent_table_and_plot(continent_stats, save_csv=save_csv)

    if save_csv:
        save_top10_mean_anxiety_by_country(mean_by_country)
        save_top10_countries_by_participants(analysis_df)


if __name__ == "__main__":
    from src.data_loading import load_data
    from src.cleaning import full_cleaning_pipeline

    raw = load_data()
    cleaned = full_cleaning_pipeline(raw)
    analysis_std = cleaned.analysis_standardized_df

    generate_geo_anxiety_outputs(analysis_std, save_csv=True)