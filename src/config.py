"""
Central configuration module for project paths.

This file centralizes all important directories used across the project,
such as data, results, and figures. Other modules import these constants
to ensure consistent file handling across the codebase.
"""

from pathlib import Path

# Base directory of the project (folder containing the main README.md)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directory
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "mental_health_survey.csv"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Geographic data directory
GEO_DATA_DIR = DATA_DIR / "geo"
WORLD_MAP_PATH = GEO_DATA_DIR / "world_countries.geojson"

def ensure_directories_exist() -> None:
    """
    Ensure that all required directories for the project exist.

    This function is safe to call multiple times. It creates data, results,
    figures and tables directories if they are missing. This helps prevent
    runtime errors when saving figures or exported tables.
    """
    directories = [DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)