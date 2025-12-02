"""
Data loading utilities for the mental health survey project.

This module provides a single, well-defined entry point to obtain the
raw survey DataFrame used in the rest of the analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from .config import RAW_DATA_PATH, ensure_directories_exist


def load_data_from_csv(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the raw survey data from a local CSV (or TSV) file.

    The function tries to automatically infer the delimiter so that it works
    with comma-separated, semicolon-separated or tab-separated files.

    Parameters
    ----------
    path : Path, optional
        Path to the data file. If None, the default RAW_DATA_PATH defined
        in config.py is used.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the raw survey data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given (or default) path.
    """
    ensure_directories_exist()

    csv_path = path or RAW_DATA_PATH

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Data file not found at {csv_path}. "
            "Place the survey data file there or provide a custom path "
            "when calling load_data_from_csv()."
        )

    # Use the Python engine with sep=None to let pandas infer the delimiter.
    df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8")

    return df


def load_data(
    source: Literal["local"] = "local",
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    High-level data loading function.

    This function wraps lower-level loading utilities and exposes a clean
    interface for the rest of the project. For now, only local CSV loading
    is supported, but the interface can be extended if needed.

    Parameters
    ----------
    source : {"local"}
        Source from which to load the data. Currently only "local" is
        implemented.
    path : Path, optional
        Optional path to a local CSV file. If None, RAW_DATA_PATH is used.

    Returns
    -------
    pd.DataFrame
        Raw survey data as a pandas DataFrame.

    Raises
    ------
    ValueError
        If an unsupported source is provided.
    """
    if source == "local":
        return load_data_from_csv(path=path)

    raise ValueError("Only 'local' source is supported in load_data().")