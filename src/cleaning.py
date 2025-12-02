"""
Data cleaning functions for the mental health survey project.

This module applies the preprocessing steps required to produce:
- raw_df: original dataset as loaded from the file.
- clean_df: cleaned dataset with recoded variables.
- clean_valid_df: cleaned dataset restricted to rows with valid VCL scores.
- analysis_df: reduced subset of columns used for statistical analysis.
- analysis_standardized_df: standardized version of analysis_df.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features import (
    DASS_ITEMS,
    DASS_LABELS,
    TIPI_ITEMS,
    TIPI_LABELS,
    VCL_ITEMS,
    VCL_LABELS,
)


@dataclass
class CleanedData:
    """
    Container object for all cleaned datasets produced by the pipeline.
    """
    raw_df: pd.DataFrame
    clean_df: pd.DataFrame
    clean_valid_df: pd.DataFrame
    analysis_df: pd.DataFrame
    analysis_standardized_df: pd.DataFrame


# Categorical mappings
GENDER_MAP: Dict = {
    1: "Male",
    2: "Female",
    3: "Other",
    0: "No response",
}
EDUCATION_MAP: Dict = {
    1: "Less than high school",
    2: "High school",
    3: "University degree",
    4: "Graduate degree",
    0: "No response",
}
URBAN_MAP: Dict = {
    1: "Rural",
    2: "Suburban",
    3: "Urban",
    0: "No response",
}
RELIGION_MAP: Dict = {
    1: "Agnostic",
    2: "Atheist",
    3: "Buddhist",
    4: "Christian (Catholic)",
    5: "Christian (Mormon)",
    6: "Christian (Protestant)",
    7: "Christian (Other)",
    8: "Hindu",
    9: "Jewish",
    10: "Muslim",
    11: "Sikh",
    12: "Other",
    0: "No response",
}
ORIENTATION_MAP: Dict = {
    1: "Heterosexual",
    2: "Bisexual",
    3: "Homosexual",
    4: "Asexual",
    5: "Other",
    0: "No response",
}
RACE_MAP: Dict = {
    10: "Asian",
    20: "Arab",
    30: "Black",
    40: "Indigenous Australian",
    50: "Native American",
    60: "White",
    70: "Other",
    0: "No response",
}
HAND_MAP: Dict = {
    1: "Right",
    2: "Left",
    3: "Ambidextrous",
    0: "No response",
}
VOTED_MAP: Dict = {
    1: "Yes",
    2: "No",
    0: "No response",
}
MARRIED_MAP: Dict = {
    1: "Never married",
    2: "Currently married",
    3: "Previously married",
    0: "No response",
}
ENGNAT_MAP: Dict = {
    1: "Yes",
    2: "No",
    0: "No response",
}
SCREENSIZE_MAP: Dict = {
    1: "Small screen",
    2: "Large screen",
    0: "No response",
}
UNIQUE_NETWORK_MAP: Dict = {
    1: "Unique",
    2: "Shared",
    0: "No response",
}
SOURCE_MAP: Dict = {
    1: "Front page of the site hosting the survey",
    2: "Google",
    3: "Other or Unknown",
    0: "No response",
}


def recode_categorical_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recode categorical variables and create corresponding *_clean columns.
    """
    df = df.copy()

    recode_instructions = {
        "gender": (GENDER_MAP, "gender_clean"),
        "education": (EDUCATION_MAP, "education_clean"),
        "urban": (URBAN_MAP, "urban_clean"),
        "religion": (RELIGION_MAP, "religion_clean"),
        "orientation": (ORIENTATION_MAP, "orientation_clean"),
        "race": (RACE_MAP, "race_clean"),
        "hand": (HAND_MAP, "hand_clean"),
        "voted": (VOTED_MAP, "voted_clean"),
        "married": (MARRIED_MAP, "married_clean"),
        "engnat": (ENGNAT_MAP, "engnat_clean"),
        "screensize": (SCREENSIZE_MAP, "screensize_clean"),
        "uniquenetworklocation": (
            UNIQUE_NETWORK_MAP,
            "uniquenetworklocation_clean",
        ),
        "source": (SOURCE_MAP, "source_clean"),
    }

    for column, (mapping, new_column) in recode_instructions.items():
        if column in df.columns:
            df[new_column] = df[column].map(mapping)

    return df


def clean_age_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the age variable and create the age_clean column.

    Logic:
    - Values that look like a birth year (1900 <= x <= current_year - 10)
      are converted to ages using (current_year - x).
    - Otherwise we assume the value is already an age.
    - Ages outside [10, 100] are dropped as unrealistic.
    """
    df = df.copy()

    current_year = pd.Timestamp.now().year

    def convert_age(value):
        if 1900 <= value <= (current_year - 10):
            return current_year - value
        return value

    df["age_clean"] = df["age"].apply(convert_age)
    df = df[(df["age_clean"] >= 10) & (df["age_clean"] <= 100)]

    return df


import pycountry
import pycountry_convert as pc_convert


def clean_country_and_continent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and recode country and continent information.

    Steps:
    - Convert ISO Alpha-2 country codes into full English country names.
    - Convert country names into continent names using pycountry-convert.
    - Assign 'Unknown' when the country cannot be resolved.
    """
    df = df.copy()

    def iso_to_country_name(iso_code):
        if pd.isna(iso_code):
            return None
        try:
            country = pycountry.countries.get(alpha_2=str(iso_code))
            return country.name if country else None
        except Exception:
            return None

    def country_to_continent(country_name):
        if pd.isna(country_name):
            return "Unknown"
        try:
            country = pycountry.countries.get(name=country_name)
            if not country:
                return "Unknown"
            alpha2 = country.alpha_2
            continent_code = pc_convert.country_alpha2_to_continent_code(alpha2)
            continent_name = pc_convert.convert_continent_code_to_continent_name(
                continent_code
            )
            return continent_name
        except Exception:
            return "Unknown"

    df["country_name"] = df["country"].apply(iso_to_country_name)
    df["continent"] = df["country_name"].apply(country_to_continent)

    return df


def clean_dass_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute DASS subscale scores (Depression, Anxiety, Stress) and total score.

    Uses the original item groupings for:
    - dassdepression
    - dassanxiety
    - dassstress
    and their sum 'dasssum'.
    """
    df = df.copy()

    depression_items = [
        "Q3A", "Q5A", "Q10A", "Q13A", "Q16A", "Q17A",
        "Q21A", "Q24A", "Q26A", "Q31A", "Q34A", "Q37A",
        "Q38A", "Q42A",
    ]

    anxiety_items = [
        "Q2A", "Q4A", "Q7A", "Q9A", "Q15A", "Q19A",
        "Q20A", "Q23A", "Q25A", "Q28A", "Q30A", "Q36A",
        "Q40A", "Q41A",
    ]

    stress_items = [
        "Q1A", "Q6A", "Q8A", "Q11A", "Q12A", "Q14A",
        "Q18A", "Q22A", "Q27A", "Q29A", "Q32A", "Q33A",
        "Q35A", "Q39A",
    ]

    df["dassdepression"] = df[depression_items].sum(axis=1)
    df["dassanxiety"] = df[anxiety_items].sum(axis=1)
    df["dassstress"] = df[stress_items].sum(axis=1)

    df["dasssum"] = (
        df["dassdepression"]
        + df["dassanxiety"]
        + df["dassstress"]
    )

    return df


def clean_tipi_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the five TIPI-based personality traits and remove rows
    with invalid trait scores.

    Steps:
    - Reverse-score items TIPI2, TIPI4, TIPI6, TIPI8, TIPI10 using (8 - value).
    - Compute the Big Five traits by averaging the appropriate item pairs.
    - Keep only rows where all five trait scores lie between 1 and 7.
    """
    df = df.copy()

    reverse_items = {
        "TIPI2": "TIPI_R2",
        "TIPI4": "TIPI_R4",
        "TIPI6": "TIPI_R6",
        "TIPI8": "TIPI_R8",
        "TIPI10": "TIPI_R10",
    }

    for original, reversed_name in reverse_items.items():
        if original in df.columns:
            df[reversed_name] = 8 - df[original]

    if {"TIPI1", "TIPI_R6"}.issubset(df.columns):
        df["Extraversion"] = (df["TIPI1"] + df["TIPI_R6"]) / 2

    if {"TIPI7", "TIPI_R2"}.issubset(df.columns):
        df["Agreeableness"] = (df["TIPI7"] + df["TIPI_R2"]) / 2

    if {"TIPI3", "TIPI_R8"}.issubset(df.columns):
        df["Conscientiousness"] = (df["TIPI3"] + df["TIPI_R8"]) / 2

    if {"TIPI_R4", "TIPI9"}.issubset(df.columns):
        df["Emotional Stability"] = (df["TIPI_R4"] + df["TIPI9"]) / 2

    if {"TIPI5", "TIPI_R10"}.issubset(df.columns):
        df["Openness"] = (df["TIPI5"] + df["TIPI_R10"]) / 2

    tipi_traits = [
        "Extraversion",
        "Agreeableness",
        "Conscientiousness",
        "Emotional Stability",
        "Openness",
    ]

    for trait in tipi_traits:
        if trait in df.columns:
            df = df[(df[trait] >= 1) & (df[trait] <= 7)]

    return df


def clean_vcl_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VCL validity, invalid responses, and VCL score.

    - VCL items: VCL1–VCL16.
    - Fake words: VCL6, VCL9, VCL12.
    - vcl_score: number of real words checked.
    - vcl_invalid: number of fake words checked.
    - vcl_valid: True when no fake words are checked.
    """
    df = df.copy()

    vcl_items = [f"VCL{i}" for i in range(1, 17)]
    fake_words = ["VCL6", "VCL9", "VCL12"]
    real_words = [item for item in vcl_items if item not in fake_words]

    df["vcl_score"] = df[real_words].sum(axis=1)
    df["vcl_invalid"] = df[fake_words].sum(axis=1)
    df["vcl_valid"] = df["vcl_invalid"] == 0

    return df


def handle_missing_values(
    df: pd.DataFrame,
    column_missing_threshold: float = 0.20,
) -> pd.DataFrame:
    """
    Drop columns with more than a given proportion of missing values,
    then drop all remaining rows with missing values.

    A column is kept only if at least (1 - threshold) of its values are non-missing.
    """
    df = df.copy()

    if not 0.0 <= column_missing_threshold < 1.0:
        raise ValueError("column_missing_threshold must be in [0, 1).")

    min_non_missing = int((1.0 - column_missing_threshold) * len(df))

    df = df.dropna(axis=1, thresh=min_non_missing)
    df = df.dropna(axis=0)

    return df


DEMOGRAPHIC_COLUMNS: List[str] = [
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
    "age_clean",
    "country_name",
    "continent",
    "familysize",
]

TARGET_COLUMNS: List[str] = [
    "dassanxiety",
    "dassstress",
    "dassdepression",
    "dasssum",
]

TIPI_TRAIT_COLUMNS: List[str] = [
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Emotional Stability",
    "Openness",
]

VCL_NEW_COLUMNS: List[str] = [
    "vcl_valid",
    "vcl_invalid",
    "vcl_score",
]


def create_clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the fully cleaned DataFrame from the raw input DataFrame.

    Includes:
    - recoding categorical variables
    - cleaning the age variable
    - cleaning country and continent
    - computing DASS, TIPI and VCL variables
    - handling missing values
    - selecting key columns
    - renaming DASS, TIPI and VCL item columns to descriptive labels
    """
    df = raw_df.copy()

    df = recode_categorical_variables(df)
    df = clean_age_column(df)
    df = clean_country_and_continent(df)
    df = clean_dass_variables(df)
    df = clean_tipi_variables(df)
    df = clean_vcl_variables(df)

    df = handle_missing_values(df, column_missing_threshold=0.20)

    cols_to_keep: List[str] = (
        DASS_ITEMS
        + TIPI_ITEMS
        + VCL_ITEMS
        + DEMOGRAPHIC_COLUMNS
        + TARGET_COLUMNS
        + TIPI_TRAIT_COLUMNS
        + VCL_NEW_COLUMNS
    )

    existing_columns = [col for col in cols_to_keep if col in df.columns]
    df = df[existing_columns].copy()

    rename_map: Dict[str, str] = {}

    for item in DASS_ITEMS:
        if item in df.columns and item in DASS_LABELS:
            rename_map[item] = DASS_LABELS[item]

    for item in TIPI_ITEMS:
        if item in df.columns and item in TIPI_LABELS:
            rename_map[item] = TIPI_LABELS[item]

    for item in VCL_ITEMS:
        if item in df.columns and item in VCL_LABELS:
            rename_map[item] = VCL_LABELS[item]

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def create_clean_valid_df(clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict the cleaned DataFrame to rows with valid VCL scores
    and convert string columns to categorical dtype.
    """
    df = clean_df.copy()

    if "vcl_valid" in df.columns:
        df = df[df["vcl_valid"]]

    category_columns = [col for col in df.columns if df[col].dtype == "object"]
    for col in category_columns:
        df[col] = df[col].astype("category")

    return df


def create_analysis_df(clean_valid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the analysis DataFrame containing only the variables
    required for statistical analyses and visualizations.
    """
    df = clean_valid_df.copy()

    columns_to_keep: List[str] = [
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
        "age_clean",
        "country_name",
        "continent",
        "familysize",
        "dassanxiety",
        "dassstress",
        "dassdepression",
        "dasssum",
        "Extraversion",
        "Agreeableness",
        "Conscientiousness",
        "Emotional Stability",
        "Openness",
        "vcl_valid",
        "vcl_invalid",
        "vcl_score",
    ]

    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[existing_columns]

    return df


def create_standardized_df(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a standardized version of the analysis DataFrame.

    Numerical variables are standardized and stored in new
    columns with the suffix "_std".
    """
    df = analysis_df.copy()

    columns_to_standardize: List[str] = [
        "age_clean",
        "familysize",
        "dassanxiety",
        "dassstress",
        "dassdepression",
        "dasssum",
        "Extraversion",
        "Agreeableness",
        "Conscientiousness",
        "Emotional Stability",
        "Openness",
        "vcl_score",
    ]

    existing_columns = [col for col in columns_to_standardize if col in df.columns]
    if not existing_columns:
        return df

    scaler = StandardScaler()
    standardized_values = scaler.fit_transform(df[existing_columns])

    standardized_columns = [f"{col}_std" for col in existing_columns]
    df[standardized_columns] = standardized_values

    return df


def full_cleaning_pipeline(raw_df: pd.DataFrame) -> CleanedData:
    """
    Run the full cleaning pipeline and return all relevant DataFrames.
    """
    clean_df = create_clean_df(raw_df)
    clean_valid_df = create_clean_valid_df(clean_df)
    analysis_df = create_analysis_df(clean_valid_df)
    analysis_standardized_df = create_standardized_df(analysis_df)

    return CleanedData(
        raw_df=raw_df,
        clean_df=clean_df,
        clean_valid_df=clean_valid_df,
        analysis_df=analysis_df,
        analysis_standardized_df=analysis_standardized_df,
    )