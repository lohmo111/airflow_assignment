"""Stage 1 - data engineering: load, clean and split the raw dataset.

Input : data/raw/penguins.csv
Output: data/processed/train.csv, data/processed/test.csv
"""
from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "penguins.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SOURCE_URL = (
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
)

TARGET = "species"
NUMERIC_FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
CATEGORICAL_FEATURES = ["island", "sex"]
TEST_SIZE = 0.2
RANDOM_STATE = 42
# Rows outside median +/- IQR_FACTOR * IQR on any numeric column are dropped.
IQR_FACTOR = 3.0


def load_data() -> pd.DataFrame:
    """Read the raw file, downloading it first if it is not there yet."""
    if not RAW_FILE.exists():
        RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
        log.info("raw file missing, downloading from %s", SOURCE_URL)
        urllib.request.urlretrieve(SOURCE_URL, RAW_FILE)
    df = pd.read_csv(RAW_FILE)
    log.info("loaded %d rows x %d columns from %s", len(df), df.shape[1], RAW_FILE)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable rows, impute missing values and remove outliers."""
    before = len(df)
    df = df[[TARGET, *CATEGORICAL_FEATURES, *NUMERIC_FEATURES]].copy()

    # A row without a label cannot be used for supervised training.
    df = df.dropna(subset=[TARGET])
    df = df.drop_duplicates()

    # The dataset encodes unknown sex both as NaN and as ".".
    df["sex"] = df["sex"].replace(".", pd.NA).str.upper()

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median = df[col].median()
        missing = int(df[col].isna().sum())
        if missing:
            log.info("imputing %d missing values in %s with median %.2f", missing, col, median)
        df[col] = df[col].fillna(median)

    for col in CATEGORICAL_FEATURES:
        mode = df[col].mode(dropna=True)
        fill = mode.iloc[0] if not mode.empty else "UNKNOWN"
        missing = int(df[col].isna().sum())
        if missing:
            log.info("imputing %d missing values in %s with mode %r", missing, col, fill)
        df[col] = df[col].fillna(fill)

    # Outlier removal: IQR rule applied to every numeric feature.
    mask = pd.Series(True, index=df.index)
    for col in NUMERIC_FEATURES:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - IQR_FACTOR * iqr, q3 + IQR_FACTOR * iqr
        mask &= df[col].between(lower, upper)
    removed = int((~mask).sum())
    if removed:
        log.info("removing %d outlier rows", removed)
    df = df[mask].reset_index(drop=True)

    log.info("cleaning: %d rows in, %d rows out", before, len(df))
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def run_data_engineering() -> dict:
    """Entry point used by the Airflow DAG."""
    df = load_data()
    df = clean_data(df)
    train_df, test_df = split_data(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary = {
        "raw_file": str(RAW_FILE),
        "train_file": str(train_path),
        "test_file": str(test_path),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "classes": sorted(df[TARGET].unique().tolist()),
    }
    log.info("stage 1 done: %s", json.dumps(summary))
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(json.dumps(run_data_engineering(), indent=2))
