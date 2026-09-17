"""Download the raw dataset into data/raw.

Kept separate from the Airflow DAG so the dataset can also be fetched by hand:
    python code/datasets/download_data.py
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "penguins.csv"
SOURCE_URL = (
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
)


def download(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists() and not force:
        print(f"raw data already present: {RAW_FILE}")
        return RAW_FILE
    print(f"downloading {SOURCE_URL} -> {RAW_FILE}")
    urllib.request.urlretrieve(SOURCE_URL, RAW_FILE)
    return RAW_FILE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = parser.parse_args()
    path = download(force=args.force)
    print(f"raw rows: {sum(1 for _ in path.open(encoding='utf-8')) - 1}")
