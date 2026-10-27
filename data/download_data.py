"""Download the Credit Risk Dataset into ``data/``.

Usage:
    python data/download_data.py

The dataset is licensed CC0 (public domain). A copy is committed to this
repository, so this script is only needed if the CSV is missing or you want to
refresh it from the source.
"""

from __future__ import annotations

import hashlib
import io
import sys
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/laotse/credit-risk-dataset"
EXPECTED_SHA256 = "ce3c6d2167717bf1627d1c0c81cbccd28323cd4aa7b96d542599366d5ff6aac8"
CSV_NAME = "credit_risk_dataset.csv"
DATA_DIR = Path(__file__).resolve().parent


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    target = DATA_DIR / CSV_NAME
    if target.exists():
        digest = sha256_of(target.read_bytes())
        if digest == EXPECTED_SHA256:
            print(f"{target} already exists and checksum matches.")
            return 0
        print(f"WARNING: {target} exists but checksum differs. Re-downloading...")

    print(f"Downloading {DATASET_URL}")
    try:
        with urlopen(DATASET_URL, timeout=60) as response:  # noqa: S310 - fixed https URL
            payload = response.read()
    except (URLError, TimeoutError) as exc:
        print(f"ERROR: download failed ({exc}).")
        print(
            "Download the dataset manually from "
            "https://www.kaggle.com/datasets/laotse/credit-risk-dataset, "
            f"unzip it, and place {CSV_NAME} in {DATA_DIR}."
        )
        return 1

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            csv_bytes = archive.read(CSV_NAME)
    except (zipfile.BadZipFile, KeyError) as exc:
        print(f"ERROR: unexpected archive contents ({exc}).")
        return 1

    digest = sha256_of(csv_bytes)
    if digest != EXPECTED_SHA256:
        print("ERROR: checksum mismatch; the source file may have changed.")
        print(f"  expected: {EXPECTED_SHA256}")
        print(f"  actual:   {digest}")
        return 1

    target.write_bytes(csv_bytes)
    print(f"Saved {target} ({len(csv_bytes):,} bytes), checksum verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
