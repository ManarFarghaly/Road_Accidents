"""
Dataset acquisition — downloads the UK Road Safety CSVs from Kaggle.

Usage:
    from src.data.acquire import download_dataset
    raw_path = download_dataset()    # returns pathlib.Path to the folder
                                     # containing Accident_Information.csv
                                     # and Vehicle_Information.csv
"""
from pathlib import Path
import shutil

import kagglehub

from src.config import RAW_DIR

KAGGLE_DATASET = "tsiaras/uk-road-safety-accidents-and-vehicles"
REQUIRED_FILES = ["Accident_Information.csv", "Vehicle_Information.csv"]


def download_dataset() -> Path:
    """
    Download the Kaggle CSVs if they are not already in data/raw/.

    Returns:
        Path to data/raw/ (the folder containing the CSVs).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Fast path — files already copied into data/raw/
    missing = [f for f in REQUIRED_FILES if not (RAW_DIR / f).exists()]
    if not missing:
        print(f"[acquire] CSVs already present in {RAW_DIR}")
        return RAW_DIR

    print(f"[acquire] downloading {KAGGLE_DATASET} from Kaggle ...")
    # kagglehub renamed its public API between 0.1 and 0.2. Try the current
    # one first, fall back to older names so old pinned envs still work.
    # If nothing works, tell the user clearly to upgrade.
    for fn_name in ("dataset_download", "datasets_download", "download"):
        fn = getattr(kagglehub, fn_name, None)
        if callable(fn):
            kaggle_cache = Path(fn(KAGGLE_DATASET))
            break
    else:
        raise RuntimeError(
            "Your installed kagglehub is too old — none of "
            "(dataset_download, datasets_download, download) exist. "
            "Fix: pip install -U kagglehub"
        )
    print(f"[acquire] Kaggle cache: {kaggle_cache}")

    for fname in REQUIRED_FILES:
        src = kaggle_cache / fname
        dst = RAW_DIR / fname
        if not src.exists():
            raise FileNotFoundError(
                f"{fname} not found in Kaggle cache at {kaggle_cache}. "
                f"Check that the dataset slug '{KAGGLE_DATASET}' still exposes this file."
            )
        if not dst.exists():
            print(f"[acquire] copying {fname} -> {dst}")
            shutil.copy2(src, dst)

    return RAW_DIR


if __name__ == "__main__":
    download_dataset()
