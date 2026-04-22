"""
Download winutils.exe and hadoop.dll for PySpark on Windows.

PySpark 3.5.x ships with Hadoop 3.x libraries internally, but on Windows it
also needs the native winutils.exe and hadoop.dll binaries to create temp
directories and handle file permissions.  This script downloads the correct
versions into <project_root>/winutils/bin/ so every teammate can run the
project without a manual Hadoop install.

Usage (run once, from any directory):
    python scripts/get_winutils.py

The download is skipped if the files are already present.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Hadoop 3.4.0 binaries — Spark 4.0.x is built against Hadoop 3.4+.
# Source: kontext-tech/winutils (hadoop-3.4.0-win10-x64 build).
# ---------------------------------------------------------------------------
WINUTILS_URL = (
    "https://github.com/kontext-tech/winutils/raw/master/hadoop-3.4.0-win10-x64/bin/winutils.exe"
)
HADOOP_DLL_URL = (
    "https://github.com/kontext-tech/winutils/raw/master/hadoop-3.4.0-win10-x64/bin/hadoop.dll"
)

# Resolve paths relative to this script so it works from any CWD
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
WINUTILS_BIN = PROJECT_ROOT / "winutils" / "bin"


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest*, showing a simple progress indicator."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    filename = dest.name
    print(f"  Downloading {filename} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size // 1024
        print(f"done ({size_kb} KB)")
    except Exception as exc:
        print(f"FAILED\n    {exc}")
        raise


def main() -> None:
    if sys.platform != "win32":
        print("winutils is only needed on Windows — nothing to do.")
        return

    print(f"Target directory: {WINUTILS_BIN}\n")

    for url, name in [
        (WINUTILS_URL, "winutils.exe"),
        (HADOOP_DLL_URL, "hadoop.dll"),
    ]:
        dest = WINUTILS_BIN / name
        if dest.exists():
            print(f"  {name} already present — skipping.")
        else:
            _download(url, dest)

    print(
        "\nDone!  HADOOP_HOME will be auto-detected from the winutils/ folder\n"
        "next time you run the pipeline — no manual environment variable needed.\n"
    )


if __name__ == "__main__":
    main()
