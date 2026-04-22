#!/usr/bin/env bash
# =============================================================================
#  Road Accidents Project — one-command Linux / macOS setup
#
#  Run this once after cloning the repo:
#      bash setup.sh
#
#  What it does:
#    1. Checks for Python 3.11 / 3.12 and Java 11+
#    2. Creates a virtual environment in road_env/
#    3. Installs all Python dependencies from requirements.txt
#    4. Prints next-step instructions
#
#  Note: winutils is Windows-only. On Linux/macOS PySpark works without it.
# =============================================================================

set -euo pipefail

echo
echo "============================================================"
echo "  Road Accidents Project — Linux/macOS Setup"
echo "============================================================"
echo

# ---------------------------------------------------------------------------
# 1. Check Python
# ---------------------------------------------------------------------------
PYTHON=""
for cmd in python3.14 python3.13 python3.12 python3.11 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=${ver%%.*}; minor=${ver##*.}
        if [[ "$major" -eq 3 && "$minor" -ge 11 ]]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "[ERROR] Python 3.11 or later not found."
    echo "        Install from https://python.org or via your package manager."
    exit 1
fi
echo "[OK] Python $($PYTHON --version)"

# ---------------------------------------------------------------------------
# 2. Check Java
# ---------------------------------------------------------------------------
if ! command -v java &>/dev/null; then
    echo "[WARNING] Java not found. PySpark requires Java 11 or 17."
    echo "          Install with:  sudo apt install openjdk-17-jdk   (Debian/Ubuntu)"
    echo "                        brew install openjdk@17            (macOS)"
else
    echo "[OK] $( java -version 2>&1 | head -1 )"
fi

# ---------------------------------------------------------------------------
# 3. Create virtual environment
# ---------------------------------------------------------------------------
if [[ -f road_env/bin/activate ]]; then
    echo "[OK] Virtual environment already exists — skipping creation."
else
    echo "[..] Creating virtual environment in road_env/ ..."
    "$PYTHON" -m venv road_env
    echo "[OK] Virtual environment created."
fi

# ---------------------------------------------------------------------------
# 4. Install dependencies
# ---------------------------------------------------------------------------
echo "[..] Installing dependencies (this may take a few minutes) ..."
# shellcheck disable=SC1091
source road_env/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "[OK] Dependencies installed."

# ---------------------------------------------------------------------------
# 5. Done
# ---------------------------------------------------------------------------
echo
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo
echo "  Activate the virtual environment:"
echo "      source road_env/bin/activate"
echo
echo "  Then run the pipeline stages:"
echo "      python -m src.data.ingest"
echo "      python -m src.data.validate"
echo "      python -m src.preprocessing.run"
echo
echo "  Or run tests:"
echo "      python -m tests.test_preprocessing"
echo
