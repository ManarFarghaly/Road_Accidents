#!/usr/bin/env bash
# =============================================================================
#  Road Accidents Project — one-command Linux / macOS setup
#
#  Run this once after cloning the repo:
#      bash setup.sh
#
#  What it does:
#    1. Checks for Python 3.11+ and Java 11+
#    2. Installs all Python dependencies into the system Python
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
# 1. Check Python (3.11+)
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
echo "[OK] $($PYTHON --version)"

# ---------------------------------------------------------------------------
# 2. Check Java
# ---------------------------------------------------------------------------
if ! command -v java &>/dev/null; then
    echo "[WARNING] Java not found. PySpark requires Java 11 or 17."
    echo "          Install: sudo apt install openjdk-17-jdk  (Debian/Ubuntu)"
    echo "                   brew install openjdk@17          (macOS)"
else
    echo "[OK] $(java -version 2>&1 | head -1)"
fi

# ---------------------------------------------------------------------------
# 3. Install dependencies
# ---------------------------------------------------------------------------
echo "[..] Installing dependencies (this may take a few minutes) ..."
"$PYTHON" -m pip install --upgrade pip --quiet
"$PYTHON" -m pip install -r requirements.txt
echo "[OK] Dependencies installed."

# ---------------------------------------------------------------------------
# 4. Done
# ---------------------------------------------------------------------------
echo
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo
echo "  Run the pipeline stages:"
echo "      python -m src.preprocessing.run"
echo

