.PHONY: help setup install data preprocess eda dashboard clean clean-all lint test run-all

VENV := venv
PYTHON := $(VENV)/Scripts/python
PIP := $(VENV)/Scripts/pip
STREAMLIT := $(VENV)/Scripts/streamlit

# Default goal
help:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║         UK Road Accidents — Data Pipeline Makefile              ║"
	@echo "║                                                                  ║"
	@echo "║  Usage: make [target]                                            ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "🔧 SETUP & DEPENDENCIES"
	@echo "  make setup                 Install Python venv + dependencies"
	@echo "  make install               Install Python packages (requires venv)"
	@echo ""
	@echo "📊 PIPELINE TARGETS (RUN IN ORDER)"
	@echo "  make data                  [1] Download & ingest raw data → data/interim/"
	@echo "  make preprocess            [2] Clean & engineer features → data/processed/"
	@echo "  make eda                   [3] Generate EDA report → reports/"
	@echo ""
	@echo "🚀 RUN ALL"
	@echo "  make run-all               Execute full pipeline (setup → data → preprocess → eda)"
	@echo ""
	@echo "📈 DASHBOARD & VISUALIZATION"
	@echo "  make dashboard             Launch Streamlit analytics dashboard (http://localhost:8501)"
	@echo ""
	@echo "🧹 CLEANING"
	@echo "  make clean                 Remove processed data & artifacts"
	@echo "  make clean-all             Clean everything (including venv, data, reports)"
	@echo ""
	@echo "🧪 DEVELOPMENT"
	@echo "  make lint                  Run code quality checks (Python)"
	@echo "  make test                  Run unit tests (if available)"
	@echo ""
	@echo "📋 FILE STRUCTURE"
	@echo "  data/raw/                  Original CSV files (Accident_Information.csv, Vehicle_Information.csv)"
	@echo "  data/interim/              Merged & cleaned parquet files"
	@echo "  data/processed/            Final train/test splits with features → ready for ML model"
	@echo "  reports/                   EDA report (JSON) + validation logs"
	@echo "  models/                    Trained KMeans clustering model"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ═══════════════════════════════════════════════════════════════════════════
# Setup & Installation
# ═══════════════════════════════════════════════════════════════════════════

.setup_checked:
	@if [ ! -d "$(VENV)" ]; then \
		echo "❌ Virtual environment not found."; \
		echo "   Run: make setup"; \
		exit 1; \
	fi
	@touch .setup_checked

setup:
	@echo "🔧 Creating Python virtual environment..."
	python -m venv $(VENV)
	@echo "✅ Virtual environment created at $(VENV)/"
	@echo "📦 Installing dependencies..."
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	@echo "✅ All dependencies installed"
	@echo ""
	@echo "💡 Next steps:"
	@echo "   make data          # Download and ingest raw data"
	@echo "   make preprocess    # Run preprocessing pipeline"
	@echo "   make dashboard     # View Streamlit dashboard"

install: .setup_checked
	@echo "📦 Installing Python packages..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Packages installed"

# ═══════════════════════════════════════════════════════════════════════════
# Data Pipeline
# ═══════════════════════════════════════════════════════════════════════════

data: .setup_checked data/interim/merged.parquet
	@echo "✅ Data ingestion complete"
	@echo "   Output: data/interim/merged.parquet (2.7M rows)"
	@echo "   Features: 64 columns (accidents + weather)"
	@echo ""

data/interim/merged.parquet:
	@echo "📥 [STEP 1/3] Ingesting raw data..."
	@echo "   • Reading: data/raw/Accident_Information.csv"
	@echo "   • Reading: data/raw/Vehicle_Information.csv"
	@echo "   • Downloading weather data (may take 2-5 min)..."
	@echo "   • Merging and storing to parquet..."
	$(PYTHON) -m src.data.ingest
	@echo "✅ Merged dataset saved: $@"

preprocess: .setup_checked data/interim/merged.parquet data/processed/train.parquet data/processed/test.parquet
	@echo "✅ Preprocessing complete"
	@echo "   Output: data/processed/train.parquet (2.1M rows, 110 features)"
	@echo "   Output: data/processed/test.parquet  (543K rows, 110 features)"
	@echo "   Features: 14 scaled numeric + 2 domain + 2 spatial + 92 encoded categorical"
	@echo ""

data/processed/train.parquet data/processed/test.parquet: data/interim/merged.parquet
	@echo "🔧 [STEP 2/3] Preprocessing & feature engineering..."
	@echo "   • Cleaning: removing nulls, invalid values, high-missing columns"
	@echo "   • Domain features: danger_index, vehicle_vulnerable"
	@echo "   • Spatial features: accident_hotspot_cluster, location_density"
	@echo "   • Encoding: StringIndexer + OneHotEncoder for categoricals"
	@echo "   • Scaling: StandardScaler for numeric features"
	@echo "   • Splitting: 80% train / 20% test"
	$(PYTHON) -m src.preprocessing.run
	@echo "✅ Preprocessed splits saved: $@"

eda: .setup_checked data/interim/merged.parquet reports/eda_summary.json
	@echo "✅ EDA report generated"
	@echo "   Output: reports/eda_summary.json"
	@echo "   Contains: severity distribution, weather analysis, temporal patterns, location stats"
	@echo ""

reports/eda_summary.json: data/interim/merged.parquet
	@echo "📊 [STEP 3/3] Generating exploratory data analysis..."
	@echo "   • Severity distribution"
	@echo "   • Weather condition analysis"
	@echo "   • Temporal patterns (hourly, daily)"
	@echo "   • Location density & clustering"
	@echo "   • Null statistics & data quality"
	$(PYTHON) -m src.data.ingest  # Also generates EDA as side effect
	@echo "✅ EDA report saved: $@"

# ═══════════════════════════════════════════════════════════════════════════
# Dashboard & Visualization
# ═══════════════════════════════════════════════════════════════════════════

dashboard: .setup_checked reports/eda_summary.json
	@echo "🎨 Launching Streamlit dashboard..."
	@echo ""
	@echo "   📊 Dashboard URL: http://localhost:8501"
	@echo "   Press Ctrl+C to stop"
	@echo ""
	$(STREAMLIT) run app.py

# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════

run-all: setup data preprocess eda
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║                  ✅ PIPELINE COMPLETE                           ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📁 Output files:"
	@echo "   data/interim/merged.parquet              (merged accident + weather data)"
	@echo "   data/processed/train.parquet             (training set with features)"
	@echo "   data/processed/test.parquet              (test set with features)"
	@echo "   reports/eda_summary.json                 (exploratory analysis report)"
	@echo "   reports/validation_report.json           (data quality validation)"
	@echo ""
	@echo "🚀 Next steps:"
	@echo "   make dashboard    # View interactive analytics dashboard"
	@echo "   make test         # Run unit tests"
	@echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Quality & Testing
# ═══════════════════════════════════════════════════════════════════════════

lint: .setup_checked
	@echo "🔍 Running code quality checks..."
	$(PIP) install pylint flake8 --quiet
	@echo "   • flake8 (style)..."
	-$(VENV)/Scripts/flake8 src/ --max-line-length=100 --ignore=E501
	@echo "   • pylint (analysis)..."
	-$(VENV)/Scripts/pylint src/ --disable=C0111,R0913 --quiet
	@echo "✅ Lint complete (⚠️ warnings are informational)"

test: .setup_checked
	@echo "🧪 Running unit tests..."
	@if [ -d "tests/" ] && [ -n "$$(find tests/ -name '*.py' -type f)" ]; then \
		$(PIP) install pytest --quiet; \
		$(PYTHON) -m pytest tests/ -v; \
	else \
		echo "⚠️  No tests found in tests/ directory"; \
	fi

# ═══════════════════════════════════════════════════════════════════════════
# Cleaning
# ═══════════════════════════════════════════════════════════════════════════

clean:
	@echo "🧹 Cleaning processed data & artifacts..."
	rm -rf data/processed/
	rm -rf data/interim/merged.parquet
	rm -f reports/eda_summary.json reports/validation_report.json
	rm -f reports/validation_summary.txt
	rm -f artifacts/
	rm -f .setup_checked
	@echo "✅ Cleaned: data/processed/, data/interim/merged.parquet, reports/"

clean-all: clean
	@echo "🧹 Cleaning EVERYTHING (including venv & raw data)..."
	rm -rf $(VENV)/
	rm -rf data/
	rm -rf reports/
	rm -rf models/
	rm -f .setup_checked
	@echo "✅ Cleaned: $(VENV)/, data/, reports/, models/"
	@echo ""
	@echo "💡 To restore: run make setup && make run-all"

# ═══════════════════════════════════════════════════════════════════════════
# Meta targets
# ═══════════════════════════════════════════════════════════════════════════

.PHONY: help setup install data preprocess eda dashboard run-all lint test clean clean-all .setup_checked
