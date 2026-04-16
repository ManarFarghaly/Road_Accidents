"""
Data Validation Script — UK Road Accidents Project
CMPS344 Applied Data Science — Phase 2

Covers all 8 validation dimensions from the course lecture:
    1. Accuracy       — business rule checks
    2. Consistency    — format and logical cross-field checks
    3. Completeness   — missing value analysis
    4. Uniqueness     — duplicate detection (exact + key-level)
    5. Outliers       — IQR, Z-score, Isolation Forest
    6. Timeliness     — date range, year coverage, COVID dip flag
    7. Distribution   — min/max/mean/std/skewness/kurtosis/KS test
    8. Relationships  — Pearson vs Spearman, high-correlation flags

Outputs (all saved to reports/):
    validation_report.json     — full machine-readable findings
    validation_summary.txt     — plain-text summary for report screenshot

Usage:
    poetry run python src/data/validate.py
    make validate
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kurtosis, skew
from sklearn.ensemble import IsolationForest
import os
from src.data.acquire import download_dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Image, SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors


# Get path from acquire module
path = download_dataset()

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────
RAW_DIR     = Path(path)
REPORTS_DIR  = Path("reports")
ACCIDENTS_FILE = RAW_DIR / "Accident_Information.csv"
VEHICLES_FILE  = RAW_DIR / "Vehicle_Information.csv" 
# ── Report accumulator ──────────────────────────────────────────────────────
report = {
    "generated_at": datetime.now().isoformat(),
    "dimensions": {},
    "summary": {"issues": []}
}
log_lines = []

HIGH_CORR_THRESHOLD = 0.85
NON_LINEAR_DIFF_THRESHOLD = 0.10
KS_SAMPLE_SIZE = 5000
# We chose 20% as threshold based on common data quality practices
MISSING_VALUE_THRESHOLD = 20.0 
VALID_DAYS = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"}

# ── Helper functions ────────────────────────────────────────────────────────
def log(msg: str = "",log_lines: list= None):
    """
    Logs a message to console and optionally appends it to a log list.
    
    Args:
        msg (str): The message to log. Defaults to empty string.
        log_lines (list): A list to accumulate log messages for 
            later output to files. If None, only prints to console.
    """
    print(msg)
    if log_lines is not None:
        log_lines.append(msg)

def issue(msg: str, dimension: str,report: dict, log_lines: list):
    """
    Records a validation issue to both the log and the report summary.
    
    Args:
        msg (str): Description of the validation issue found.
        dimension (str): The validation dimension category (e.g., "accuracy",
            "consistency", "completeness", "uniqueness", "outliers", 
            "timeliness", "distribution", "relationships").
        report (dict): The main report dictionary where issues are accumulated.
        log_lines (list): The list for logging messages.
    """
    log(f"  [ISSUE] {msg}", log_lines)
    report["summary"]["issues"].append({"dimension": dimension, "message": msg})

def to_numeric_safe(series):
    """
    Safely converts a pandas Series to numeric type, coercing errors to NaN.
    
    Args:
        series (pd.Series): The pandas Series to convert.
    """
    return pd.to_numeric(series, errors="coerce")

def column_exists(df, col, log_lines: list):
    """
    Checks if a column exists in a DataFrame, logging a skip message if not.
    
    Args:
        df (pd.DataFrame): The DataFrame to check.
        col (str): The column name to look for.
        log_lines (list): The list for logging messages.
    
    Returns:
        bool: True if the column exists, False otherwise.
    """
    if col not in df.columns:
        log(f"  [SKIP] Column '{col}' not found", log_lines)
        return False
    return True

def generate_histogram_image(key, hist_data, buf_store: list):
    """
    Generate a histogram as a ReportLab Image from stored bin data.
    Args:
    key (str): The column name for the histogram title.
    hist_data (dict): Contains 'bin_edges' and 'bin_counts' for the histogram.
    buf_store (list): A list to store the in-memory image buffer to keep it alive until report generation.
    Returns:
    Image: A ReportLab Image object containing the histogram.
    
    """
    bin_edges = hist_data["bin_edges"]
    bin_counts = hist_data["bin_counts"]
    
    fig, ax = plt.subplots(figsize=(6, 3))
    widths = [bin_edges[i+1] - bin_edges[i] for i in range(len(bin_counts))]
    ax.bar(bin_edges[:-1], bin_counts, width=widths, align='edge', color='steelblue', edgecolor='white')
    ax.set_title(f"Histogram of {key}", fontsize=11)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x:,.0f}'))
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    plt.close(fig)
    buf.seek(0)
    buf_store.append(buf)  # ← keep alive until doc.build() finishes
    return Image(buf, width=5*inch, height=2.5*inch)

# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_tables(log_lines: list):
    """
    Loads the accident and vehicle CSV files into pandas DataFrames.

    Args:
        log_lines (list): The list for logging messages.
    
    Returns:
        tuple: A tuple containing (accidents_df, vehicles_df) as pandas DataFrames.
    """
    log("=" * 65, log_lines)
    log("  UK Road Accidents — Data Validation Report", log_lines)
    log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_lines)
    log("=" * 65, log_lines)

    for path in [ACCIDENTS_FILE, VEHICLES_FILE]:
        if not path.exists():
            log(f"\n[ERROR] {path} not found.", log_lines)
            log("        Run: make data   or   poetry run python src/data/acquire.py", log_lines)
            raise SystemExit(1)

    acc = pd.read_csv(ACCIDENTS_FILE,encoding='latin1', low_memory=False)
    veh = pd.read_csv(VEHICLES_FILE,encoding='latin1',  low_memory=False)

    log(f"\nLoaded accidents : {acc.shape[0]:>9,} rows x {acc.shape[1]} columns", log_lines)
    log(f"Loaded vehicles  : {veh.shape[0]:>9,} rows x {veh.shape[1]} columns", log_lines)
    return acc, veh
# ════════════════════════════════════════════════════════════════════════════
#  DIMENSION 1 — ACCURACY
#  Does data correctly represent reality?
#  Apply business rules specific to UK road accident data.
# ════════════════════════════════════════════════════════════════════════════

def check_accuracy(acc: pd.DataFrame, veh: pd.DataFrame, log_lines: list, report: dict):
    """
    Performs accuracy validation checks based on UK road accident business rules.
    
    Validates that data correctly represents reality by checking:
    - Accident_Severity is one of 3 valid classes (Slight/Serious/Fatal)
    - Speed_limit is a valid UK legal limit (20, 30, 40, 50, 60, 70 mph)
    - Coordinates fall within UK geographic bounds (49-61°N, -8 to 2°E)
    - Number_of_Vehicles is positive (≥1)
    - Age_of_Vehicle is non-negative
    - Age_Band_of_Driver contains valid age bands
    - Day_of_Week contains valid day names
    - Sex_of_Driver is Male or Female
    
    Args:
        acc (pd.DataFrame): The accidents DataFrame.
        veh (pd.DataFrame): The vehicles DataFrame.
        log_lines (list): The list for logging messages.
        report (dict): The main report dictionary to update with findings.
    """
    log("\n" + "-" * 65, log_lines)
    log("DIMENSION 1 — ACCURACY", log_lines)
    log("-" * 65, log_lines)
    dim = {}

    # 1a. Target variable must be one of 3 valid classes
    valid_severity = {"Slight", "Serious", "Fatal", "1", "2", "3"}
    if column_exists(acc, "Accident_Severity", log_lines):
        bad = ~acc["Accident_Severity"].astype(str).isin(valid_severity)
        n = int(bad.sum())
        dim["invalid_severity_values"] = n
        
        if n:
            issue(f"{n:,} rows have unrecognised Accident_Severity values", "accuracy", report, log_lines)
        else:
            log("  [PASS] Accident_Severity — all values valid (Slight/Serious/Fatal)", log_lines)

    # 1b. Speed_limit must be a UK legal limit: 20,30,40,50,60,70
    uk_speed_limits = {20, 30, 40, 50, 60, 70}
    if column_exists(acc, "Speed_limit", log_lines):
        speed = to_numeric_safe(acc["Speed_limit"]).dropna()
        bad_speed = speed[~speed.isin(uk_speed_limits)]
        n = len(bad_speed)
        dim["invalid_speed_limits"] = n
        if n:
            issue(f"{n:,} rows have Speed_limit not in UK legal set {uk_speed_limits}", "accuracy", report, log_lines)
        else:
            log("  [PASS] Speed_limit — all values are valid UK legal limits", log_lines)

    # 1c. Coordinates must be within UK geographic bounds
    if column_exists(acc, "Latitude", log_lines) and column_exists(acc, "Longitude", log_lines):
        lat = to_numeric_safe(acc["Latitude"])
        lon = to_numeric_safe(acc["Longitude"])
        bad_lat = int(((lat < 49.0) | (lat > 61.0)).sum())
        bad_lon = int(((lon < -8.0) | (lon >  2.0)).sum())
        dim["out_of_bounds_latitude"]  = bad_lat
        dim["out_of_bounds_longitude"] = bad_lon
        if bad_lat:
            issue(f"{bad_lat:,} rows outside UK latitude range (49–61)", "accuracy", report, log_lines)
        else:
            log("  [PASS] Latitude — all within UK bounds (49–61)", log_lines)
        if bad_lon:
            issue(f"{bad_lon:,} rows outside UK longitude range (-8 to 2)", "accuracy", report, log_lines)
        else:
            log("  [PASS] Longitude — all within UK bounds (-8 to 2)", log_lines)

    # 1d. Casualty and vehicle counts must be >= 1
    for col in ["Number_of_Vehicles"]:
        if column_exists(acc, col, log_lines):
            bad = int((to_numeric_safe(acc[col]) <= 0).sum())
            dim[f"non_positive_{col}"] = bad
            if bad:
                issue(f"{bad:,} rows have {col} <= 0 (must be >= 1)", "accuracy", report, log_lines)
            else:
                log(f"  [PASS] {col} — all values >= 1", log_lines)

    # 1e. Vehicle age cannot be negative
    if column_exists(veh, "Age_of_Vehicle", log_lines):
        neg = int((to_numeric_safe(veh["Age_of_Vehicle"]) < 0).sum())
        dim["negative_vehicle_age"] = neg
        if neg:
            issue(f"{neg:,} vehicle records have negative Age_of_Vehicle", "accuracy", report, log_lines)
        else:
            log("  [PASS] Age_of_Vehicle — no negative values", log_lines)
    if column_exists(veh, "Age_Band_of_Driver", log_lines):
        unique_vals = veh["Age_Band_of_Driver"].dropna().unique()
        dim["age_band_unique_values_count"] = len(unique_vals)
        log(f"  [INFO] Age_Band_of_Driver has {len(unique_vals)} unique categories", log_lines)
        log(f"  Sample values: {list(unique_vals)[:10]}", log_lines)
    if column_exists(acc, "Day_of_Week", log_lines):
        bad_band = int((~acc["Day_of_Week"].isin(VALID_DAYS)).sum())
        dim["invalid_day_of_week"] = bad_band
        if bad_band:
            issue(f"{bad_band:,} rows have invalid Day_of_Week values", "accuracy", report, log_lines)
        else:
            log("  [PASS] Day_of_Week — all values valid", log_lines)

    if column_exists(veh,"Sex_of_Driver", log_lines):
        sex = veh["Sex_of_Driver"].astype(str).str.strip().str.capitalize()
        bad = ~sex.isin(["Male", "Female"])
        bad_band = int(bad.sum())
        dim["invalid_sex_band"] = bad_band
        if bad_band:
            issue(f"{bad_band:,} vehicle records have invalid Sex_of_Driver values", "accuracy", report, log_lines)
        else:
            log("  [PASS] Sex_of_Driver — all values valid sex", log_lines)

    report["dimensions"]["accuracy"] = dim


# ════════════════════════════════════════════════════════════════════════════
#  DIMENSION 2 — CONSISTENCY
#  Is data uniform across fields, formats, and both tables?
# ════════════════════════════════════════════════════════════════════════════

def check_consistency(acc: pd.DataFrame, veh: pd.DataFrame, log_lines: list, report: dict):
    """    
    Validates that data is uniform across fields and both tables by checking:
    - Date column parses correctly as datetime (YYYY-MM-DD format)
    - Year column matches the year extracted from Date
    - Day_of_Week matches the actual day calculated from Date
    - Time follows HH:MM format
    - Year values are consistent across accidents and vehicles tables for the same accident
    
    Args:
        acc (pd.DataFrame): The accidents DataFrame.
        veh (pd.DataFrame): The vehicles DataFrame.
        log_lines (list): The list for logging messages.
        report (dict): The main report dictionary to update with findings.
    """
    log("\n" + "-" * 65, log_lines)
    log("DIMENSION 2 — CONSISTENCY", log_lines)
    log("-" * 65, log_lines)
    dim = {}

    if column_exists(acc, "Date", log_lines):
        parsed = pd.to_datetime(acc["Date"], format='%Y-%m-%d', errors="coerce")

        # 2a. Date column must parse as datetime
        n_failed = int(parsed.isna().sum() - acc["Date"].isna().sum())
        dim["date_parse_failures"] = n_failed
        if n_failed:
            issue(f"{n_failed:,} Date values could not be parsed as datetime", "consistency", report, log_lines)
        else:
            log("  [PASS] Date column — parses correctly as datetime", log_lines)

        # 2b. Year column must match year extracted from Date
        if column_exists(acc, "Year", log_lines):
            mismatch = int(
            (parsed.dt.year != to_numeric_safe(acc["Year"])).sum()
        )
        dim["year_date_mismatch"] = mismatch
        if mismatch:
            issue(f"{mismatch:,} rows where Year column does not match year in Date", "consistency", report, log_lines)
        else:
            log("  [PASS] Year column — matches year extracted from Date in all rows", log_lines)

        # 2c. Day_of_Week must match actual day derived from Date
        # UK STATS19 encoding: 1=Sunday, 2=Monday ... 7=Saturday
    if column_exists(acc, "Day_of_Week", log_lines) and column_exists(acc, "Date", log_lines):
        parsed = pd.to_datetime(acc["Date"], format='%Y-%m-%d', errors="coerce")
        actual_day_names = parsed.dt.day_name()  
        recorded_day_names = acc["Day_of_Week"]
        
        mismatches = (actual_day_names != recorded_day_names).sum()
        if mismatches:
            log(f"  [WARN] {mismatches:,} Date/Day_of_Week mismatches found", log_lines)
        else:
            log(f"  [PASS] All Day_of_Week values match their dates", log_lines)
    # 2d. Time must follow HH:MM format
    if column_exists(acc, "Time", log_lines) :
        bad_time = int(
            (acc["Time"].dropna().str.match(r"^\d{1,2}:\d{2}$") == False).sum()
        )
        dim["invalid_time_format"] = bad_time
        if bad_time:
            issue(f"{bad_time:,} Time values do not follow HH:MM format", "consistency", report, log_lines)
        else:
            log("  [PASS] Time column — all values follow HH:MM format", log_lines)

    # 2e. Year must be consistent across both tables for the same accident
    if column_exists(acc, "Year", log_lines) and column_exists(veh, "Year", log_lines):
        merged = acc[["Accident_Index", "Year"]].merge(
            veh[["Accident_Index", "Year"]].drop_duplicates("Accident_Index"),
            on="Accident_Index", suffixes=("_acc", "_veh"), how="inner"
        )
        cross_mismatch = int(
            (to_numeric_safe(merged["Year_acc"]) !=
             to_numeric_safe(merged["Year_veh"])).sum()
        )
        dim["cross_table_year_mismatch"] = cross_mismatch
        if cross_mismatch:
            issue(f"{cross_mismatch:,} accidents have different Year in accidents vs vehicles table",
                  "consistency", report, log_lines)
        else:
            log("  [PASS] Year — consistent across both tables for all matched accidents", log_lines)

    report["dimensions"]["consistency"] = dim

# ════════════════════════════════════════════════════════════════════════════
#  DIMENSION 3 — COMPLETENESS
#  Is all required data present? Missing value analysis per column.
# ════════════════════════════════════════════════════════════════════════════

def check_completeness(acc: pd.DataFrame, veh: pd.DataFrame, log_lines: list, report: dict):
    """
    Performs comprehensive missing value analysis:
    - Identifies all columns with missing data
    - Calculates missing counts and percentages
    - Flags required columns with any nulls
    - Flags any column with >20% missing values
    - Displays top 12 columns by missing percentage
    
    Args:
        acc (pd.DataFrame): The accidents DataFrame.
        veh (pd.DataFrame): The vehicles DataFrame.
        log_lines (list): The list for logging messages.
        report (dict): The main report dictionary to update with findings.
    """
    log("\n" + "-" * 65, log_lines)
    log("DIMENSION 3 — COMPLETENESS", log_lines)
    log("-" * 65, log_lines)
    dim = {}

    REQUIRED_ACC = [
        "Accident_Index", "Accident_Severity", "Date", "Latitude", "Longitude",
        "Year", "Speed_limit", "Road_Type", "Weather_Conditions",
        "Light_Conditions", "Road_Surface_Conditions", "Urban_or_Rural_Area"
    ]
    REQUIRED_VEH = [
        "Accident_Index", "Vehicle_Type", "Vehicle_Manoeuvre", "Age_Band_of_Driver"
    ]

    for df, label, required in [
        (acc, "ACCIDENTS", REQUIRED_ACC),
        (veh, "VEHICLES",  REQUIRED_VEH)
    ]:
        log(f"\n  [{label}]", log_lines)
        total = len(df)
        missing_summary = {}

        for col in df.columns:
            n = int(df[col].isna().sum())
            if n > 0:
                missing_summary[col] = {
                    "count": n,
                    "percent": round(n / total * 100, 2)
                }

        dim[f"{label.lower()}_missing"] = missing_summary

        # Print top 12 columns by missing %
        sorted_m = sorted(missing_summary.items(),
                          key=lambda x: x[1]["percent"], reverse=True)
        log(f"  {'Column':<46} {'Missing':>9}  {'%':>6}", log_lines)
        log(f"  {'-'*46} {'-'*9}  {'-'*6}", log_lines)
        for col, info in sorted_m[:12]:
            marker = "  <- REQUIRED" if col in required else ""
            log(f"  {col:<46} {info['count']:>9,}  {info['percent']:>5.1f}%{marker}", log_lines)
        if len(sorted_m) > 12:
            log(f"  ... and {len(sorted_m)-12} more columns with missing values", log_lines)

        # Flag required columns with any nulls
        crit = [c for c in required if c in df.columns and df[c].isna().sum() > 0]
        if crit:
            for c in crit:
                issue(f"[{label}] Required column '{c}' has "
                      f"{missing_summary[c]['count']:,} null values "
                      f"({missing_summary[c]['percent']:.1f}%)", "completeness", report, log_lines)
        else:
            log(f"\n  [PASS] No nulls in any required columns for {label}")

        
        high = [(c, v) for c, v in sorted_m if v["percent"] > MISSING_VALUE_THRESHOLD]
        if high:
            for c, v in high:
                issue(f"[{label}] '{c}' is {v['percent']:.1f}% missing (>{MISSING_VALUE_THRESHOLD}% threshold)",
                      "completeness", report, log_lines)

    report["dimensions"]["completeness"] = dim


# ════════════════════════════════════════════════════════════════════════════
#  DIMENSION 4 — UNIQUENESS
#  Exact duplicates, key-level duplicates, composite key check.
# ════════════════════════════════════════════════════════════════════════════

def check_uniqueness(acc: pd.DataFrame, veh: pd.DataFrame, log_lines: list, report: dict):
    """
    Performs duplicate detection at multiple levels:
    - Exact duplicates (all fields identical) in both tables
    - Accident_Index duplicates in accidents table (should be unique)
    - Accident_Index duplicates in vehicles table (expected by design)
    - Composite key (Accident_Index, Vehicle_Reference) duplicates in vehicles
    
    Args:
        acc (pd.DataFrame): The accidents DataFrame.
        veh (pd.DataFrame): The vehicles DataFrame.
        log_lines (list): The list for logging messages.
        report (dict): The main report dictionary to update with findings.
    """    
    log("\n" + "-" * 65, log_lines)
    log("DIMENSION 4 — UNIQUENESS", log_lines)
    log("-" * 65, log_lines)
    dim = {}

    # Accidents: Accident_Index MUST be unique (one row per accident)
    acc_exact = int(acc.duplicated().sum())
    acc_key   = int(acc["Accident_Index"].duplicated().sum())
    dim["accidents_exact_duplicates"] = acc_exact
    dim["accidents_key_duplicates"]   = acc_key

    if acc_exact:
        issue(f"Accidents table: {acc_exact:,} fully duplicate rows", "uniqueness", report, log_lines)
    else:
        log("  [PASS] Accidents — zero fully duplicate rows", log_lines)

    if acc_key:
        issue(f"Accidents table: {acc_key:,} duplicate Accident_Index values "
              f"(each accident must appear exactly once)", "uniqueness", report, log_lines)
    else:
        log(f"  [PASS] Accidents — all {len(acc):,} Accident_Index values are unique", log_lines)

    # Vehicles: Accident_Index duplicates are EXPECTED (multiple vehicles per accident)
    veh_exact = int(veh.duplicated().sum())
    veh_key   = int(veh["Accident_Index"].duplicated().sum())
    dim["vehicles_exact_duplicates"]             = veh_exact
    dim["vehicles_duplicate_accident_index"]     = veh_key
    dim["vehicles_duplicate_accident_index_note"] = (
        "Duplicate Accident_Index values in the vehicles table are EXPECTED "
        "by design — multiple vehicles can be involved in one accident."
    )

    if veh_exact:
        issue(f"Vehicles table: {veh_exact:,} fully duplicate rows "
              f"(all fields identical — unexpected)", "uniqueness", report, log_lines)
    else:
        log("  [PASS] Vehicles — zero fully duplicate rows", log_lines)

    log(f"  [INFO] Vehicles — {veh_key:,} duplicate Accident_Index values (expected by design)", log_lines)

    # Vehicles composite key: (Accident_Index + Vehicle_Reference) must be unique
    if column_exists(veh, "Vehicle_Reference", log_lines) and column_exists(veh, "Accident_Index", log_lines):
        comp_dups = int(
            veh.duplicated(subset=["Accident_Index", "Vehicle_Reference"]).sum()
        )
        dim["vehicles_composite_key_duplicates"] = comp_dups
        if comp_dups:
            issue(f"Vehicles: {comp_dups:,} duplicate (Accident_Index, Vehicle_Reference) pairs",
                  "uniqueness", report, log_lines)
        else:
            log("  [PASS] Vehicles — (Accident_Index, Vehicle_Reference) composite key is fully unique", log_lines)

    report["dimensions"]["uniqueness"] = dim


# ════════════════════════════════════════════════════════════════════════════
#  DIMENSION 5 — OUTLIERS
#  IQR method + Z-score method + Isolation Forest (all 3 from lecture)
# ════════════════════════════════════════════════════════════════════════════

def check_outliers(acc: pd.DataFrame, veh: pd.DataFrame, log_lines: list, report: dict):
    """
    Applies three outlier detection approaches:
    1. IQR Method (univariate): Flags values outside Q1 - 1.5*IQR to Q3 + 1.5*IQR
    2. Isolation Forest (multivariate): Detects anomalies using ensemble method
       with 5% contamination rate
    
    Args:
        acc (pd.DataFrame): The accidents DataFrame.
        veh (pd.DataFrame): The vehicles DataFrame.
        log_lines (list): The list for logging messages.
        report (dict): The main report dictionary to update with findings.
    """
    log("\n" + "-" * 65, log_lines)
    log("DIMENSION 5 — OUTLIERS", log_lines)
    log("-" * 65, log_lines)
    dim = {}

    COLS_ACC = ["Speed_limit", "Number_of_Casualties", "Number_of_Vehicles"]
    COLS_VEH = ["Age_of_Vehicle", "Engine_Capacity_.CC."]

    # ── IQR method ──────────────────────────────────────────────────────────
    log("\n  IQR Method  (bounds = Q1 - 1.5*IQR  and  Q3 + 1.5*IQR):", log_lines)
    log(f"  {'Column':<42} {'Outliers':>10}  {'Lower fence':>12}  {'Upper fence':>12}", log_lines)
    log(f"  {'-'*42} {'-'*10}  {'-'*12}  {'-'*12}", log_lines)

    iqr_res = {}
    for df, cols, label in [(acc, COLS_ACC, "accidents"),
                             (veh, COLS_VEH, "vehicles")]:
        for col in cols:
            if col not in df.columns:
                continue
            s = to_numeric_safe(df[col]).dropna()
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR = Q3 - Q1
            lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            n_out = int(((s < lo) | (s > hi)).sum())
            key = f"{label}.{col}"
            iqr_res[key] = {"count": n_out,
                            "lower_fence": round(float(lo), 2),
                            "upper_fence": round(float(hi), 2)}
            log(f"  {key:<42} {n_out:>10,}  {lo:>12.2f}  {hi:>12.2f}")
    dim["iqr"] = iqr_res

    # ── Isolation Forest (multivariate) ────────────────────────────────────
    log("\n  Isolation Forest  (contamination=0.05, multivariate):", log_lines)
    iso_res = {}

    for df, cols, label in [(acc, COLS_ACC, "accidents"),
                             (veh, COLS_VEH, "vehicles")]:
        avail = [c for c in cols if c in df.columns]
        if len(avail) < 2:
            continue
        sub = df[avail].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < 100:
            log(f"  [SKIP] {label} — too few non-null rows")
            continue
        sample = sub.sample(min(100_000, len(sub)), random_state=42)
        clf = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        preds = clf.fit_predict(sample)
        n_anom = int((preds == -1).sum())
        rate   = round(n_anom / len(sample) * 100, 2)
        iso_res[label] = {
            "features_used":      avail,
            "sample_size":        len(sample),
            "anomalies_detected": n_anom,
            "anomaly_rate_pct":   rate
        }
        log(f"  {label}: {n_anom:,} anomalies in {len(sample):,} rows ({rate:.1f}%)", log_lines)

    dim["isolation_forest"] = iso_res
    log("\n  [NOTE] Outliers are FLAGGED only — removal decisions made in cleaning step.", log_lines)
    report["dimensions"]["outliers"] = dim


# ════════════════════════════════════════════════════════════════════════════
#  DIMENSION 6 — TIMELINESS
#  Date range validity, year coverage, COVID dip detection.
# ════════════════════════════════════════════════════════════════════════════

def check_timeliness(acc: pd.DataFrame, log_lines: list, report: dict):
    """
    Performs temporal validation checks:
    - Identifies earliest and latest record dates
    - Flags records with future dates (after current date)
    - Flags records with dates before 2005 (outside expected UK STATS19 range)
    - Documents date range coverage
    
    Args:
        acc (pd.DataFrame): The accidents DataFrame.
        log_lines (list): The list for logging messages.
        report (dict): The main report dictionary to update with findings.
    """
    log("\n" + "-" * 65, log_lines)
    log("DIMENSION 6 — TIMELINESS", log_lines)
    log("-" * 65, log_lines)
    dim = {}

    if not column_exists(acc, "Date", log_lines):
        log("  [SKIP] No Date column", log_lines)
        return

    parsed = pd.to_datetime(acc["Date"], errors="coerce")
    valid  = parsed.dropna()

    dim["earliest_date"] = str(valid.min().date()) if len(valid) else "N/A"
    dim["latest_date"]   = str(valid.max().date()) if len(valid) else "N/A"
    log(f"  Earliest record : {dim['earliest_date']}", log_lines)
    log(f"  Latest record   : {dim['latest_date']}", log_lines)

    # Future dates
    future = int((valid > pd.Timestamp.now()).sum())
    dim["future_dates"] = future
    if future:
        issue(f"{future:,} records have future dates", "timeliness", report, log_lines)
    else:
        log("  [PASS] No future dates", log_lines)

    # Pre-2005 dates
    pre2005 = int((valid < pd.Timestamp("2005-01-01")).sum())
    dim["pre_2005_dates"] = pre2005
    if pre2005:
        issue(f"{pre2005:,} records have dates before 2005", "timeliness", report, log_lines)
    else:
        log("  [PASS] No dates before 2005", log_lines)


    report["dimensions"]["timeliness"] = dim


# ════════════════════════════════════════════════════════════════════════════
#  DIMENSION 7 — DISTRIBUTION PROFILE
#  Descriptive stats + skewness + kurtosis + KS test vs normal
# ════════════════════════════════════════════════════════════════════════════

def check_distribution(acc: pd.DataFrame, veh: pd.DataFrame, log_lines: list, report: dict):
    """    
    Performs comprehensive distribution analysis:
    - Descriptive statistics (min, max, mean, median, std)
    - Shape metrics (skewness, kurtosis)
    - Cardinality (unique value counts)
    - Histograms (10 equal-width bins with ASCII bar charts)
    - Kolmogorov-Smirnov test against normal distribution
    - Target class distribution (Accident_Severity)
    - Categorical feature value distributions
    
    Args:
        acc (pd.DataFrame): The accidents DataFrame.
        veh (pd.DataFrame): The vehicles DataFrame.
        log_lines (list): The list for logging messages.
        report (dict): The main report dictionary to update with findings.
    """    
    log("\n" + "-" * 65, log_lines)
    log("DIMENSION 7 — DISTRIBUTION PROFILE", log_lines)
    log("-" * 65, log_lines)
    dim = {}

    COLS_ACC = ["Speed_limit", "Number_of_Casualties", "Number_of_Vehicles",
                "Latitude", "Longitude"]
    COLS_VEH = ["Age_of_Vehicle", "Engine_Capacity_.CC."]

    log(f"\n  {'Column':<42} {'Min':>7} {'Max':>8} {'Mean':>8} "
        f"{'Median':>8} {'Std':>8} {'Skew':>7} {'Kurt':>7}  {'Card':>6}", log_lines)
    log(f"  {'-'*42} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*7}  {'-'*6}", log_lines)

    stats_res = {}
    ks_res    = {}
    hist_res  = {}

    for df, cols, label in [(acc, COLS_ACC, "accidents"),
                             (veh, COLS_VEH, "vehicles")]:
        for col in cols:
            if not column_exists(df, col, log_lines):
                continue
            s = to_numeric_safe(df[col]).dropna()
            if len(s) < 10:
                continue
            
            # Cardinality
            cardinality = s.nunique()
            
            sk = float(skew(s))
            ku = float(kurtosis(s))
            cs = {
                "min":      round(float(s.min()), 3),
                "max":      round(float(s.max()), 3),
                "mean":     round(float(s.mean()), 3),
                "median":   round(float(s.median()), 3),
                "std":      round(float(s.std()), 3),
                "skewness": round(sk, 3),
                "kurtosis": round(ku, 3),
                "n_valid":  len(s),
                "cardinality": cardinality
            }
            key = f"{label}.{col}"
            stats_res[key] = cs
            log(f"  {key:<42} {cs['min']:>7.1f} {cs['max']:>8.1f} {cs['mean']:>8.2f} "
                f"{cs['median']:>8.2f} {cs['std']:>8.2f} {sk:>7.2f} {ku:>7.2f}  {cardinality:>6,}", log_lines)

            # Histogram (10 equal-width bins)
            hist_counts, bin_edges = np.histogram(s, bins=10)
            histogram = {
                "n_bins": 10,
                "bin_edges": [round(float(x), 3) for x in bin_edges],
                "bin_counts": [int(c) for c in hist_counts],
                "histogram_pct": [round(100 * c / len(s), 1) for c in hist_counts]
            }
            hist_res[key] = histogram
            
            # Print histogram as ASCII bar chart
            log(f"\n    Histogram ({key}):", log_lines)
            for i, (count, pct) in enumerate(zip(hist_counts, histogram["histogram_pct"])):
                bin_start = round(bin_edges[i], 2)
                bin_end = round(bin_edges[i+1], 2)
                bar = "▓" * max(1, int(pct / 2))  # Scale to 50% width
                log(f"      [{bin_start:>8} – {bin_end:>8}]  {count:>6,}  ({pct:>5.1f}%)  {bar}", log_lines)

            # KS test against normal distribution
            samp = s.sample(min(KS_SAMPLE_SIZE, len(s)), random_state=42)
            ks_stat, p_val = stats.kstest(
                samp, "norm", args=(float(samp.mean()), float(samp.std()))
            )
            ks_res[key] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value":      round(float(p_val), 6),
                "interpretation": (
                    "NOT normally distributed (p < 0.05) — use Spearman for correlations"
                    if p_val < 0.05 else
                    "Cannot reject normality (p >= 0.05) — Pearson is appropriate"
                )
            }

    dim["descriptive_stats"] = stats_res
    dim["histograms"] = hist_res
    dim["ks_test_vs_normal"] = ks_res

    # KS results summary
    log("\n  KS Test vs Normal Distribution:", log_lines)
    log(f"  {'Column':<42} {'KS stat':>9}  {'p-value':>10}  Interpretation", log_lines)
    log(f"  {'-'*42} {'-'*9}  {'-'*10}  {'-'*40}", log_lines)
    for key, res in ks_res.items():
        log(f"  {key:<42} {res['ks_statistic']:>9.4f}  {res['p_value']:>10.6f}  "
            f"{res['interpretation'][:45]}", log_lines)

    # Target class distribution
    if column_exists(acc, "Accident_Severity", log_lines):
        severity_dist = acc["Accident_Severity"].value_counts()
        dim["target_severity_distribution"] = severity_dist.to_dict()

    # Categorical feature distributions
    CATEGORICAL_COLS_ACC = [
        "Day_of_Week", "Light_Conditions", "Weather_Conditions", 
        "Road_Surface_Conditions", "Urban_or_Rural_Area", "Road_Type"
    ]
    CATEGORICAL_COLS_VEH = [
        "Age_Band_of_Driver", "Sex_of_Driver", "Vehicle_Type", "Vehicle_Manoeuvre"
    ]

    dim["categorical_distributions"] = {}

    for df, cols, label in [(acc, CATEGORICAL_COLS_ACC, "accidents"),
                             (veh, CATEGORICAL_COLS_VEH, "vehicles")]:
        for col in cols:
            if column_exists(df, col, log_lines):
                value_counts = df[col].value_counts()
                dim["categorical_distributions"][f"{label}_{col}"] = value_counts.to_dict()
    
    report["dimensions"]["distribution"] = dim


# ════════════════════════════════════════════════════════════════════════════
#  DIMENSION 8 — RELATIONSHIPS
#  Pearson vs Spearman on numeric pairs, flag high correlations
# ════════════════════════════════════════════════════════════════════════════

def check_relationships(acc: pd.DataFrame, log_lines: list, report: dict):
    """    
    Performs correlation analysis:
    - Computes Pearson correlation matrix (linear relationships)
    - Computes Spearman correlation matrix (monotonic relationships)
    - Compares Pearson vs Spearman for each pair (|diff| > 0.10 indicates non-linear)
    - Flags high correlations (|r| > 0.85) for potential feature selection

    Args:
        acc (pd.DataFrame): The accidents DataFrame.
        log_lines (list): The list for logging messages.
        report (dict): The main report dictionary to update with findings.
    """
    log("\n" + "-" * 65, log_lines)
    log("DIMENSION 8 — RELATIONSHIPS", log_lines)
    log("-" * 65, log_lines)
    dim = {}

    NUM_COLS = [c for c in [
        "Speed_limit", "Number_of_Casualties", "Number_of_Vehicles",
        "Latitude", "Longitude"
    ] if column_exists(acc, c, log_lines)]

    if len(NUM_COLS) < 2:
        log("  [SKIP] Not enough numeric columns")
        return

    sub = acc[NUM_COLS].apply(pd.to_numeric, errors="coerce").dropna()
    pearson_m  = sub.corr(method="pearson").round(3)
    spearman_m = sub.corr(method="spearman").round(3)

    dim["pearson_matrix"]  = pearson_m.to_dict()
    dim["spearman_matrix"] = spearman_m.to_dict()

    # Print Pearson matrix
    log("\n  Pearson Correlation Matrix:", log_lines)
    header = "  " + " " * 25 + "".join(f"{c[:11]:>13}" for c in NUM_COLS)
    log(header, log_lines)
    for row in NUM_COLS:
        vals = "".join(f"{pearson_m.loc[row, c]:>13.3f}" for c in NUM_COLS)
        log(f"  {row[:24]:<25}{vals}", log_lines)

    # Pairwise Pearson vs Spearman comparison
    log("\n  Pearson vs Spearman  (|diff| > 0.10 = non-linear relationship detected):", log_lines)
    log(f"  {'Pair':<44} {'Pearson':>9}  {'Spearman':>9}  {'|Diff|':>8}  Note", log_lines)
    log(f"  {'-'*44} {'-'*9}  {'-'*9}  {'-'*8}  {'-'*25}", log_lines)

    pair_res = {}
    high_corr = []

    for i, c1 in enumerate(NUM_COLS):
        for c2 in NUM_COLS[i+1:]:
            p   = float(pearson_m.loc[c1, c2])
            sp  = float(spearman_m.loc[c1, c2])
            diff = abs(p - sp)
            note = "non-linear" if diff > NON_LINEAR_DIFF_THRESHOLD else ""
            key  = f"{c1} x {c2}"
            pair_res[key] = {"pearson": round(p,3), "spearman": round(sp,3),
                            "diff": round(diff,3)}
            log(f"  {key:<44} {p:>9.3f}  {sp:>9.3f}  {diff:>8.3f}  {note}", log_lines)

            if abs(p) > HIGH_CORR_THRESHOLD or abs(sp) > HIGH_CORR_THRESHOLD:
                high_corr.append(key)
                issue(
                    f"High correlation: '{c1}' x '{c2}' "
                    f"(Pearson={p:.3f}, Spearman={sp:.3f}) — "
                    f"consider dropping one in feature selection",
                    "relationships", report, log_lines
                )

    dim["pair_comparison"] = pair_res
    dim["high_correlation_pairs"] = high_corr

    if not high_corr:
        log("\n  [PASS] No high-correlation pairs found (threshold |r| > 0.85)", log_lines)

    report["dimensions"]["relationships"] = dim


# ════════════════════════════════════════════════════════════════════════════
#  JOIN INTEGRITY — multi-source merge validation (rubric requirement)
# ════════════════════════════════════════════════════════════════════════════

def check_join_integrity(acc: pd.DataFrame, veh: pd.DataFrame, log_lines: list, report: dict):
    """
    Validates multi-source merge integrity by analyzing:
    - Unique Accident_Index values in each table
    - Number of keys that match (will survive inner join)
    - Orphaned accidents (no vehicle records)
    - Orphaned vehicles (no matching accident record)
    
    Args:
        acc (pd.DataFrame): The accidents DataFrame.
        veh (pd.DataFrame): The vehicles DataFrame.
        log_lines (list): The list for logging messages.
        report (dict): The main report dictionary to update with findings.
    """
    log("\n" + "-" * 65, log_lines)
    log("JOIN INTEGRITY  (Accident_Index key across both tables)", log_lines)
    log("-" * 65, log_lines)
    dim = {}

    acc_keys = set(acc["Accident_Index"].dropna())
    veh_keys = set(veh["Accident_Index"].dropna())
    matched  = acc_keys & veh_keys

    dim["accidents_unique_keys"] = len(acc_keys)
    dim["vehicles_unique_keys"]  = len(veh_keys)
    dim["matched_keys"]          = len(matched)
    dim["only_in_accidents"]     = len(acc_keys - veh_keys)
    dim["only_in_vehicles"]      = len(veh_keys - acc_keys)

    log(f"  Unique Accident_Index in accidents table : {len(acc_keys):>10,}", log_lines)
    log(f"  Unique Accident_Index in vehicles table  : {len(veh_keys):>10,}", log_lines)
    log(f"  Keys matched (will survive inner join)   : {len(matched):>10,}", log_lines)

    if dim["only_in_accidents"]:
        pct = dim["only_in_accidents"] / len(acc_keys) * 100
        issue(
            f"{dim['only_in_accidents']:,} accidents ({pct:.1f}%) have no vehicle record — "
            f"dropped on inner join, document in cleaning step",
            "join_integrity", report, log_lines
        )
    else:
        log("  [PASS] Every accident key has at least one vehicle record", log_lines)

    if dim["only_in_vehicles"]:
        pct = dim["only_in_vehicles"] / len(veh_keys) * 100
        issue(
            f"{dim['only_in_vehicles']:,} vehicle records ({pct:.1f}%) have no matching accident — "
            f"orphaned records, dropped on inner join",
            "join_integrity", report, log_lines
        )

    report["dimensions"]["join_integrity"] = dim


# ════════════════════════════════════════════════════════════════════════════
#  SAVE ALL OUTPUTS
# ════════════════════════════════════════════════════════════════════════════

def save_outputs(log_lines: list, report: dict):
    """    
    Generates three output files in the reports/ directory:
    1. validation_report.json - Machine-readable JSON with all findings
    2. validation_summary.txt - Plain-text log suitable for appendix
    3. validation_report.pdf - Formatted PDF with tables and summaries
    
    Args:
        log_lines (list): The list of all logged messages.
        report (dict): The final report dictionary containing all validation results.
    """
    total = len(report["summary"]["issues"])
    report["summary"]["total_issue_count"] = total

    # Create reports directory if it doesn't exist
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    log("\n" + "=" * 65, log_lines)
    log(f"  VALIDATION COMPLETE", log_lines)
    log(f"  Total issues found across all 8 dimensions: {total}", log_lines)
    log("=" * 65, log_lines)

    if report["summary"]["issues"]:
        log(f"\n  Issues by dimension:", log_lines)
        for issue_item in report["summary"]["issues"]:
            log(f"    [{issue_item['dimension'].upper()}] {issue_item['message']}", log_lines)
    else:
        log("\n  [PASS] No issues found!", log_lines)

    # Save JSON
    json_path = REPORTS_DIR / "validation_report.json"
    
    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    log(f"\n  [SAVED] {json_path}", log_lines)

    txt_path = REPORTS_DIR / "validation_summary.txt"
    with open(txt_path, "w", encoding='utf-8') as f:
        f.write("\n".join(log_lines))
    log(f"  [SAVED] {txt_path}", log_lines)

    pdf_path = REPORTS_DIR / "validation_report.pdf"
    generate_pdf_report(pdf_path, total, report, log_lines)
    log(f"  [SAVED] {pdf_path}", log_lines)
    


def generate_pdf_report(pdf_path, total, report,log_lines):
    """    
    Creates a formatted PDF document with:
    - Title page with generation timestamp
    - Summary table of issues by dimension
    - Detailed list of all issues found
    - Dimension-by-dimension statistics summary
    
    Args:
        pdf_path (str or Path): The file path to save the PDF.
        total (int): The total number of issues found across all dimensions.
        report (dict): The main report dictionary containing all findings.
    """
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        alignment=1  # Center
    )
    

    elements.append(Paragraph("UK Road Accidents — Data Validation Report", title_style))
    elements.append(Spacer(1, 0.2*inch))
    

    generated_at = report.get("generated_at", "N/A")
    elements.append(Paragraph(f"<b>Generated:</b> {generated_at}", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    

    elements.append(Paragraph(f"<b>Total Issues Found: {total}</b>", styles['Heading2']))
    elements.append(Spacer(1, 0.15*inch))
    

    issue_counts = {}
    for issue_item in report["summary"]["issues"]:
        dim = issue_item.get("dimension", "UNKNOWN").upper()
        issue_counts[dim] = issue_counts.get(dim, 0) + 1
    
    if issue_counts:
        table_data = [["Dimension", "Issues"]]
        for dim, count in sorted(issue_counts.items()):
            table_data.append([dim, str(count)])
        
        table = Table(table_data, colWidths=[3*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
    else:
        elements.append(Paragraph("<b>✓ All dimensions passed validation</b>", styles['Normal']))
    
    elements.append(Spacer(1, 0.3*inch))
    elements.append(PageBreak())
    

    # if report["summary"]["issues"]:
    #     elements.append(Paragraph("Detailed Issues", styles['Heading2']))
    #     elements.append(Spacer(1, 0.15*inch))
        
    #     for issue_item in report["summary"]["issues"]:
    #         issue_text = f"<b>[{issue_item.get('dimension', 'UNKNOWN')}]</b> {issue_item.get('message', 'N/A')}"
    #         elements.append(Paragraph(issue_text, styles['Normal']))
    #         elements.append(Spacer(1, 0.1*inch))
    
    # ── Histogram pages ──────────────────────────────────
    dist = report.get("dimensions", {}).get("distribution", {})
    hist_res   = dist.get("histograms", {})
    
    if hist_res:
        elements.append(PageBreak())
        elements.append(Paragraph("Distribution Histograms", styles['Heading2']))
        elements.append(Spacer(1, 0.15*inch))
        buf_store = [] 
        for key, hist_data in hist_res.items():
            elements.append(Paragraph(f"<b>{key}</b>", styles['Heading3']))
            img = generate_histogram_image(key, hist_data, buf_store)
            elements.append(img)
            elements.append(Spacer(1, 0.2*inch))
    # ── Full Validation Log ──────────────────────────────
    elements.append(PageBreak())
    elements.append(Paragraph("Full Validation Log", styles['Heading2']))
    elements.append(Spacer(1, 0.15*inch))

    mono_style = ParagraphStyle(
        'Mono',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=6.5,
        leading=9,
        wordWrap='CJK',  # prevents long lines from overflowing
    )

    for line in log_lines:
        safe_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        elements.append(Paragraph(safe_line if safe_line.strip() else "&nbsp;", mono_style))

    # if report.get("dimensions"):
    #     elements.append(PageBreak())
    #     elements.append(Paragraph("Validation Summary by Dimension", styles['Heading2']))
    #     elements.append(Spacer(1, 0.15*inch))
        
    #     for dim_name, dim_data in report["dimensions"].items():
    #         if isinstance(dim_data, dict):
    #             elements.append(Paragraph(f"<b>{dim_name.upper()}</b>", styles['Heading3']))
    #             summary_text = f"{len(dim_data)} checks performed"
    #             elements.append(Paragraph(summary_text, styles['Normal']))
    #             elements.append(Spacer(1, 0.1*inch))
    
    doc.build(elements)


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    """    
    Orchestrates the complete validation process by:
    1. Loading both data tables
    2. Running all 8 validation dimensions in sequence
    3. Checking join integrity between tables
    4. Saving outputs in multiple formats
    
    Returns:
        The complete validation report.
    """
    acc, veh = load_tables(log_lines)
    check_accuracy(acc, veh, log_lines, report)         
    check_consistency(acc, veh, log_lines, report )     
    check_completeness(acc, veh, log_lines, report)    
    check_uniqueness(acc, veh, log_lines, report)       
    check_outliers(acc, veh, log_lines, report)         
    check_timeliness(acc, log_lines, report)            
    check_distribution(acc, veh, log_lines, report)     
    check_relationships(acc, log_lines, report)         
    check_join_integrity(acc, veh, log_lines, report)   
    save_outputs(log_lines, report)
    return report


if __name__ == "__main__":
    main()