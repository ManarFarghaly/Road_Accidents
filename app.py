"""
UK Road Accidents — Streamlit Analytics Dashboard

Professional analytics dashboard for 2.7M UK road accident records (2005–2017).
Designed so the critical risk insight is visible within 5 seconds.

Data sources (tried in order):
  1. reports/eda_summary.json  — pre-computed by src.eda (run src.data.ingest first)
  2. data/interim/merged.parquet — raw parquet sampled via pyarrow (no Spark needed)
  3. Built-in demo data         — realistic UK STATS19 distributions, works offline

Run:
    streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="UK Road Accidents Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent
EDA_JSON = ROOT / "reports" / "eda_summary.json"
PARQUET  = ROOT / "data" / "interim" / "merged.parquet"

# ─── Design tokens ───────────────────────────────────────────────────────────
C_TEAL    = "#1abc9c"
C_CORAL   = "#e74c3c"
C_BLUE    = "#3498db"
C_ORANGE  = "#f39c12"
C_DARK    = "#2c3e50"
C_MID     = "#566573"  # darkened for WCAG AA (5.5:1 on white)
C_LIGHT   = "#ecf0f1"
C_BG      = "#f4f7f9"

C_FATAL   = C_CORAL
C_SERIOUS = C_ORANGE
C_SLIGHT  = C_TEAL

SEV_ORDER  = ["Fatal", "Serious", "Slight"]
SEV_COLORS = {"Fatal": C_FATAL, "Serious": C_SERIOUS, "Slight": C_SLIGHT}
DOW_ORDER  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ─── Global Plotly template ───────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    font_family="Inter, system-ui, sans-serif",
    font_color=C_DARK,
    paper_bgcolor="white",
    plot_bgcolor="white",
    hoverlabel=dict(bgcolor="white", font_size=13, bordercolor="#e0e0e0"),
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    .stApp { background-color: #f4f7f9; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    div[data-testid="stHorizontalBlock"] { gap: 0 !important; }

    /* ── Top bar ── */
    .topbar {
        background: #2c3e50;
        padding: 0.7rem 2rem 0.65rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0;
    }
    .topbar-title {
        color: white;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        margin: 0;
    }
    .topbar-sub {
        color: #aab7c4;  /* on dark #2c3e50 bg: 4.6:1 ✓ */
        font-size: 0.78rem;
        margin: 0;
    }
    .demo-pill {
        background: #e74c3c;
        color: white;
        border-radius: 20px;
        padding: 0.18rem 0.75rem;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }

    /* ── Page padding ── */
    .page-body { padding: 0.35rem 1.5rem 2rem; }

    /* ── KPI card ── */
    .kpi-card {
        background: white;
        border-radius: 8px;
        padding: 1rem 1.25rem 0.85rem;
        box-shadow: 0 1px 5px rgba(0,0,0,0.07);
        height: 100%;
        border-bottom: 4px solid #ddd;
        position: relative;
    }
    .kpi-label {
        font-size: 0.67rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #5d6d7e;  /* 5.0:1 on white ✓ WCAG AA */
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        color: #2c3e50;
        line-height: 1.1;
        margin-bottom: 0.25rem;
    }
    .kpi-delta {
        font-size: 0.75rem;
        font-weight: 600;
    }
    .kpi-delta.up   { color: #e74c3c; }
    .kpi-delta.down { color: #1abc9c; }
    .kpi-delta.neu  { color: #566573; }  /* 5.5:1 on white ✓ WCAG AA */

    /* ── Chart card ── */
    .chart-card {
        background: white;
        border-radius: 8px;
        box-shadow: 0 1px 5px rgba(0,0,0,0.07);
        padding: 0.1rem 0.1rem 0;
        height: 100%;
    }
    .chart-title {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #5d6d7e;  /* 5.0:1 on white ✓ WCAG AA */
        padding: 0.75rem 1rem 0;
        margin: 0;
    }

    /* ── Section header ── */
    .section-label {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #5d6d7e;  /* 5.0:1 on #f4f7f9 ✓ WCAG AA */
        margin: 0.25rem 0 0.45rem;
    }

    /* ── Footer ── */
    .dash-footer {
        font-size: 0.7rem;
        color: #566573;  /* 5.5:1 on white ✓ WCAG AA */
        border-top: 1px solid #e0e0e0;
        padding-top: 0.6rem;
        margin-top: 1rem;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def _normalise_severity_keys(sev_dict: dict) -> dict:
    """
    Normalise severity keys to "Fatal" / "Serious" / "Slight" regardless of
    whether the source used STATS19 numeric codes (1/2/3) or string labels.
    """
    mapping = {
        "1": "Fatal",   "Fatal":   "Fatal",
        "2": "Serious", "Serious": "Serious",
        "3": "Slight",  "Slight":  "Slight",
    }
    return {mapping.get(str(k), str(k)): v for k, v in sev_dict.items()}


@st.cache_data(show_spinner="Loading analytics report …")
def load_eda_report() -> dict | None:
    if not EDA_JSON.exists():
        return None
    with open(EDA_JSON) as f:
        report = json.load(f)
    if "severity_distribution" in report:
        report["severity_distribution"] = _normalise_severity_keys(
            report["severity_distribution"]
        )
    return report


@st.cache_data(show_spinner="Sampling accident locations …")
def load_geo_sample(n: int = 40_000) -> pd.DataFrame | None:
    if not PARQUET.exists():
        return None
    try:
        import pyarrow.parquet as pq
        wanted = ["Latitude", "Longitude", "Accident_Severity"]
        df = pq.ParquetDataset(str(PARQUET)).read(columns=wanted).to_pandas()
        df = df.dropna(subset=["Latitude", "Longitude"])
        df["Accident_Severity"] = df["Accident_Severity"].astype(str)
        sev_map = {"1": "Fatal", "2": "Serious", "3": "Slight"}
        df["Accident_Severity"] = df["Accident_Severity"].replace(sev_map)
        if len(df) > n:
            df = df.sample(n, random_state=42)
        return df
    except Exception:
        return None


def _demo_report() -> dict:
    """
    Realistic UK STATS19 distributions for demo / when pipeline has not been run.
    All counts are proportional to the full 2 715 940-row dataset.
    """
    return {
        "_source": "demo",
        "dataset_shape": {"total_rows": 2_715_940, "total_columns": 64},
        "severity_distribution": {
            "Fatal":   {"count": 27_159,   "percentage": 1.0,  "avg_casualties": 1.41},
            "Serious": {"count": 380_231,  "percentage": 14.0, "avg_casualties": 1.22},
            "Slight":  {"count": 2_308_550,"percentage": 85.0, "avg_casualties": 1.10},
        },
        "weather_analysis": {
            "Fine no high winds":    {"count": 1_903_158, "avg_casualties": 1.12},
            "Raining no high winds": {"count": 421_270,  "avg_casualties": 1.13},
            "Fine + high winds":     {"count": 89_625,   "avg_casualties": 1.16},
            "Raining + high winds":  {"count": 62_460,   "avg_casualties": 1.17},
            "Fog or mist":           {"count": 15_340,   "avg_casualties": 1.29},
            "Snowing no high winds": {"count": 12_110,   "avg_casualties": 1.09},
            "Snowing + high winds":  {"count":  3_820,   "avg_casualties": 1.11},
            "Other":                 {"count": 31_200,   "avg_casualties": 1.10},
        },
        "temporal_patterns": {
            "hourly": {
                "0": 8200,  "1": 4100,  "2": 2500,  "3": 1500,  "4": 1200,
                "5": 3900,  "6": 11000, "7": 26000, "8": 38000, "9": 26000,
                "10": 24000,"11": 27000,"12": 34000,"13": 34500,"14": 35000,
                "15": 37000,"16": 42000,"17": 50000,"18": 43000,"19": 35000,
                "20": 27000,"21": 22000,"22": 19000,"23": 14000,
            },
            "day_of_week": {
                "Monday": 382000, "Tuesday": 376000, "Wednesday": 380000,
                "Thursday": 382000, "Friday": 424000,
                "Saturday": 360000, "Sunday": 302000,
            },
        },
        "location_analysis": {
            "by_density": {
                "urban":    {"count": 1_900_158, "avg_casualties": 1.12},
                "suburban": {"count":   503_200,  "avg_casualties": 1.13},
                "rural":    {"count":   312_582,  "avg_casualties": 1.25},
                "unknown":  {"count":        0,   "avg_casualties": 0.0},
            },
        },
    }


def _demo_geo() -> pd.DataFrame:
    """
    Synthetic accident scatter over the UK for the map demo.
    Weighted toward real population centres (London, Manchester, Birmingham, Leeds).
    """
    rng = np.random.default_rng(42)
    centres = [
        (51.51, -0.12, 0.35),   # London
        (53.48, -2.24, 0.13),   # Manchester
        (52.48, -1.90, 0.11),   # Birmingham
        (53.80, -1.55, 0.10),   # Leeds
        (55.86, -4.26, 0.07),   # Glasgow
        (53.38, -1.47, 0.08),   # Sheffield
        (51.45, -2.58, 0.06),   # Bristol
        (54.97, -1.61, 0.05),   # Newcastle
        (50.72, -1.88, 0.05),   # Southampton
    ]
    rows = []
    n = 30_000
    for lat, lon, frac in centres:
        k = int(n * frac)
        rows.append(pd.DataFrame({
            "Latitude":  rng.normal(lat, 0.28, k),
            "Longitude": rng.normal(lon, 0.35, k),
        }))
    df = pd.concat(rows, ignore_index=True)
    sev = rng.choice(["Fatal", "Serious", "Slight"],
                     size=len(df), p=[0.01, 0.14, 0.85])
    df["Accident_Severity"] = sev
    df = df[(df.Latitude.between(49.5, 60.5)) & (df.Longitude.between(-8, 2))]
    return df.reset_index(drop=True)


def get_data() -> tuple[dict, pd.DataFrame, str]:
    report = load_eda_report()
    geo    = load_geo_sample()

    if report is not None:
        source = "Live data — reports/eda_summary.json"
    else:
        report = _demo_report()
        source = "Demo data — run src.data.ingest then src.eda to see live stats"

    if geo is None:
        geo = _demo_geo()

    return report, geo, source


# ═══════════════════════════════════════════════════════════════════════════════
# Chart builders
# ═══════════════════════════════════════════════════════════════════════════════

def fig_severity_donut(sev: dict) -> go.Figure:
    labels, values, colors = [], [], []
    for s in SEV_ORDER:
        if s in sev:
            labels.append(s)
            values.append(sev[s]["count"])
            colors.append(SEV_COLORS[s])

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.62,
        domain=dict(x=[0.0, 1.0], y=[0.20, 1.0]),
        marker_colors=colors,
        direction="clockwise",
        sort=False,
        textinfo="none",
        texttemplate="%{percent:.0%}",
        textposition="outside",
        automargin=True,
        textfont_size=13,
        hovertemplate="<b>%{label}</b><br>%{value:,} accidents<br>%{percent}<extra></extra>",
    ))
    total = sum(values)
    fig.add_annotation(
        text=f"<b>{total:,.0f}</b><br><span style='font-size:10px;color:{C_MID}'>TOTAL</span>",
        x=0.5, y=0.5, showarrow=False,
        font_size=16, align="center",
        xanchor="center", yanchor="middle",
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        uniformtext_minsize=12,
        uniformtext_mode="show",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.02,
            xanchor="center",
            x=0.5,
            font=dict(size=13, color=C_DARK),
            bgcolor="rgba(255,255,255,0.0)",
            itemwidth=70,
            itemsizing="constant",
        ),
        height=300,
        margin=dict(l=16, r=16, t=22, b=8),
    )
    return fig


def fig_hourly(hourly: dict) -> go.Figure:
    hours  = sorted([int(h) for h in hourly])
    counts = [hourly[str(h)] for h in hours]
    peak_h = hours[int(np.argmax(counts))]
    colors = [C_CORAL if h == peak_h else C_TEAL for h in hours]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hours, y=counts,
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="<b>%{x}:00</b> — %{y:,} accidents<extra></extra>",
    ))
    fig.add_annotation(
        x=peak_h, y=max(counts),
        text=f"Peak {peak_h}:00",
        showarrow=True, arrowhead=2, arrowcolor=C_CORAL,
        font=dict(color=C_CORAL, size=11, family="Inter"),
        ay=-30, ax=0,
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        xaxis=dict(title="", tickvals=list(range(0, 24, 3)),
                   showgrid=False, zeroline=False,
                   tickfont=dict(color="#2c3e50", size=11)),
        yaxis=dict(title="", gridcolor="#dde1e5", zeroline=False,
                   tickfont=dict(color="#2c3e50", size=11)),
        bargap=0.12,
        height=300,
        margin=dict(l=8, r=8, t=8, b=8),
    )
    return fig


def fig_day_of_week(dow: dict) -> go.Figure:
    days   = [d for d in DOW_ORDER if d in dow]
    counts = [dow[d] for d in days]
    colors = [C_CORAL if d == "Friday" else
              (C_ORANGE if d in ("Saturday", "Sunday") else C_TEAL)
              for d in days]
    labels = [d[:3] for d in days]

    fig = go.Figure(go.Bar(
        x=labels, y=counts,
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="<b>%{x}</b> — %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        xaxis=dict(showgrid=False, zeroline=False,
                   tickfont=dict(color="#2c3e50", size=11)),
        yaxis=dict(gridcolor="#dde1e5", zeroline=False,
                   tickfont=dict(color="#2c3e50", size=11)),
        bargap=0.28,
        height=300,
        margin=dict(l=8, r=8, t=8, b=8),
    )
    return fig


def fig_weather(weather: dict) -> go.Figure:
    rows = (
        pd.DataFrame([
            {"weather": k, "avg_casualties": v["avg_casualties"], "count": v["count"]}
            for k, v in weather.items()
            if k not in ("None", "nan", "Unknown", "Data missing or out of range")
        ])
        .sort_values("avg_casualties", ascending=True)
        .tail(8)
    )
    norm = (rows["avg_casualties"] - rows["avg_casualties"].min())
    norm = norm / norm.max() if norm.max() > 0 else norm
    # Teal→Coral gradient
    bar_colors = [
        f"rgba({int(27 + (231-27)*v)},{int(188 - (188-76)*v)},{int(156 - (156-60)*v)},0.85)"
        for v in norm
    ]
    fig = go.Figure(go.Bar(
        x=rows["avg_casualties"],
        y=rows["weather"],
        orientation="h",
        marker_color=bar_colors,
        marker_line_width=0,
        text=rows["avg_casualties"].round(2),
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>Avg casualties: %{x:.2f} (n=%{customdata:,})<extra></extra>",
        customdata=rows["count"],
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        xaxis=dict(title="Avg casualties", gridcolor="#dde1e5",
                   range=[1.0, rows["avg_casualties"].max() * 1.2], zeroline=False,
                   tickfont=dict(color="#2c3e50", size=11),
                   title_font=dict(color="#2c3e50", size=11)),
        yaxis=dict(title="", tickfont=dict(color="#2c3e50", size=10)),
        height=320,
        margin=dict(l=8, r=32, t=8, b=8),
    )
    return fig


def fig_location_density(loc: dict) -> go.Figure:
    by_density = loc.get("by_density", {})
    rows = pd.DataFrame([
        {"density": k.title(), "count": v["count"], "avg_casualties": v["avg_casualties"]}
        for k, v in by_density.items()
        if v["count"] > 0 and k != "unknown"
    ]).sort_values("count", ascending=False)

    pal = {"Urban": C_CORAL, "Suburban": C_ORANGE, "Rural": C_TEAL}

    fig = go.Figure(go.Bar(
        x=rows["density"],
        y=rows["count"],
        marker_color=[pal.get(d, C_MID) for d in rows["density"]],
        marker_line_width=0,
        text=[f"{v:,.0f}" for v in rows["count"]],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="<b>%{x}</b><br>%{y:,} accidents<br>Avg casualties: %{customdata:.2f}<extra></extra>",
        customdata=rows["avg_casualties"],
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        xaxis=dict(showgrid=False, zeroline=False,
                   tickfont=dict(color="#2c3e50", size=11)),
        yaxis=dict(gridcolor="#dde1e5", zeroline=False,
                   tickfont=dict(color="#2c3e50", size=11)),
        showlegend=False,
        bargap=0.38,
        height=320,
        margin=dict(l=8, r=8, t=8, b=8),
    )
    return fig


def fig_uk_map(geo: pd.DataFrame) -> go.Figure:
    sev_map = {
        "Fatal":   dict(color=C_CORAL,   size=5,  opacity=0.85),
        "Serious": dict(color=C_ORANGE,  size=3,  opacity=0.55),
        "Slight":  dict(color=C_TEAL,    size=2,  opacity=0.30),
    }
    fig = go.Figure()
    for sev in SEV_ORDER:
        sub = geo[geo["Accident_Severity"] == sev]
        if sub.empty:
            continue
        props = sev_map[sev]
        fig.add_trace(go.Scattergeo(
            lat=sub["Latitude"], lon=sub["Longitude"],
            mode="markers", name=sev,
            marker=dict(color=props["color"], size=props["size"],
                        opacity=props["opacity"], line_width=0),
            hoverinfo="skip",
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        geo=dict(
            scope="europe", resolution=50,
            lonaxis_range=[-8.5, 2.5], lataxis_range=[49.5, 61.0],
            showland=True, landcolor="#f5f5f0",
            showcoastlines=True, coastlinecolor="#c8c8c8",
            showframe=False,
            showocean=True, oceancolor="#ddeef8",
            showcountries=True, countrycolor="#d0d0d0",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color=C_DARK),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#d6dde3",
            borderwidth=1,
        ),
        height=320,
        margin=dict(l=0, r=0, t=8, b=8),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: KPI card HTML
# ═══════════════════════════════════════════════════════════════════════════════

def kpi_card(label: str, value: str, delta: str, delta_dir: str, accent: str) -> str:
    """
    Render a white KPI card with a coloured bottom accent strip.
    delta_dir: 'up' (bad — red), 'down' (good — teal), 'neu' (grey)
    """
    return f"""
    <div class="kpi-card" style="border-bottom-color:{accent};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-delta {delta_dir}">{delta}</div>
    </div>"""


def chart_card(title: str, fig, key: str) -> None:
    """Wrap a plotly chart in a white card with a small uppercase title."""
    st.markdown(
        f'<div class="chart-card"><p class="chart-title">{title}</p></div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False}, key=key)


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard layout
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    report, geo, source = get_data()

    sev   = report.get("severity_distribution", {})
    temp  = report.get("temporal_patterns", {})
    wx    = report.get("weather_analysis", {})
    loc   = report.get("location_analysis", {})
    shape = report.get("dataset_shape", {})

    # Pre-compute KPIs
    total        = shape.get("total_rows", sum(v["count"] for v in sev.values()))
    fatal_cnt    = sev.get("Fatal",   {}).get("count",      0)
    fatal_pct    = sev.get("Fatal",   {}).get("percentage", 0.0)
    serious_pct  = sev.get("Serious", {}).get("percentage", 0.0)
    hourly       = temp.get("hourly", {})
    peak_h       = max(hourly, key=lambda h: hourly[h], default="17") if hourly else "17"
    fog_cas      = wx.get("Fog or mist",       {}).get("avg_casualties", 0)
    clear_cas    = wx.get("Fine no high winds", {}).get("avg_casualties", 1)
    fog_uplift   = ((fog_cas / clear_cas) - 1) * 100 if clear_cas else 0

    # ── Top bar ──────────────────────────────────────────────────────────────
    is_demo = report.get("_source") == "demo"
    pill    = '<span class="demo-pill">DEMO DATA</span>' if is_demo else \
              '<span style="color:#1abc9c;font-size:0.78rem;font-weight:600;">LIVE DATA</span>'

    st.markdown(f"""
    <div class="topbar">
        <div>
            <p class="topbar-title">UK Road Accidents — Analytics Dashboard</p>
            <p class="topbar-sub">2.7 M STATS19 records &nbsp;·&nbsp; 2005–2017 &nbsp;·&nbsp; Severity prediction pipeline</p>
        </div>
        <div>{pill}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="page-body">', unsafe_allow_html=True)

    # ── Row 1: KPI cards ─────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Key Metrics</p>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(kpi_card(
        "Total Accidents", f"{total:,}",
        "2005–2017 UK STATS19", "neu", C_BLUE), unsafe_allow_html=True)
    k2.markdown(kpi_card(
        "Fatal Accidents", f"{fatal_cnt:,}",
        f"{fatal_pct:.1f}% of all accidents", "up", C_CORAL), unsafe_allow_html=True)
    k3.markdown(kpi_card(
        "Serious Accidents", f"{serious_pct:.0f}%",
        "of total accidents", "up", C_ORANGE), unsafe_allow_html=True)
    k4.markdown(kpi_card(
        "Peak Hour", f"{int(peak_h):02d}:00",
        "highest accident volume", "neu", C_TEAL), unsafe_allow_html=True)
    k5.markdown(kpi_card(
        "Fog vs Clear", f"+{fog_uplift:.0f}%",
        "more casualties in fog", "up", C_DARK), unsafe_allow_html=True)

    st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)

    # ── Row 2: Severity donut | Hourly bar | Day of week ─────────────────────
    st.markdown('<p class="section-label">Severity & Temporal Patterns</p>',
                unsafe_allow_html=True)
    c1, gap1, c2, gap2, c3 = st.columns([1, 0.04, 1.6, 0.04, 1])

    with c1:
        st.markdown('<div class="chart-card"><p class="chart-title">Severity Split</p>', unsafe_allow_html=True)
        st.plotly_chart(fig_severity_donut(sev), use_container_width=True,
                        config={"displayModeBar": False}, key="donut")
        st.markdown('</div>', unsafe_allow_html=True)

    with gap1:
        st.empty()

    with c2:
        st.markdown('<div class="chart-card"><p class="chart-title">Accidents by Hour of Day</p>', unsafe_allow_html=True)
        st.plotly_chart(fig_hourly(hourly), use_container_width=True,
                        config={"displayModeBar": False}, key="hourly")
        st.markdown('</div>', unsafe_allow_html=True)

    with gap2:
        st.empty()

    with c3:
        st.markdown('<div class="chart-card"><p class="chart-title">Day of Week</p>', unsafe_allow_html=True)
        dow = temp.get("day_of_week", {})
        st.plotly_chart(fig_day_of_week(dow), use_container_width=True,
                        config={"displayModeBar": False}, key="dow")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:0.2rem'></div>", unsafe_allow_html=True)

    # ── Row 3: Weather | Location density | Map ───────────────────────────────
    st.markdown('<p class="section-label">Risk Factors & Geography</p>',
                unsafe_allow_html=True)
    c4, gap3, c5, gap4, c6 = st.columns([1.5, 0.04, 0.85, 0.04, 1.4])

    with c4:
        st.markdown('<div class="chart-card"><p class="chart-title">Avg Casualties by Weather Condition</p>', unsafe_allow_html=True)
        if wx:
            st.plotly_chart(fig_weather(wx), use_container_width=True,
                            config={"displayModeBar": False}, key="weather")
        else:
            st.info("Run the EDA pipeline for weather data.")
        st.markdown('</div>', unsafe_allow_html=True)

    with gap3:
        st.empty()

    with c5:
        st.markdown('<div class="chart-card"><p class="chart-title">Location Density</p>', unsafe_allow_html=True)
        if loc.get("by_density"):
            st.plotly_chart(fig_location_density(loc), use_container_width=True,
                            config={"displayModeBar": False}, key="density")
        else:
            st.info("Location density data not available.")
        st.markdown('</div>', unsafe_allow_html=True)

    with gap4:
        st.empty()

    with c6:
        st.markdown('<div class="chart-card"><p class="chart-title">Accident Hotspot Map — UK</p>', unsafe_allow_html=True)
        st.plotly_chart(fig_uk_map(geo), use_container_width=True,
                        config={"displayModeBar": False}, key="map")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="dash-footer">
        Data: {source} &nbsp;·&nbsp;
        UK DfT STATS19 road accidents 2005–2017 &nbsp;·&nbsp;
        Map: {len(geo):,} sampled locations
    </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close page-body


if __name__ == "__main__":
    main()
