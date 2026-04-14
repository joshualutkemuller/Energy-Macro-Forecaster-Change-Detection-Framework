# grp132 — Oil Market Data Pipeline & EDA

**CSE 6242 Data & Visual Analytics · Spring 2026 · Team 132**

> Tracking Oil Price Impacts on Drilling Activity, Production, and Economic Trends

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Use Cases](#2-use-cases)
3. [Data Sources](#3-data-sources)
4. [Repository Structure](#4-repository-structure)
5. [Assumptions & Design Decisions](#5-assumptions--design-decisions)
6. [Setup & Installation](#6-setup--installation)
7. [Configuration](#7-configuration)
8. [Running the Pipeline](#8-running-the-pipeline)
9. [Outputs](#9-outputs)
10. [EDA Figures Reference](#10-eda-figures-reference)
11. [Pipeline Architecture](#11-pipeline-architecture)
12. [Extending the Pipeline](#12-extending-the-pipeline)
13. [Known Limitations](#13-known-limitations)
14. [Team](#14-team)

---

## 1. Motivation

Oil prices do not immediately translate into changes in drilling activity or production. Companies must weigh the cost of drilling a new well against expected future revenues, which means that a sustained price increase typically precedes an uptick in rig counts by **2–6 months**, and production responses can lag even longer. Understanding this dynamic is critical for:

- **E&P companies** deciding when to initiate new drilling programs
- **Investors and banks** sizing positions or credit exposure in energy stocks
- **State policymakers** whose tax revenue is directly tied to drilling and production activity
- **Researchers** studying commodity price transmission and macroeconomic spillovers

Existing analyses rely on static charts or specialized VAR models published in technical papers. This project builds a **reproducible, frequency-aware data pipeline** that assembles all relevant data series in one place, cleans and aligns them, and produces a rich EDA layer (summary statistics, cross-correlations, lagged regressions, Granger causality tests, and 10 standardized figures) that feeds directly into the downstream interactive dashboard.

---

## 2. Use Cases

| Who | What they get from this pipeline |
|-----|----------------------------------|
| Dashboard developer | Three clean master CSVs (daily / weekly / monthly) ready to load into Dash/Plotly/Tableau |
| Data analyst | Full EDA suite — lag plots, CCF bars, heatmaps, R² curves — reproducible with one command |
| Researcher | Granger causality and lagged OLS tables saved as CSVs for further modeling |
| Teammate onboarding | `--use-cache` flag lets anyone re-run EDA without re-downloading data |

---

## 3. Data Sources

### 3.1 EIA API v2 — U.S. Energy Information Administration
| Series | Frequency | Description |
|--------|-----------|-------------|
| `RWTC` | Weekly | WTI Cushing Spot Price ($/bbl) |
| U.S. Crude Production | Monthly | Field-level crude oil production (thousand barrels/day) |

- **API registration (free):** https://www.eia.gov/opendata/register.php
- **API docs:** https://www.eia.gov/opendata/documentation.php
- Fallback: if EIA fails or key is missing, WTI is sourced from FRED `DCOILWTICO` and resampled to weekly.

### 3.2 Baker Hughes — North America Rotary Rig Count
| Series | Frequency | Description |
|--------|-----------|-------------|
| NA Rig Count | Weekly | Total active rotary rigs in North America |

- **Download page:** https://rigcount.bakerhughes.com/na-rig-count
- The pipeline attempts an automatic download from two known Baker Hughes URLs and caches the result locally. If the download fails (URL changes), the pipeline prints step-by-step instructions for a manual download and falls back to the cached file.

### 3.3 FRED API — Federal Reserve Bank of St. Louis
| Variable | Series ID | Frequency | Description |
|----------|-----------|-----------|-------------|
| WTI Price (daily) | `DCOILWTICO` | Daily | Crude Oil Prices: WTI ($/bbl) |
| 10Y–2Y Treasury Spread | `T10Y2Y` | Daily | Yield curve spread (pp) |
| USD Trade-Weighted Index | `DTWEXBGS` | Daily | Broad dollar index |
| Federal Funds Rate | `FEDFUNDS` | Monthly | Effective Fed Funds Rate (%) |
| Unemployment Rate | `UNRATE` | Monthly | Civilian unemployment (%) |
| Industrial Production | `INDPRO` | Monthly | Industrial production index |
| CPI | `CPIAUCSL` | Monthly | All-urban CPI (index) |
| Henry Hub Gas Price | `MHHNGSP` | Monthly | Natural gas spot price ($/MMBtu) |
| Recession Indicator | `USREC` | Monthly | NBER recession dummy (0/1) |
| Real GDP | `GDPC1` | Quarterly | Chained 2017 dollars (billions) |

- **API registration (free):** https://fred.stlouisfed.org/docs/api/api_key.html
- `fredapi` Python package is used (already available in the project environment).

---

## 4. Repository Structure

```
grp132_datawrangler/
│
├── config.py               # Central configuration: API keys, date range,
│                           # series IDs, file paths, resampling settings
│
├── requirements.txt        # Python package dependencies
├── run_pipeline.py         # Main entry point (CLI with flags)
│
├── data/
│   ├── raw/
│   │   ├── eia/            # Cached raw EIA CSVs
│   │   ├── baker_hughes/   # Cached BH Excel + parsed CSV
│   │   └── fred/           # Cached raw FRED CSVs
│   └── processed/
│       ├── master_daily.csv    # Business-day aligned master
│       ├── master_weekly.csv   # Friday week-end aligned master
│       └── master_monthly.csv  # Month-start aligned master
│
├── outputs/
│   ├── figures/            # 10 EDA PNG figures (150 DPI)
│   ├── summary_*.csv       # Extended descriptive stats per master
│   ├── corr_pearson_*.csv  # Correlation matrices
│   ├── corr_spearman_*.csv
│   ├── ccf_wti_vs_*.csv    # Cross-correlation tables
│   ├── lagged_reg_*.csv    # Lagged OLS regression tables
│   └── granger_*.csv       # Granger causality test results
│
└── src/
    ├── __init__.py
    ├── acquire/
    │   ├── __init__.py
    │   ├── eia.py          # EIA API v2 fetcher (WTI + production)
    │   ├── baker_hughes.py # BH rig count downloader + parser
    │   └── fred.py         # FRED fetcher (daily / monthly / quarterly)
    │
    ├── pipeline/
    │   ├── __init__.py
    │   ├── clean.py        # Per-series cleaning (dedup, coerce, winsorise)
    │   └── merge.py        # Frequency-aware merge → three master datasets
    │
    └── eda/
        ├── __init__.py
        ├── summary_stats.py   # Extended descriptive statistics
        ├── correlations.py    # CCF, lagged OLS, Granger causality
        └── plots.py           # All 10 EDA figures
```

---

## 5. Assumptions & Design Decisions

### Frequency Alignment Strategy
Each series has a **native frequency** (daily, weekly, monthly, or quarterly). Rather than disaggregate coarse series into finer ones (which would fabricate precision), the pipeline uses the following rules:

| Direction | Method |
|-----------|--------|
| High → Low (e.g., weekly WTI → monthly) | **Mean aggregation** — takes the average of all observations in the period |
| Low → High (e.g., monthly FRED → weekly master) | **Forward-fill** — the last known value is carried forward up to `MAX_GAP_PERIODS` periods |
| Quarterly → Monthly | Forward-fill with a limit of 3 months |

This means the weekly and monthly masters contain the most recently available monthly/quarterly observation, not an interpolated estimate.

#### How each series gets into `master_weekly`

| Series | Native freq | How it gets to weekly |
|--------|-------------|----------------------|
| `wti_price_weekly` | Weekly (EIA) | Resampled to `W-FRI` mean |
| `rig_count` | Weekly (BH) | Resampled to `W-FRI` mean |
| `fed_funds`, `unemployment`, `indpro`, `cpi`, `ppi`, `ng_price`, `recession` | Monthly (FRED) | Forward-filled from month-start up to 5 weeks |
| `sp500` | Daily (Stooq) | Resampled to `W-FRI` last close; `sp500_ret_w` = week-over-week `pct_change` |

#### How each series gets into `master_monthly`

| Series | Native freq | How it gets to monthly |
|--------|-------------|------------------------|
| `wti_price_weekly` | Weekly (EIA) | Resampled to month-start mean |
| `rig_count` | Weekly (BH) | Resampled to month-start mean |
| `us_production_mbbld` | Monthly (EIA) | Native monthly; index normalised to month-start |
| `fed_funds`, `unemployment`, `indpro`, `cpi`, `ppi`, `ng_price`, `recession` | Monthly (FRED) | Native monthly; index normalised to month-start |
| `sp500` | Daily (FRED) | Resampled to month-start last close; `sp500_ret_m` = month-over-month `pct_change` |

### WTI Price: Two Sources, One Column
The EIA API provides weekly WTI spot prices (`wti_price_weekly`). FRED provides a daily series (`wti_price_d`). Both are kept:
- `master_weekly` uses `wti_price_weekly` (EIA native weekly)
- `master_daily` uses `wti_price_d` (FRED native daily)
- `master_monthly` aggregates the EIA weekly series to monthly mean

If the EIA API key is unavailable, `wti_price_weekly` is derived by resampling the FRED daily series to Friday week-ends — clearly logged as a fallback.

### Baker Hughes Rig Count
Baker Hughes does not offer a formal REST API. The pipeline fetches the Excel workbook from two known direct-download URLs. If both fail (URLs are subject to change), it falls back to the last cached file and prints manual download instructions. The parsed result is always cached to `data/raw/baker_hughes/bh_rig_count.xlsx` so subsequent runs work offline.

### Date Range
Default start date is **2000-01-01**, capturing:
- The post-1998 oil price recovery
- The 2001 recession
- The 2003–2008 super-cycle
- The 2008–2009 financial crisis crash and recovery
- The 2014–2016 shale-driven price collapse
- The 2020 COVID demand shock
- The 2021–2022 post-COVID price spike

The end date defaults to **today** (pulled live). Both can be overridden via `--start` / `--end` CLI flags or in `config.py`.

### Outlier Treatment
Winsorisation is available (4-sigma rule) but **disabled by default**. Oil prices can legitimately spike (e.g., April 2020 WTI went negative; March 2022 saw extreme highs). Automatic clipping would distort exactly the structural breaks the downstream analysis is designed to detect. Enable with `winsorise=True` in `clean.py` if preprocessing for a specific model requires it.

### Missing Data
- Gaps up to `MAX_GAP_PERIODS = 4` periods are forward-filled in the master datasets.
- Columns with more than **80% missing** values are dropped automatically with a warning.
- Missing percentages per column are logged at INFO level (>5% missing) or DEBUG level (<5%).

### Index Conventions
| Master | Index frequency | Anchor |
|--------|-----------------|--------|
| `master_daily` | Business days (`B`) | Calendar |
| `master_weekly` | Weekly (`W-FRI`) | Friday |
| `master_monthly` | Month-start (`MS`) | 1st of month |

---

## 6. Setup & Installation

### Prerequisites
- Python 3.10+ (tested on 3.13)
- `pip` or `conda`

### Install dependencies

```bash
# From the repo root
pip install -r requirements.txt
```

Core packages used:

| Package | Purpose |
|---------|---------|
| `pandas` | DataFrames, resampling, merging |
| `numpy` | Numerical operations |
| `requests` | HTTP calls to EIA and Baker Hughes |
| `fredapi` | FRED API client |
| `matplotlib` | Figure rendering |
| `seaborn` | Statistical plot styling |
| `scipy` | Q-Q plots, probability distributions |
| `statsmodels` | OLS regression, Granger causality |
| `openpyxl` / `xlrd` | Baker Hughes Excel parsing |

---

## 7. Configuration

All settings live in **`config.py`**. The two required changes before first run are the API keys.

### API Keys

Set them directly in `config.py`:
```python
EIA_API_KEY  = "your_eia_key_here"
FRED_API_KEY = "your_fred_key_here"
```

Or set them as environment variables (recommended for shared repos):
```bash
export EIA_API_KEY="your_eia_key_here"
export FRED_API_KEY="your_fred_key_here"
python run_pipeline.py
```

Or use a `.env` file with `python-dotenv`:
```bash
# .env
EIA_API_KEY=your_eia_key_here
FRED_API_KEY=your_fred_key_here
```
```python
# add to the top of run_pipeline.py
from dotenv import load_dotenv; load_dotenv()
```

### Changing the Date Range

```python
# config.py
START_DATE = "1990-01-01"   # go back further
END_DATE   = "2024-12-31"   # fix an end date
```

### Adding / Removing FRED Series

To add a new monthly series, append to the dict in `config.py`:
```python
FRED_MONTHLY = {
    ...
    "vix": "VIXCLS",   # CBOE Volatility Index (daily → resample if needed)
}
```
The series will automatically appear in the monthly master and all EDA outputs on the next run.

---

## 8. Running the Pipeline

```bash
cd "path/to/grp132_datawrangler"
python run_pipeline.py [OPTIONS]
```

### Modes

| Command | What it does |
|---------|-------------|
| `python run_pipeline.py` | Full run: download → clean → merge → EDA |
| `python run_pipeline.py --use-cache` | Skip live downloads; use CSVs in `data/raw/` |
| `python run_pipeline.py --eda-only` | Skip everything; re-run EDA on existing master CSVs |
| `python run_pipeline.py --no-eda` | Acquire, clean, and merge only; skip EDA |
| `python run_pipeline.py --start 2010-01-01 --end 2023-12-31` | Custom date range |

### What a full run looks like

```
09:12:01  INFO      grp132 Data Pipeline — 2026-03-14 09:12
09:12:01  INFO      Start: 2000-01-01  |  End: today
09:12:01  INFO      ============================================================
09:12:01  INFO      STEP 1 — Data Acquisition
09:12:01  INFO      ============================================================
09:12:01  INFO      EIA: fetching WTI weekly spot price ...
09:12:03  INFO        → 1,304 weekly WTI records (2000-01-07 – 2026-03-07)
09:12:03  INFO      EIA: fetching U.S. crude production (monthly) ...
09:12:05  INFO        → 313 monthly production records (2000-01-01 – 2026-01-01)
09:12:05  INFO      Baker Hughes: fetching NA rig count ...
09:12:08  INFO        BH download succeeded
09:12:08  INFO        → 1,356 weekly rig-count records (1999-12-31 – 2026-03-07)
09:12:08  INFO      FRED: fetching daily series ...
09:12:12  INFO        → 6,843 daily records (2000-01-03 – 2026-03-13)
09:12:12  INFO      FRED: fetching monthly series ...
09:12:16  INFO        → 314 monthly records (2000-01-01 – 2026-02-01)
09:12:16  INFO      FRED: fetching quarterly series ...
09:12:17  INFO        → 105 quarterly records
...
09:12:45  INFO      Pipeline complete in 44s.

============================================================
  MASTER DATASETS
============================================================

  master_daily
    rows : 6,843
    cols : ['wti_price_d', 't10y2y', 'dxy']
    range: 2000-01-03 → 2026-03-13

  master_weekly
    rows : 1,356
    cols : ['wti_price_weekly', 'rig_count', 'fed_funds', 'unemployment', ...]
    range: 2000-01-07 → 2026-03-07

  master_monthly
    rows : 314
    cols : ['wti_price_weekly', 'rig_count', 'us_production_mbbld', ...]
    range: 2000-01-01 → 2026-02-01
```

Progress and warnings are also written to `pipeline.log` in the repo root.

---

## 9. Outputs

### Master Datasets (`data/processed/`)

#### `master_daily.csv` — business-day index (`B`), ~6,800 rows

| Column | What it represents | Units | Source | Calculation |
|--------|--------------------|-------|--------|-------------|
| `wti_price_d` | WTI crude oil spot price — the benchmark U.S. oil price | USD per barrel | FRED `DCOILWTICO` | Raw daily value; forward-filled up to 5 business days to cover weekends and holidays |
| `t10y2y` | Yield curve spread — 10-year Treasury yield minus 2-year Treasury yield. Negative values signal an inverted yield curve, often a recession predictor | Percentage points | FRED `T10Y2Y` | Raw daily value; forward-filled up to 5 days |
| `dxy` | U.S. dollar strength index — measures the USD against a basket of major trading partner currencies. Higher = stronger dollar | Index (2006=100) | FRED `DTWEXBGS` | Raw daily value; forward-filled up to 5 days |
| `sp500` | S&P 500 index closing level — broad U.S. equity market benchmark | Index level | Stooq `^SPX` | Raw daily close; forward-filled up to 5 business days |
| `sp500_ret_d` | S&P 500 daily percent return | % | Derived | `sp500.pct_change() * 100` |

#### `master_weekly.csv` — Friday week-end index (`W-FRI`), ~1,350 rows

| Column | What it represents | Units | Source | Calculation |
|--------|--------------------|-------|--------|-------------|
| `wti_price_weekly` | WTI crude oil spot price — weekly average | USD per barrel | EIA `RWTC` | Resampled to `W-FRI` mean (normalises any non-Friday EIA observation dates) |
| `rig_count` | North America active rotary rig count — a leading indicator of future drilling activity and production | Number of rigs | Baker Hughes NAM Weekly sheet | Read directly from the NAM Weekly sheet; resampled to `W-FRI` mean to align anchor day |
| `fed_funds` | Effective Federal Funds Rate — the overnight interest rate banks charge each other, set by the Federal Reserve. Reflects monetary policy tightness | % per annum | FRED `FEDFUNDS` | Native monthly value; forward-filled from month-start up to 5 weeks |
| `unemployment` | U.S. civilian unemployment rate — share of the labor force that is jobless and seeking work | % | FRED `UNRATE` | Native monthly value; forward-filled up to 5 weeks |
| `indpro` | Industrial Production Index — measures real output of U.S. manufacturing, mining, and electric/gas utilities sectors. A proxy for industrial demand for energy | Index (2017=100) | FRED `INDPRO` | Native monthly value; forward-filled up to 5 weeks |
| `cpi` | Consumer Price Index (All Urban Consumers) — measures overall price inflation across a broad basket of consumer goods and services | Index (1982–84=100) | FRED `CPIAUCSL` | Native monthly value; forward-filled up to 5 weeks |
| `ppi` | Producer Price Index (All Commodities) — measures price changes received by domestic producers for their output. A leading indicator of consumer inflation | Index (1982=100) | FRED `PPIACO` | Native monthly value; forward-filled up to 5 weeks |
| `ng_price` | Henry Hub natural gas spot price — the U.S. benchmark natural gas price at the Henry Hub pipeline interchange in Louisiana | USD per MMBtu | FRED `MHHNGSP` | Native monthly value; forward-filled up to 5 weeks |
| `recession` | NBER recession indicator — binary flag marking official U.S. recession periods as determined by the National Bureau of Economic Research | 0 = expansion, 1 = recession | FRED `USREC` | Native monthly value; forward-filled up to 5 weeks |
| `sp500` | S&P 500 Friday closing level | Index level | Stooq `^SPX` | Daily closes resampled to `W-FRI` last — the Friday closing level |
| `sp500_ret_w` | S&P 500 week-over-week percent return | % | Derived | `sp500_weekly.pct_change() * 100` |

#### `master_monthly.csv` — month-start index (`MS`), ~310 rows

| Column | What it represents | Units | Source | Calculation |
|--------|--------------------|-------|--------|-------------|
| `wti_price_weekly` | WTI crude oil spot price — monthly average | USD per barrel | EIA `RWTC` | Weekly EIA prices aggregated to month-start by **mean** across all weeks in the month |
| `rig_count` | Total North America active rotary rig count | Number of rigs | Baker Hughes NAM Monthly sheet | Sum of all `Rig Count Value` rows for each Year+Month across all countries, basins, and trajectories |
| `rig_country_canada` | Canada rig count | Number of rigs | Baker Hughes NAM Monthly sheet | Sum of `Rig Count Value` where Country = CANADA, per month |
| `rig_country_united_states` | United States rig count | Number of rigs | Baker Hughes NAM Monthly sheet | Sum of `Rig Count Value` where Country = UNITED STATES, per month |
| `rig_basin_{name}` *(15 columns)* | Rig count for a specific named basin | Number of rigs | Baker Hughes NAM Monthly sheet | Sum per month for each basin: Permian, Eagle Ford, Marcellus, Haynesville, DJ-Niobrara, Williston, Granite Wash, Barnett, Cana Woodford, Utica, Ardmore Woodford, Arkoma Woodford, Fayetteville, Mississippian, Other |
| `rig_drillfor_oil` | Rigs drilling for oil | Number of rigs | Baker Hughes NAM Monthly sheet | Sum of `Rig Count Value` where DrillFor = Oil, per month |
| `rig_drillfor_gas` | Rigs drilling for gas | Number of rigs | Baker Hughes NAM Monthly sheet | Sum of `Rig Count Value` where DrillFor = Gas, per month |
| `rig_drillfor_miscellaneous` | Rigs drilling for miscellaneous targets | Number of rigs | Baker Hughes NAM Monthly sheet | Sum of `Rig Count Value` where DrillFor = Miscellaneous, per month |
| `rig_state_{name}` *(39 columns)* | Rig count for a specific U.S. state or Canadian province | Number of rigs | Baker Hughes NAM Monthly sheet | Sum per month for each of 39 states/provinces (e.g. `rig_state_texas`, `rig_state_alberta`, `rig_state_north_dakota`) |
| `rig_traj_horizontal` | Horizontal well rig count | Number of rigs | Baker Hughes NAM Monthly sheet | Sum of `Rig Count Value` where Trajectory = Horizontal, per month |
| `rig_traj_vertical` | Vertical well rig count | Number of rigs | Baker Hughes NAM Monthly sheet | Sum of `Rig Count Value` where Trajectory = Vertical, per month |
| `rig_traj_directional` | Directional well rig count | Number of rigs | Baker Hughes NAM Monthly sheet | Sum of `Rig Count Value` where Trajectory = Directional, per month |
| `rig_traj_other` | Other trajectory rig count | Number of rigs | Baker Hughes NAM Monthly sheet | Sum of `Rig Count Value` where Trajectory = Other, per month |
| `us_production_mbbld` | U.S. crude oil field production — how many thousand barrels per day the U.S. is producing. A key supply-side indicator | Thousand barrels per day (Mbbld) | EIA production API | Native monthly value; index normalised to month-start |
| `fed_funds` | Effective Federal Funds Rate — the overnight interest rate banks charge each other, set by the Federal Reserve | % per annum | FRED `FEDFUNDS` | Native monthly value; index normalised to month-start |
| `unemployment` | U.S. civilian unemployment rate | % | FRED `UNRATE` | Native monthly value; index normalised to month-start |
| `indpro` | Industrial Production Index — measures real output of U.S. manufacturing, mining, and electric/gas utilities sectors. A proxy for industrial demand for energy | Index (2017=100) | FRED `INDPRO` | Native monthly value; index normalised to month-start |
| `cpi` | Consumer Price Index (All Urban Consumers) — overall consumer price inflation | Index (1982–84=100) | FRED `CPIAUCSL` | Native monthly value; index normalised to month-start |
| `ppi` | Producer Price Index (All Commodities) — price changes received by domestic producers | Index (1982=100) | FRED `PPIACO` | Native monthly value; index normalised to month-start |
| `ng_price` | Henry Hub natural gas spot price | USD per MMBtu | FRED `MHHNGSP` | Native monthly value; index normalised to month-start |
| `recession` | NBER recession indicator — binary flag for official U.S. recession periods | 0 = expansion, 1 = recession | FRED `USREC` | Native monthly value; index normalised to month-start |
| `sp500` | S&P 500 end-of-month closing level | Index level | Stooq `^SPX` | Daily closes resampled to month-start **last** — the final trading day close of each month |
| `sp500_ret_m` | S&P 500 month-over-month percent return | % | Derived | `sp500_monthly.pct_change() * 100` |
| `wti_mom_pct` | WTI price month-over-month percent change | % | Derived | `wti_price_weekly.pct_change() * 100` |
| `cpi_mom_pct` | CPI month-over-month percent change — monthly inflation rate | % | Derived | `cpi.pct_change() * 100` |
| `ppi_mom_pct` | PPI month-over-month percent change — monthly producer inflation rate | % | Derived | `ppi.pct_change() * 100` |

All CSVs use `date` as the index column (ISO 8601 format, UTC-naive).

### Analytical Tables (`outputs/`)

| File | Description |
|------|-------------|
| `summary_master_monthly.csv` | Count, mean, std, min, p25, median, p75, max, skew, kurtosis, CV% |
| `corr_pearson_monthly.csv` | Full Pearson correlation matrix |
| `corr_spearman_monthly.csv` | Full Spearman correlation matrix |
| `ccf_wti_vs_rig_count.csv` | Cross-correlation at lags −18…+18 months with ±95% CI |
| `ccf_wti_vs_us_production_mbbld.csv` | Same for production |
| `lagged_reg_wti_vs_rig_count.csv` | OLS β, SE, t, p, R² at each lag 0–18 months |
| `lagged_reg_wti_vs_us_production_mbbld.csv` | Same for production |
| `granger_wti_vs_rig_count.csv` | Granger F-stat and p-value at each lag 1–12 months |
| `granger_wti_vs_us_production_mbbld.csv` | Same for production |

---

## 10. EDA Figures Reference

All figures are saved to `outputs/figures/` as 150 DPI PNGs.

| # | Filename | What it shows |
|---|----------|---------------|
| 01 | `01_time_series_overview.png` | Three-panel stacked time series: WTI price, rig count, U.S. production. NBER recessions shaded in peach. |
| 02 | `02_wti_vs_rig_count.png` | Dual-axis time series (WTI left, rig count right) + scatter plot with OLS trend line. |
| 03 | `03_wti_vs_production.png` | Same layout for WTI vs. U.S. crude production. |
| 04 | `04_wti_vs_macro.png` | Grid of scatter plots: WTI vs. each macro variable (Fed Funds, unemployment, industrial production, CPI, gas price, GDP, yield spread, dollar index). Points colored by WTI level; Pearson r annotated. |
| 05 | `05_correlation_heatmap.png` | Lower-triangle Pearson correlation heatmap (all monthly variables), annotated with r values. |
| 06 | `06_lag_plots_rig_count.png` | 4×2 grid of scatter plots: WTI(t−k) vs. rig_count(t) at k = 0, 1, 2, 3, 6, 9, 12, 18 months. Correlation annotated per panel. |
| 06b | `06_lag_plots_us_production_mbbld.png` | Same grid for WTI vs. production. |
| 07 | `07_ccf_wti_vs_rig_count.png` | Bar chart of cross-correlation at lags −18…+18 months. Bars above zero = positive correlation. Dashed lines = ±95% CI. |
| 07b | `07_ccf_wti_vs_us_production_mbbld.png` | Same for production. |
| 08 | `08_lagged_reg_rig_count.png` | Two panels: (left) R² by lag curve with best-lag marker; (right) OLS β ± 95% CI by lag with significant lags (p<0.05) highlighted in red. |
| 08b | `08_lagged_reg_us_production_mbbld.png` | Same for production. |
| 09 | `09_recession_overlays.png` | Three-panel time series with NBER recession shading on all panels. |
| 10 | `10_returns_distribution.png` | Histogram + KDE of WTI monthly returns (left); Normal Q-Q plot (right). Annotated with mean, σ, skewness, kurtosis. |

---

## 11. Pipeline Architecture

```
                        ┌─────────────────────────────┐
                        │     run_pipeline.py          │
                        │  (orchestrator / CLI)        │
                        └──────────┬──────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
      src/acquire/eia.py   src/acquire/            src/acquire/
      (EIA API v2)         baker_hughes.py          fred.py
      • WTI weekly         (Excel download          (fredapi)
      • US production      + parser)                • daily
        monthly            • NA rig count           • monthly
                             weekly                 • quarterly
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   ▼
                        src/pipeline/clean.py
                        • Validate DatetimeIndex
                        • Numeric coerce
                        • Dedup + sort
                        • Report missing
                        • Optional winsorise
                                   │
                                   ▼
                        src/pipeline/merge.py
                        ┌──────────────────────────┐
                        │ build_master_daily        │ ← FRED daily only
                        │ build_master_weekly       │ ← EIA + BH + FRED (ffill)
                        │ build_master_monthly      │ ← All sources aggregated
                        └──────────┬───────────────┘
                                   │ saves to data/processed/
                                   ▼
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
  src/eda/summary_stats   src/eda/correlations   src/eda/plots
  • Extended desc. stats  • Pearson/Spearman     • 10 PNG figures
  • Per-master CSVs       • CCF at ±18 lags      • Saved to
                          • Lagged OLS             outputs/figures/
                          • Granger causality
                          • CSVs to outputs/
```

---

## 12. Extending the Pipeline

### Add a new FRED series
1. Add an entry to the appropriate dict in `config.py`:
   ```python
   FRED_MONTHLY["vix_monthly"] = "VIXCLS"   # will be auto-resampled
   ```
2. Re-run: `python run_pipeline.py`

### Add a new data source
1. Create `src/acquire/my_source.py` following the pattern of `eia.py` — return a `pd.DataFrame` with a `DatetimeIndex`.
2. Call it inside the `acquire()` function in `run_pipeline.py` and add it to the `raw` dict.
3. Incorporate it into the relevant `build_master_*` function in `src/pipeline/merge.py`.

### Change the cross-correlation lag window
```python
# src/eda/correlations.py
MAX_LAG_MONTHS = 24   # extend to 2 years
```

### Run EDA on a different frequency master
```python
from src.pipeline.merge import load_masters
from src.eda.plots import plot_correlation_heatmap

masters = load_masters()
plot_correlation_heatmap(masters["master_weekly"])
```

---

## 13. Known Limitations

| Limitation | Detail |
|------------|--------|
| Baker Hughes URL instability | The Excel download URL is not part of a stable public API and may change. If the auto-download fails, follow the manual instructions printed to the console. |
| EIA API rate limits | The EIA v2 API allows up to 5,000 rows per request. The pipeline paginates automatically, but requests are throttled to ~5 req/sec. Very large date ranges may be slow. |
| Quarterly GDP not in weekly/monthly masters | Real GDP (`real_gdp`) is a quarterly series and is intentionally excluded from `master_weekly` and `master_monthly` to avoid implying finer precision than the data supports. |
| WTI negative prices (April 2020) | The April 2020 WTI futures negative print is retained in the raw data (as it should be for structural break analysis) but may look anomalous in scatter plots. |
| Baker Hughes reports North America, not U.S.-only | The BH rig count covers Canada + U.S. For a U.S.-only count, filter to the appropriate BH sheet tab (the parser selects the first North America-labelled sheet by default). |
| No geographic disaggregation yet | The EIA production series is U.S.-level. State-level production data from EIA API is available and can be added in a future sprint. |

---

## 14. Team

**Team 132 — CSE 6242 Spring 2026**

| Name | Role |
|------|------|
| Hossein Ariannejad | |
| Ryan Burinescu | |
| Joshua Kahle | |
| Josha Lutkemullmer | |
| Michael Shepherd | Contact Person |
| Ena Tabakovic | |

---

*This pipeline is the data foundation for the interactive dashboard described in the team proposal. Weeks 1–2 deliverable.*
