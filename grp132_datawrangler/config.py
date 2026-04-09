"""
Central configuration for the grp132 data pipeline.
API keys are read from keys/eia_key.txt and keys/fred_key.txt,
or overridden via environment variables EIA_API_KEY / FRED_API_KEY.
"""
import os
import pathlib

_KEYS_DIR = pathlib.Path(__file__).parent / "keys"


def _read_key(filename: str, env_var: str) -> str:
    """Return env var if set, else read from keys/<filename>."""
    if env_var in os.environ:
        return os.environ[env_var]
    key_file = _KEYS_DIR / filename
    if key_file.exists():
        return key_file.read_text().strip()
    return ""


# ── API Keys ──────────────────────────────────────────────────────────────────
# EIA (Energy Information Administration)
# Register at: https://www.eia.gov/opendata/register.php
EIA_API_KEY = _read_key("eia_key.txt", "EIA_API_KEY")

# FRED (Federal Reserve Bank of St. Louis)
# Register at: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = _read_key("fred_key.txt", "FRED_API_KEY")

# ── Date Range ────────────────────────────────────────────────────────────────
START_DATE = "2000-01-01"
END_DATE   = None          # None = today

# ── EIA Series ────────────────────────────────────────────────────────────────
# Petroleum spot prices endpoint
EIA_SPOT_PRICE_SERIES = "RWTC"          # WTI weekly spot price ($/bbl)

# U.S. crude oil field production (monthly, thousand barrels/day)
EIA_PRODUCTION_DUOAREA = "NUS"          # National US
EIA_PRODUCTION_PRODUCT = "EPC0"         # Crude oil

# ── FRED Series ───────────────────────────────────────────────────────────────
# Daily
FRED_DAILY = {
    "wti_price_d":   "DCOILWTICO",   # WTI spot price (USD/bbl)
    "t10y2y":        "T10Y2Y",       # 10Y–2Y Treasury spread (bp)
    "dxy":           "DTWEXBGS",     # USD trade-weighted index
    "sp500":         "SP500",        # S&P 500 index level (daily close)
}

# Weekly (or will be resampled from daily)
FRED_WEEKLY = {}

# Monthly
FRED_MONTHLY = {
    "fed_funds":     "FEDFUNDS",     # Effective Federal Funds Rate (%)
    "unemployment":  "UNRATE",       # Unemployment Rate (%)
    "indpro":        "INDPRO",       # Industrial Production Index
    "cpi":           "CPIAUCSL",     # CPI All Urban Consumers (index)
    "ppi":           "PPIACO",       # PPI: All Commodities (index)
    "ng_price":      "MHHNGSP",      # Henry Hub Natural Gas Price ($/MMBtu)
    "recession":     "USREC",        # NBER Recession Indicator (0/1)
    "t_bill_3m":     "TB3MS",        # 3-Month T-Bill Secondary Market Rate (%)
    "treasury_2y":   "GS2",          # 2-Year Treasury Constant Maturity Rate (%)
    "treasury_10y":  "GS10",         # 10-Year Treasury Constant Maturity Rate (%)
}

# Quarterly
FRED_QUARTERLY = {
    "real_gdp":      "GDPC1",        # Real GDP (billions chained 2017 $)
}

# ── Baker Hughes ──────────────────────────────────────────────────────────────
# File glob pattern — matches any BH report file in data/raw/baker_hughes/
BH_FILE_PATTERN  = "*North America Rig Count Report*"
BH_WEEKLY_SHEET  = "NAM Weekly"    # sheet for master_weekly
BH_MONTHLY_SHEET = "NAM Monthly"   # sheet for master_monthly
BH_HEADER_ROW    = 10              # 0-indexed (row 11 in Excel)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS        = ROOT / "outputs"
FIGURES        = OUTPUTS / "figures"

# ── Pipeline Settings ─────────────────────────────────────────────────────────
# When resampling to lower frequency, use these aggregation methods
RESAMPLE_AGG = {
    "price":      "mean",
    "rig_count":  "mean",
    "production": "mean",
    "rate":       "mean",
    "index":      "mean",
    "indicator":  "max",     # recession indicator: 1 if any week in month = 1
}

# Max allowable gap (in periods) before a series is flagged as sparse
MAX_GAP_PERIODS = 4
