"""
grp132 Data Pipeline — main entry point.

Usage
-----
# Full run (acquire → clean → merge → EDA):
    python run_pipeline.py

# Skip re-downloading and use cached raw data:
    python run_pipeline.py --use-cache

# Only run EDA (assumes master CSVs already exist in data/processed/):
    python run_pipeline.py --eda-only

# Set API keys via environment variables:
    EIA_API_KEY=xxx FRED_API_KEY=yyy python run_pipeline.py
"""
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# ── project imports ───────────────────────────────────────────────────────────
import config
from src.acquire import eia, baker_hughes, fred as fred_module, yahoo
from src.pipeline.clean import clean
from src.pipeline.merge import build_all_masters, load_masters, save_masters
from src.eda.summary_stats import run_all_summaries
from src.eda.correlations import run_all_correlations
from src.eda.plots import run_all_plots

# ── raw-data cache paths ──────────────────────────────────────────────────────
_RAW_CACHE = {
    "eia_wti_weekly":             config.DATA_RAW / "eia"          / "wti_weekly.csv",
    "eia_us_production_monthly":  config.DATA_RAW / "eia"          / "us_production_monthly.csv",
    "bh_rig_count_weekly":        config.DATA_RAW / "baker_hughes" / "rig_count_weekly.csv",
    "bh_rig_count_monthly":       config.DATA_RAW / "baker_hughes" / "rig_count_monthly.csv",
    "sp500_daily":                config.DATA_RAW / "yahoo"        / "sp500_daily.csv",
    "fred_daily":                 config.DATA_RAW / "fred"         / "fred_daily.csv",
    "fred_monthly":               config.DATA_RAW / "fred"         / "fred_monthly.csv",
    "fred_quarterly":             config.DATA_RAW / "fred"         / "fred_quarterly.csv",
}


def _save_raw(raw: dict[str, pd.DataFrame]) -> None:
    for key, df in raw.items():
        path = _RAW_CACHE[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        logger.info("  raw cache saved → %s", path)


def _load_raw_cache() -> dict[str, pd.DataFrame] | None:
    """Load raw DataFrames from cached CSVs; return None if any are missing."""
    raw = {}
    for key, path in _RAW_CACHE.items():
        if not path.exists():
            logger.warning("  cache miss: %s", path)
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        raw[key] = df
        logger.info("  loaded cache: %s (%d rows)", key, len(df))
    return raw


# ── acquisition ───────────────────────────────────────────────────────────────

def acquire(start: str = config.START_DATE,
            end: str | None = config.END_DATE) -> dict[str, pd.DataFrame]:
    """Download all raw data and return as dict of DataFrames."""
    logger.info("=" * 60)
    logger.info("STEP 1 — Data Acquisition")
    logger.info("=" * 60)

    raw = {}

    # EIA
    try:
        eia_data = eia.fetch_all(start, end)
        raw["eia_wti_weekly"]            = eia_data["wti_weekly"]
        raw["eia_us_production_monthly"] = eia_data["us_production_monthly"]
    except Exception as exc:
        logger.error("EIA acquisition failed: %s", exc)
        logger.warning("Continuing without EIA data — WTI from FRED will be used.")
        raw["eia_wti_weekly"]            = pd.DataFrame()
        raw["eia_us_production_monthly"] = pd.DataFrame()

    # Baker Hughes
    try:
        raw["bh_rig_count_weekly"]  = baker_hughes.fetch_rig_count_weekly(start, end)
        raw["bh_rig_count_monthly"] = baker_hughes.fetch_rig_count_monthly(start, end)
    except Exception as exc:
        logger.error("Baker Hughes acquisition failed: %s", exc)
        raw["bh_rig_count_weekly"]  = pd.DataFrame()
        raw["bh_rig_count_monthly"] = pd.DataFrame()

    # Yahoo Finance — S&P 500 (primary source; longer history than FRED)
    try:
        raw["sp500_daily"] = yahoo.fetch_sp500(start, end)
    except Exception as exc:
        logger.error("Yahoo Finance acquisition failed: %s", exc)
        raw["sp500_daily"] = pd.DataFrame()

    # FRED
    try:
        fred_data = fred_module.fetch_all(start, end)
        raw["fred_daily"]     = fred_data["daily"]
        raw["fred_monthly"]   = fred_data["monthly"]
        raw["fred_quarterly"] = fred_data["quarterly"]
    except Exception as exc:
        logger.error("FRED acquisition failed: %s", exc)
        raw["fred_daily"]     = pd.DataFrame()
        raw["fred_monthly"]   = pd.DataFrame()
        raw["fred_quarterly"] = pd.DataFrame()

    # Inject Yahoo S&P 500 into fred_daily, falling back to FRED SP500 if Yahoo failed
    if not raw["sp500_daily"].empty:
        raw["fred_daily"]["sp500"] = raw["sp500_daily"]["sp500"].reindex(
            raw["fred_daily"].index, method="ffill"
        )
        logger.info("  S&P 500: using Yahoo Finance data.")
    else:
        logger.warning("  S&P 500: Yahoo failed — falling back to FRED SP500 (from 2013).")

    # If EIA WTI is empty, fall back to FRED DCOILWTICO (resample weekly)
    if raw["eia_wti_weekly"].empty and not raw["fred_daily"].empty:
        if "wti_price_d" in raw["fred_daily"].columns:
            logger.info("  Using FRED DCOILWTICO as WTI weekly proxy (resampled).")
            raw["eia_wti_weekly"] = (
                raw["fred_daily"][["wti_price_d"]]
                .resample("W-FRI").mean()
                .rename(columns={"wti_price_d": "wti_price_weekly"})
            )

    _save_raw(raw)
    return raw


# ── cleaning ──────────────────────────────────────────────────────────────────

def clean_raw(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    logger.info("=" * 60)
    logger.info("STEP 2 — Cleaning")
    logger.info("=" * 60)
    cleaned = {}
    for key, df in raw.items():
        if df.empty:
            logger.warning("  %s is empty — skipping", key)
            cleaned[key] = df
        else:
            cleaned[key] = clean(df, label=key, winsorise=False)
    return cleaned


# ── merging ───────────────────────────────────────────────────────────────────

def merge(cleaned: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    logger.info("=" * 60)
    logger.info("STEP 3 — Merging into master datasets")
    logger.info("=" * 60)
    masters = build_all_masters(cleaned)
    return masters


# ── EDA ───────────────────────────────────────────────────────────────────────

def run_eda(masters: dict[str, pd.DataFrame]) -> None:
    logger.info("=" * 60)
    logger.info("STEP 4 — Exploratory Data Analysis")
    logger.info("=" * 60)

    run_all_summaries(masters)

    mm = masters.get("master_monthly")
    if mm is not None and not mm.empty:
        run_all_correlations(mm, out_dir=config.OUTPUTS)
    else:
        logger.warning("  master_monthly empty — skipping correlation analysis.")

    run_all_plots(masters)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="grp132 Oil Data Pipeline")
    p.add_argument("--use-cache",  action="store_true",
                   help="Load raw data from local cache instead of downloading")
    p.add_argument("--eda-only",   action="store_true",
                   help="Skip acquisition/merge and run EDA on existing master CSVs")
    p.add_argument("--start",      default=config.START_DATE,
                   help=f"Start date (default: {config.START_DATE})")
    p.add_argument("--end",        default=config.END_DATE,
                   help="End date (default: today)")
    p.add_argument("--no-eda",     action="store_true",
                   help="Skip EDA step")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = datetime.now()

    logger.info("grp132 Data Pipeline — %s", t0.strftime("%Y-%m-%d %H:%M"))
    logger.info("Start: %s  |  End: %s", args.start, args.end or "today")

    if args.eda_only:
        masters = load_masters()
        if not masters:
            logger.error("No master CSVs found in data/processed/. Run without --eda-only first.")
            sys.exit(1)
        run_eda(masters)
    else:
        # ── Acquisition
        if args.use_cache:
            logger.info("Loading raw data from cache ...")
            raw = _load_raw_cache()
            if raw is None:
                logger.warning("Cache incomplete — falling back to live download.")
                raw = acquire(args.start, args.end)
        else:
            raw = acquire(args.start, args.end)

        # ── Clean
        cleaned = clean_raw(raw)

        # ── Merge
        masters = merge(cleaned)

        # ── EDA
        if not args.no_eda:
            run_eda(masters)

    elapsed = (datetime.now() - t0).seconds
    logger.info("Pipeline complete in %ds. Outputs in: %s", elapsed, config.OUTPUTS)
    logger.info("Master datasets in: %s", config.DATA_PROCESSED)

    _print_summary(masters)


def _print_summary(masters: dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 60)
    print("  MASTER DATASETS")
    print("=" * 60)
    for name, df in masters.items():
        if df.empty:
            print(f"  {name}: EMPTY")
            continue
        print(f"\n  {name}")
        print(f"    rows : {len(df)}")
        print(f"    cols : {df.shape[1]}  {list(df.columns)}")
        print(f"    range: {df.index.min().date()} → {df.index.max().date()}")
        print(f"    file : data/processed/{name}.csv")
    print()


if __name__ == "__main__":
    main()
