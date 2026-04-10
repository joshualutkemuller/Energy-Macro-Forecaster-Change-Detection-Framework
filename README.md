This repository is used for Tracking Oil Price Impacts on Drilling, Formations, and Economic Trends.

## Dash Explorer

The repo includes a Python Dash app for exploring the processed oil price, rig activity, and production data with interactive change detection, EDA, and rolling correlation analysis.

### Run locally

1. `pip install -r grp132_datawrangler/requirements.txt`
2. `pip install ruptures`
3. `python dash_app_final.py`

The app reads the processed CSV files in `grp132_datawrangler/data/processed`.

## Additional Docs

Reference the 'docs/' sub-root for implementation details for the Change Detection & PELT algorithms as well as how the Data Pipeline & EDA components were built.

---

## Changelog — Joshua Lutkemuller

### 2026-04-09
- **Merged PR #6** — Change Detection Dashboard Updates & Fixes
- **Archived** `dash_app.py` → `archive/dash_app.py`; renamed `dash_app_jl_with_cd.py` → `dash_app_final.py` as the canonical dashboard
- **Fixed** three broken Dash callbacks:
  - `update_rolling_corr` — callback was missing entirely despite layout components existing
  - `update_cd_panel` — added `cd-regime-table` output; corrected all early-return arities from 4 to 5 values
  - `update_eda_panel` — added `eda-asym-ccf-figure` output and wired `eda-shock-filter` / `date-range` inputs
- **Installed** `ruptures` v1.1.10 — PELT change detection now fully operational
- **Added** `tests/test_dash_app_cd.py` — 41 unit and end-to-end tests covering CUSUM, PELT, asymmetric CCF, rolling correlation, and regime-conditioned summary statistics (all passing)

### 2026-03-28
- **Merged PR #5** — Additional EDA and detection algorithms
- **Updated** `dash_app_jl_with_cd.py` with three new analytical features:
  - Asymmetric shock analysis (positive vs. negative WTI shocks compared separately)
  - Rolling correlation over time (detects decoupling periods between WTI and rig count)
  - Regime-conditioned summary statistics table (mean/std per detected regime segment)
- **Merged PR #4 and PR #3** — Change detection implementation iterations
- **Added** `docs/ChangeDetectionImplementation.md` — documents CUSUM and PELT math, implementation details, and recommended future EDA/detection features
- **Added** `dash_app_jl_with_cd.py` — working copy of the dashboard with integrated CUSUM and PELT change detection
- **Added** change detection v2 — refined CUSUM algorithm and PELT via `ruptures`
- **Renamed** `docs/README.md` to `Pipeline & EDA README.md`
- **Merged PR #2** — initial change detection integration

### 2026-03-15
- **Merged PR #1** — data pipeline upload
- **Added** initial processed data files and pipeline outputs to the repository
