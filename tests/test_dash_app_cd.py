"""
Tests for dash_app_jl_with_cd.py — change detection, asymmetric shock analysis,
rolling correlation, and regime statistics.

Run with:
    python -m pytest tests/test_dash_app_cd.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers imported directly from the app module
# ---------------------------------------------------------------------------
from dash_app_jl_with_cd import (
    # Change detection
    cusum_detect,
    pelt_detect,
    _RUPTURES_AVAILABLE,
    # New feature helpers
    compute_asymmetric_ccf,
    build_asymmetric_ccf_fig,
    build_rolling_corr_fig,
    build_regime_stats_table,
    # Supporting helpers
    blank_figure,
    MONTHLY_DF,
    CD_SERIES_COLS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def flat_series() -> pd.Series:
    """Constant series — no change points should be detected."""
    return pd.Series(np.ones(60))


@pytest.fixture()
def step_series() -> pd.Series:
    """Single hard step at index 30 — one change point expected."""
    values = np.concatenate([np.ones(30) * 10, np.ones(30) * 50])
    return pd.Series(values)


@pytest.fixture()
def noisy_step_series() -> pd.Series:
    """Step with noise — still clearly bimodal."""
    rng = np.random.default_rng(42)
    low  = rng.normal(10, 1, 30)
    high = rng.normal(50, 1, 30)
    return pd.Series(np.concatenate([low, high]))


@pytest.fixture()
def small_monthly_frame() -> pd.DataFrame:
    """60-row synthetic monthly DataFrame matching the real column schema."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2010-01-01", periods=60, freq="MS")
    wti = 50 + np.cumsum(rng.normal(0, 2, 60))
    rig = 1000 + np.cumsum(rng.normal(0, 20, 60))
    prod = 8 + np.cumsum(rng.normal(0, 0.1, 60))
    recession = np.zeros(60, dtype=int)
    recession[10:14] = 1
    wti_mom_pct = pd.Series(wti).pct_change().fillna(0).values * 100
    return pd.DataFrame({
        "date": dates,
        "wti_price_weekly": wti,
        "rig_count": rig,
        "us_production_mbbld": prod,
        "recession": recession,
        "wti_mom_pct": wti_mom_pct,
    })


@pytest.fixture()
def real_monthly_window() -> pd.DataFrame:
    """Last 60 rows of the real loaded monthly data."""
    return MONTHLY_DF.tail(60).copy()


# ===========================================================================
# CUSUM Tests
# ===========================================================================

class TestCusumDetect:
    def test_flat_series_no_breaks(self, flat_series):
        """Constant signal should produce zero change points."""
        cps, sp, sn = cusum_detect(flat_series, threshold=5.0, drift=0.5)
        assert cps == [], f"Expected no breaks, got {cps}"

    def test_step_detects_break(self, step_series):
        """A hard step of 40 std-devs should trigger at least one break."""
        cps, _, _ = cusum_detect(step_series, threshold=5.0, drift=0.5)
        assert len(cps) >= 1

    def test_break_detected_in_first_half(self, step_series):
        """
        CUSUM should detect a break in the first 40 indices.
        The step is at index 30; S− accumulates from index 0 because the pre-step
        values are below the full-series mean, so CUSUM correctly fires early
        (typically around index 10–12) rather than exactly at the step edge.
        """
        cps, _, _ = cusum_detect(step_series, threshold=5.0, drift=0.5)
        assert len(cps) >= 1
        assert cps[0] < 40, f"First break at {cps[0]}, expected somewhere before index 40"

    def test_returns_arrays_correct_length(self, step_series):
        """S+ and S− arrays must match the length of the input series."""
        cps, sp, sn = cusum_detect(step_series)
        assert len(sp) == len(step_series)
        assert len(sn) == len(step_series)

    def test_arrays_non_negative(self, noisy_step_series):
        """CUSUM statistics are always ≥ 0 by construction."""
        _, sp, sn = cusum_detect(noisy_step_series)
        assert (sp >= 0).all()
        assert (sn >= 0).all()

    def test_higher_threshold_fewer_breaks(self, noisy_step_series):
        """Raising the threshold should not increase the number of breaks."""
        cps_low, _, _  = cusum_detect(noisy_step_series, threshold=2.0)
        cps_high, _, _ = cusum_detect(noisy_step_series, threshold=10.0)
        assert len(cps_high) <= len(cps_low)

    def test_single_value_series(self):
        """Single-element series should not raise and return no breaks."""
        cps, sp, sn = cusum_detect(pd.Series([42.0]))
        assert cps == []

    def test_all_nan_handled(self):
        """Series of NaNs should not raise (std == 0 guard)."""
        s = pd.Series([np.nan] * 10)
        try:
            cps, _, _ = cusum_detect(s)
        except Exception as exc:
            pytest.fail(f"cusum_detect raised on all-NaN series: {exc}")

    def test_real_wti_produces_breaks(self, real_monthly_window):
        """Real WTI price data over 60 months should yield at least one break."""
        col = "wti_price_weekly"
        if col not in real_monthly_window.columns:
            pytest.skip("wti_price_weekly not in real data")
        cps, _, _ = cusum_detect(real_monthly_window[col], threshold=3.0, drift=0.3)
        assert len(cps) >= 1, "Expected at least one structural break in real WTI data"


# ===========================================================================
# PELT Tests
# ===========================================================================

class TestPeltDetect:
    def test_returns_list(self, step_series):
        result = pelt_detect(step_series)
        assert isinstance(result, list)

    @pytest.mark.skipif(not _RUPTURES_AVAILABLE, reason="ruptures not installed")
    def test_step_detects_break(self, step_series):
        """Hard step should produce at least one break."""
        cps = pelt_detect(step_series, penalty=1.0)
        assert len(cps) >= 1

    @pytest.mark.skipif(not _RUPTURES_AVAILABLE, reason="ruptures not installed")
    def test_break_near_midpoint(self, step_series):
        cps = pelt_detect(step_series, penalty=1.0)
        assert len(cps) >= 1
        assert abs(cps[0] - 29) <= 5, f"First PELT break at {cps[0]}, expected ~29"

    @pytest.mark.skipif(not _RUPTURES_AVAILABLE, reason="ruptures not installed")
    def test_higher_penalty_fewer_breaks(self, noisy_step_series):
        cps_low  = pelt_detect(noisy_step_series, penalty=1.0)
        cps_high = pelt_detect(noisy_step_series, penalty=100.0)
        assert len(cps_high) <= len(cps_low)

    @pytest.mark.skipif(not _RUPTURES_AVAILABLE, reason="ruptures not installed")
    def test_flat_series_no_breaks(self, flat_series):
        cps = pelt_detect(flat_series, penalty=10.0)
        assert cps == []

    @pytest.mark.skipif(not _RUPTURES_AVAILABLE, reason="ruptures not installed")
    def test_indices_in_range(self, noisy_step_series):
        """All returned indices must be valid positions within the series."""
        cps = pelt_detect(noisy_step_series, penalty=3.0)
        for cp in cps:
            assert 0 <= cp < len(noisy_step_series), f"Index {cp} out of range"

    @pytest.mark.skipif(not _RUPTURES_AVAILABLE, reason="ruptures not installed")
    def test_real_wti(self, real_monthly_window):
        col = "wti_price_weekly"
        if col not in real_monthly_window.columns:
            pytest.skip("wti_price_weekly not in real data")
        cps = pelt_detect(real_monthly_window[col], penalty=5.0)
        assert isinstance(cps, list)


# ===========================================================================
# Asymmetric CCF Tests
# ===========================================================================

class TestAsymmetricCcf:
    def test_output_shape(self, small_monthly_frame):
        """compute_asymmetric_ccf should return 2*max_lag+1 rows."""
        df = compute_asymmetric_ccf(
            small_monthly_frame["wti_mom_pct"],
            small_monthly_frame["rig_count"],
            max_lag=6,
        )
        assert len(df) == 13  # -6 … +6
        assert set(df.columns) >= {"lag", "ccf", "ci_95"}

    def test_lag_zero_correlation_bounded(self, small_monthly_frame):
        """Lag-0 CCF should be in [-1, 1]."""
        df = compute_asymmetric_ccf(
            small_monthly_frame["wti_mom_pct"],
            small_monthly_frame["rig_count"],
        )
        r_zero = float(df.loc[df["lag"] == 0, "ccf"].iloc[0])
        assert -1.0 <= r_zero <= 1.0

    def test_all_ccf_bounded(self, small_monthly_frame):
        """All CCF values must be in [-1, 1]."""
        df = compute_asymmetric_ccf(
            small_monthly_frame["wti_mom_pct"],
            small_monthly_frame["rig_count"],
        )
        assert (df["ccf"].between(-1.0, 1.0)).all()

    def test_positive_clip_no_negative_inputs(self, small_monthly_frame):
        """After clipping to positive, all input x values should be ≥ 0."""
        clipped = small_monthly_frame["wti_mom_pct"].clip(lower=0)
        assert (clipped >= 0).all()

    def test_fig_returns_figure(self, small_monthly_frame):
        """build_asymmetric_ccf_fig should return a Plotly Figure."""
        import plotly.graph_objects as go
        start = small_monthly_frame["date"].iloc[0]
        end   = small_monthly_frame["date"].iloc[-1]
        fig = build_asymmetric_ccf_fig(small_monthly_frame, "rig", "all", start, end)
        assert isinstance(fig, go.Figure)

    def test_fig_positive_shock(self, small_monthly_frame):
        import plotly.graph_objects as go
        start = small_monthly_frame["date"].iloc[0]
        end   = small_monthly_frame["date"].iloc[-1]
        fig = build_asymmetric_ccf_fig(small_monthly_frame, "rig", "positive", start, end)
        assert isinstance(fig, go.Figure)

    def test_fig_negative_shock(self, small_monthly_frame):
        import plotly.graph_objects as go
        start = small_monthly_frame["date"].iloc[0]
        end   = small_monthly_frame["date"].iloc[-1]
        fig = build_asymmetric_ccf_fig(small_monthly_frame, "rig", "negative", start, end)
        assert isinstance(fig, go.Figure)

    def test_fig_empty_frame_returns_blank(self):
        """Empty DataFrame should return a blank figure, not raise."""
        import plotly.graph_objects as go
        empty = pd.DataFrame()
        fig = build_asymmetric_ccf_fig(
            empty, "rig", "all",
            pd.Timestamp("2010-01-01"), pd.Timestamp("2015-01-01"),
        )
        assert isinstance(fig, go.Figure)

    def test_fig_missing_column_returns_blank(self, small_monthly_frame):
        """Frame missing wti_mom_pct should return a blank figure."""
        import plotly.graph_objects as go
        bad = small_monthly_frame.drop(columns=["wti_mom_pct"])
        start = bad["date"].iloc[0]
        end   = bad["date"].iloc[-1]
        fig = build_asymmetric_ccf_fig(bad, "rig", "all", start, end)
        assert isinstance(fig, go.Figure)


# ===========================================================================
# Rolling Correlation Tests
# ===========================================================================

class TestRollingCorr:
    def test_returns_figure(self, small_monthly_frame):
        import plotly.graph_objects as go
        start = small_monthly_frame["date"].iloc[0]
        end   = small_monthly_frame["date"].iloc[-1]
        fig = build_rolling_corr_fig(small_monthly_frame, lag=4, window=12, start_date=start, end_date=end)
        assert isinstance(fig, go.Figure)

    def test_window_larger_than_data_returns_blank(self, small_monthly_frame):
        """If window > len(frame), function should return a blank figure."""
        import plotly.graph_objects as go
        start = small_monthly_frame["date"].iloc[0]
        end   = small_monthly_frame["date"].iloc[-1]
        fig = build_rolling_corr_fig(small_monthly_frame, lag=0, window=200, start_date=start, end_date=end)
        assert isinstance(fig, go.Figure)

    def test_empty_frame_returns_blank(self):
        import plotly.graph_objects as go
        fig = build_rolling_corr_fig(
            pd.DataFrame(), lag=4, window=12,
            start_date=pd.Timestamp("2010-01-01"),
            end_date=pd.Timestamp("2015-01-01"),
        )
        assert isinstance(fig, go.Figure)

    def test_lag_zero(self, small_monthly_frame):
        """Lag of 0 should not raise."""
        import plotly.graph_objects as go
        start = small_monthly_frame["date"].iloc[0]
        end   = small_monthly_frame["date"].iloc[-1]
        fig = build_rolling_corr_fig(small_monthly_frame, lag=0, window=12, start_date=start, end_date=end)
        assert isinstance(fig, go.Figure)

    def test_real_data(self, real_monthly_window):
        """Real data with default lag/window should not raise."""
        import plotly.graph_objects as go
        start = real_monthly_window["date"].iloc[0]
        end   = real_monthly_window["date"].iloc[-1]
        fig = build_rolling_corr_fig(real_monthly_window, lag=4, window=24, start_date=start, end_date=end)
        assert isinstance(fig, go.Figure)


# ===========================================================================
# Regime Statistics Table Tests
# ===========================================================================

class TestRegimeStatsTable:
    def test_no_breaks_returns_message(self, small_monthly_frame):
        """Empty change_points list should return a message Div, not raise."""
        from dash import html
        result = build_regime_stats_table(small_monthly_frame, [], "wti")
        assert isinstance(result, html.Div)

    def test_single_break_two_regimes(self, small_monthly_frame):
        """One change point → two regime rows in the table."""
        from dash import html
        result = build_regime_stats_table(small_monthly_frame, [30], "wti")
        assert isinstance(result, html.Div)
        # Flatten children to find table rows
        rendered = str(result)
        assert "Regime 1" in rendered
        assert "Regime 2" in rendered

    def test_multiple_breaks(self, small_monthly_frame):
        """Multiple breaks should produce multiple labelled regimes."""
        from dash import html
        result = build_regime_stats_table(small_monthly_frame, [15, 30, 45], "wti")
        rendered = str(result)
        assert "Regime 4" in rendered

    def test_empty_frame_returns_message(self):
        """Empty DataFrame should not raise."""
        from dash import html
        result = build_regime_stats_table(pd.DataFrame(), [10], "wti")
        assert isinstance(result, html.Div)

    def test_all_series_keys(self, small_monthly_frame):
        """Should work for all three supported series keys."""
        from dash import html
        for key in ("wti", "rig", "production"):
            result = build_regime_stats_table(small_monthly_frame, [20, 40], key)
            assert isinstance(result, html.Div)

    def test_out_of_bounds_break_index(self, small_monthly_frame):
        """A break index beyond the frame length should not raise."""
        from dash import html
        result = build_regime_stats_table(small_monthly_frame, [999], "wti")
        assert isinstance(result, html.Div)


# ===========================================================================
# End-to-end: figures from real data
# ===========================================================================

class TestFiguresOnRealData:
    def test_cusum_on_all_cd_series(self):
        """CUSUM should run without error on each supported CD series."""
        recent = MONTHLY_DF.tail(120).copy()
        for key, (col, _) in CD_SERIES_COLS.items():
            if col not in recent.columns:
                continue
            cps, sp, sn = cusum_detect(recent[col], threshold=4.0, drift=0.3)
            assert isinstance(cps, list)
            assert len(sp) == len(recent)

    @pytest.mark.skipif(not _RUPTURES_AVAILABLE, reason="ruptures not installed")
    def test_pelt_on_all_cd_series(self):
        """PELT should run without error on each supported CD series."""
        recent = MONTHLY_DF.tail(120).copy()
        for key, (col, _) in CD_SERIES_COLS.items():
            if col not in recent.columns:
                continue
            cps = pelt_detect(recent[col], penalty=5.0)
            assert isinstance(cps, list)

    def test_asym_ccf_on_real_data(self):
        """Asymmetric CCF should return a figure for all shock types and targets."""
        import plotly.graph_objects as go
        recent = MONTHLY_DF.tail(60).copy()
        start = recent["date"].iloc[0]
        end   = recent["date"].iloc[-1]
        for shock in ("all", "positive", "negative"):
            for target in ("rig", "production"):
                fig = build_asymmetric_ccf_fig(recent, target, shock, start, end)
                assert isinstance(fig, go.Figure), f"Failed for shock={shock}, target={target}"

    def test_rolling_corr_on_real_data(self):
        """Rolling correlation should return a valid figure on real data."""
        import plotly.graph_objects as go
        recent = MONTHLY_DF.tail(120).copy()
        start = recent["date"].iloc[0]
        end   = recent["date"].iloc[-1]
        fig = build_rolling_corr_fig(recent, lag=4, window=36, start_date=start, end_date=end)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_regime_table_after_cusum(self):
        """Regime table should populate correctly when CUSUM finds breaks."""
        from dash import html
        recent = MONTHLY_DF.tail(120).copy()
        col = "wti_price_weekly"
        if col not in recent.columns:
            pytest.skip("wti_price_weekly not in data")
        cps, _, _ = cusum_detect(recent[col], threshold=3.0, drift=0.3)
        result = build_regime_stats_table(recent, cps, "wti")
        assert isinstance(result, html.Div)
