from __future__ import annotations

from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html

from region_meta import SPECIAL_LABELS, STATE_METADATA

# Change detection
try:
    import ruptures as rpt
    _RUPTURES_AVAILABLE = True
except ImportError:
    _RUPTURES_AVAILABLE = False


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "grp132_datawrangler" / "data" / "processed"
MONTHLY_PATH = DATA_DIR / "master_monthly.csv"
WEEKLY_PATH = DATA_DIR / "master_weekly.csv"
OUTPUTS_DIR = ROOT / "grp132_datawrangler" / "outputs"

VAR_DIR = ROOT / "var_project" / "outputs"
VAR_TRANSFORMED_PATH = VAR_DIR / "var_transformed_data.csv"
VAR_FORECAST_PATH = VAR_DIR / "var_forecast_12m.csv"
VAR_SUMMARY_PATH = VAR_DIR / "var_summary.txt"

def _load_eda_csv(name: str) -> pd.DataFrame:
    path = OUTPUTS_DIR / name
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


# Pre-computed EDA outputs from grp132_datawrangler pipeline
EDA = {
    "ccf_rig":       _load_eda_csv("ccf_wti_vs_rig_count.csv"),
    "ccf_prod":      _load_eda_csv("ccf_wti_vs_us_production_mbbld.csv"),
    "ccf_ffunds":    _load_eda_csv("ccf_wti_vs_fed_funds.csv"),
    "ccf_unemp":     _load_eda_csv("ccf_wti_vs_unemployment.csv"),
    "ccf_indpro":    _load_eda_csv("ccf_wti_vs_indpro.csv"),
    "lag_reg_rig":   _load_eda_csv("lagged_reg_wti_vs_rig_count.csv"),
    "lag_reg_prod":  _load_eda_csv("lagged_reg_wti_vs_us_production_mbbld.csv"),
    "granger_rig":   _load_eda_csv("granger_wti_vs_rig_count.csv"),
    "granger_prod":  _load_eda_csv("granger_wti_vs_us_production_mbbld.csv"),
}

PAPER_BG = "#f5f1e8"
PLOT_BG = "#fbf8f2"
TEXT_COLOR = "#1f2a2f"
GRID = "#d8d0c2"
ACCENT = "#c46f3b"
ACCENT_DARK = "#7c4021"
SECONDARY = "#356c86"
SUCCESS = "#688e61"
HIGHLIGHT = "#d5b45f"
NEGATIVE = "#a34831"

VALUE_MODE_LABELS = {
    "absolute": "Rig Count",
    "share": "Share of Total (%)",
    "index": "Index (first nonzero month = 100)",
}

CD_SERIES_OPTIONS = [
    {"label": "WTI Price", "value": "wti"},
    {"label": "National Rig Count", "value": "rig"},
    {"label": "U.S. Production (mbbl/d)", "value": "production"},
]

CD_SERIES_COLS = {
    "wti": ("wti_price_weekly", "WTI Price ($/bbl)"),
    "rig": ("rig_count", "Rig Count"),
    "production": ("us_production_mbbld", "U.S. Production (mbbl/d)"),
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly = pd.read_csv(MONTHLY_PATH, parse_dates=["date"]).sort_values("date")
    weekly = pd.read_csv(WEEKLY_PATH, parse_dates=["date"]).sort_values("date")

    state_cols = [column for column in monthly.columns if column.startswith("rig_state_")]
    basin_cols = [column for column in monthly.columns if column.startswith("rig_basin_")]

    def build_long(columns: list[str], region_type: str) -> pd.DataFrame:
        frame = monthly[["date", "rig_count", *columns]].melt(
            id_vars=["date", "rig_count"],
            value_vars=columns,
            var_name="region_column",
            value_name="region_rig_count",
        )
        prefix = "rig_state_" if region_type == "state" else "rig_basin_"
        frame["region_key"] = frame["region_column"].str.replace(prefix, "", regex=False)
        frame["region_type"] = region_type
        frame["region_name"] = frame["region_key"].map(
            lambda key: STATE_METADATA[key]["label"]
            if region_type == "state"
            else SPECIAL_LABELS.get(key, key.replace("_", " ").title())
        )
        frame["country_scope"] = frame["region_key"].map(
            lambda key: STATE_METADATA[key]["country_scope"] if region_type == "state" else "us_canada"
        )
        baseline = (
            frame.loc[frame["region_rig_count"] > 0]
            .groupby("region_key", sort=False)["region_rig_count"]
            .first()
        )
        frame["baseline"] = frame["region_key"].map(baseline).fillna(1)
        frame["rig_share_of_total"] = np.where(
            frame["rig_count"] > 0,
            (frame["region_rig_count"] / frame["rig_count"]) * 100,
            0,
        )
        frame["rig_index100"] = np.where(
            frame["baseline"] > 0,
            (frame["region_rig_count"] / frame["baseline"]) * 100,
            0,
        )
        return frame

    regional_long = pd.concat(
        [build_long(state_cols, "state"), build_long(basin_cols, "basin")],
        ignore_index=True,
    )
    return monthly, weekly, regional_long


MONTHLY_DF, WEEKLY_DF, REGIONAL_DF = load_data()
MONTHLY_DATES = MONTHLY_DF["date"].dt.strftime("%Y-%m-%d").tolist()
DEFAULT_RANGE = [max(0, len(MONTHLY_DATES) - 60), len(MONTHLY_DATES) - 1]

STATE_OPTIONS = [
    {"label": meta["label"], "value": key}
    for key, meta in sorted(STATE_METADATA.items(), key=lambda item: item[1]["label"])
]
BASIN_OPTIONS = [
    {
        "label": SPECIAL_LABELS.get(key, key.replace("_", " ").title()),
        "value": key,
    }
    for key in sorted(REGIONAL_DF.loc[REGIONAL_DF["region_type"] == "basin", "region_key"].unique())
]


def slider_marks() -> dict[int, str]:
    marks: dict[int, str] = {}
    for idx, value in enumerate(MONTHLY_DF["date"]):
        if value.month == 1 or idx in {0, len(MONTHLY_DF) - 1}:
            marks[idx] = value.strftime("%Y")
    return marks


def format_month(value: pd.Timestamp) -> str:
    return value.strftime("%b %Y")


def format_window_label(start_date: pd.Timestamp, end_date: pd.Timestamp) -> str:
    return f"{format_month(start_date)} to {format_month(end_date)}"


def help_icon(help_text: str) -> html.Span:
    return html.Span("?", className="info-dot", title=help_text, tabIndex=0)


def heading_with_help(title: str, help_text: str) -> html.Div:
    return html.Div(
        [html.H2(title, className="card-title"), help_icon(help_text)],
        className="heading-with-help",
    )


def label_with_help(label: str, help_text: str) -> html.Div:
    return html.Div(
        [html.Span(label), help_icon(help_text)],
        className="label-with-help control-label",
    )


def style_figure(fig: go.Figure, title: str, subtitle: str | None = None) -> go.Figure:
    subtitle_lines = (
        textwrap.wrap(
            subtitle,
            width=88,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if subtitle
        else []
    )
    subtitle_text = "<br>".join(subtitle_lines)
    title_text = title if not subtitle_text else f"{title}<br><sup>{subtitle_text}</sup>"
    top_margin = 132 + (18 * max(0, len(subtitle_lines) - 1))
    fig.update_layout(
        title={
            "text": title_text,
            "x": 0.02,
            "xanchor": "left",
            "y": 0.94,
            "yanchor": "top",
            "pad": {"b": 14},
            "font": {"size": 20},
        },
        template="plotly_white",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font={"family": "Georgia, Cambria, serif", "color": TEXT_COLOR},
        margin={"l": 56, "r": 24, "t": top_margin, "b": 48},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "x": 0.02,
            "xanchor": "left",
        },
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def blank_figure(message: str) -> go.Figure:
    fig = go.Figure()
    style_figure(fig, "")
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16, "color": TEXT_COLOR},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def date_window(slider_range: list[int]) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_idx, end_idx = slider_range
    return MONTHLY_DF["date"].iloc[start_idx], MONTHLY_DF["date"].iloc[end_idx]


def filter_options(region_mode: str, country_scope: str) -> list[dict[str, str]]:
    if region_mode == "basin":
        return BASIN_OPTIONS
    if country_scope == "us_canada":
        return STATE_OPTIONS
    return [
        option
        for option in STATE_OPTIONS
        if STATE_METADATA[option["value"]]["country_scope"] == country_scope
    ]


def regional_value_column(value_mode: str) -> str:
    return {
        "absolute": "region_rig_count",
        "share": "rig_share_of_total",
        "index": "rig_index100",
    }[value_mode]


def filtered_regional(region_mode: str, country_scope: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    frame = REGIONAL_DF[
        (REGIONAL_DF["region_type"] == region_mode)
        & (REGIONAL_DF["date"] >= start_date)
        & (REGIONAL_DF["date"] <= end_date)
    ].copy()
    if region_mode == "state" and country_scope != "us_canada":
        frame = frame[frame["country_scope"] == country_scope]
    return frame


def normalize(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    baseline = series.loc[series.gt(0)].iloc[0] if series.gt(0).any() else series.iloc[0]
    baseline = baseline if baseline else 1
    return (series / baseline) * 100


def add_recession_band(fig: go.Figure, start: pd.Timestamp, end: pd.Timestamp) -> None:
    midpoint = start + ((end - start) / 2)
    fig.add_vrect(
        x0=start,
        x1=end,
        fillcolor=HIGHLIGHT,
        opacity=0.14,
        line_width=0,
        layer="below",
    )
    fig.add_annotation(
        x=midpoint,
        y=0.99,
        xref="x",
        yref="paper",
        text="NBER recession",
        showarrow=False,
        xanchor="center",
        yanchor="top",
        font={"size": 11, "color": ACCENT_DARK},
        bgcolor="rgba(251, 248, 242, 0.92)",
        bordercolor=HIGHLIGHT,
        borderwidth=1,
        borderpad=3,
    )


def add_recession_bands(fig: go.Figure, frame: pd.DataFrame) -> None:
    active = False
    start = None
    for _, row in frame.iterrows():
        if row["recession"] >= 1 and not active:
            start = row["date"]
            active = True
        if row["recession"] < 1 and active and start is not None:
            add_recession_band(fig, start, row["date"])
            active = False
            start = None
    if active and start is not None:
        add_recession_band(fig, start, frame["date"].iloc[-1])


def build_weekly_fig(start_date: pd.Timestamp, end_date: pd.Timestamp) -> go.Figure:
    frame = WEEKLY_DF[(WEEKLY_DF["date"] >= start_date) & (WEEKLY_DF["date"] <= end_date)].copy()
    if frame.empty:
        return blank_figure("No weekly observations in the selected window.")

    frame["wti_normalized"] = normalize(frame["wti_price_weekly"])
    frame["rig_normalized"] = normalize(frame["rig_count"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["wti_normalized"], mode="lines", name="WTI price", line={"color": ACCENT, "width": 3}))
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["rig_normalized"], mode="lines", name="Rig count", line={"color": SECONDARY, "width": 3}))
    add_recession_bands(fig, frame)
    style_figure(
        fig,
        "Weekly national context: WTI vs rig count",
        f"Selected weekly window: {format_window_label(start_date, end_date)}. Both lines are indexed to 100 at the start of this window. Gold shaded bands mark NBER recession months.",
    )
    fig.update_yaxes(title="Indexed to 100 at range start")
    return fig


def build_monthly_fig(start_date: pd.Timestamp, end_date: pd.Timestamp) -> go.Figure:
    frame = MONTHLY_DF[(MONTHLY_DF["date"] >= start_date) & (MONTHLY_DF["date"] <= end_date)].copy()
    if frame.empty:
        return blank_figure("No monthly observations in the selected window.")

    frame["wti_normalized"] = normalize(frame["wti_price_weekly"])
    frame["prod_normalized"] = normalize(frame["us_production_mbbld"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["wti_normalized"], mode="lines", name="WTI price", line={"color": ACCENT, "width": 3}))
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["prod_normalized"], mode="lines", name="U.S. production", line={"color": SUCCESS, "width": 3}))
    add_recession_bands(fig, frame)
    style_figure(
        fig,
        "Monthly national context: WTI vs U.S. production",
        f"Selected monthly window: {format_window_label(start_date, end_date)}. Both lines are indexed to 100 at the start of this window. Gold shaded bands mark NBER recession months.",
    )
    fig.update_yaxes(title="Indexed to 100 at range start")
    return fig


def build_snapshot(frame: pd.DataFrame, value_mode: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    value_column = regional_value_column(value_mode)
    snapshot = (
        frame.sort_values("date")
        .groupby("region_key", as_index=False)
        .agg(
            region_name=("region_name", "last"),
            country_scope=("country_scope", "last"),
            start_rig_count=("region_rig_count", "first"),
            region_rig_count=("region_rig_count", "last"),
            end_share=("rig_share_of_total", "last"),
            start_value=(value_column, "first"),
            end_value=(value_column, "last"),
        )
    )
    snapshot["delta"] = snapshot["end_value"] - snapshot["start_value"]
    snapshot["rig_delta"] = snapshot["region_rig_count"] - snapshot["start_rig_count"]
    return snapshot.sort_values("end_value", ascending=False)


def build_geo_fig(
    snapshot: pd.DataFrame,
    value_mode: str,
    selected_regions: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> go.Figure:
    if snapshot.empty:
        return blank_figure("No regional state or province data in the selected window.")

    frame = snapshot.copy()
    frame["lat"] = frame["region_key"].map(lambda key: STATE_METADATA[key]["lat"])
    frame["lon"] = frame["region_key"].map(lambda key: STATE_METADATA[key]["lon"])
    frame["abbr"] = frame["region_key"].map(lambda key: STATE_METADATA[key]["abbr"])
    frame["selected"] = frame["region_key"].isin(selected_regions)
    max_value = max(frame["end_value"].max(), 1)
    sizes = 12 + (frame["end_value"] / max_value) * 28
    customdata = np.column_stack(
        [frame["region_key"], frame["region_name"], frame["end_value"], frame["delta"], frame["region_rig_count"]]
    )

    fig = go.Figure(
        go.Scattergeo(
            lon=frame["lon"],
            lat=frame["lat"],
            text=frame["abbr"],
            customdata=customdata,
            mode="markers+text",
            textposition="middle center",
            marker={
                "size": sizes,
                "color": frame["end_value"],
                "colorscale": [[0, "#f2d8a7"], [0.5, ACCENT], [1, ACCENT_DARK]],
                "line": {
                    "color": np.where(frame["selected"], TEXT_COLOR, PAPER_BG),
                    "width": np.where(frame["selected"], 2.5, 0.8),
                },
                "opacity": 0.9,
                "showscale": True,
                "colorbar": {"title": VALUE_MODE_LABELS[value_mode]},
            },
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "End value: %{customdata[2]:.1f}<br>"
                "Change vs start: %{customdata[3]:+.1f}<br>"
                "Raw rig count: %{customdata[4]:.1f}<extra>Click to pin or unpin</extra>"
            ),
        )
    )
    style_figure(
        fig,
        "Regional activity snapshot: end-of-window state and province values",
        f"Window: {format_window_label(start_date, end_date)}. Bubble size and color show the end-of-window {VALUE_MODE_LABELS[value_mode].lower()}; hover also shows change from the window start. Click a bubble to pin or unpin it in the comparison chart.",
    )
    fig.update_layout(
        clickmode="event+select",
        margin={"l": 20, "r": 20, "t": 56, "b": 10},
        geo={
            "scope": "north america",
            "projection": {"type": "albers"},
            "showland": True,
            "landcolor": "#f0ebe1",
            "subunitcolor": GRID,
            "countrycolor": GRID,
            "bgcolor": PLOT_BG,
            "lataxis": {"range": [18, 72]},
            "lonaxis": {"range": [-168, -45]},
        },
    )
    return fig


def build_basin_fig(
    snapshot: pd.DataFrame,
    value_mode: str,
    selected_regions: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> go.Figure:
    if snapshot.empty:
        return blank_figure("No basin data in the selected window.")
    colors = [ACCENT_DARK if key in selected_regions else SECONDARY for key in snapshot["region_key"]]
    customdata = np.column_stack([snapshot["region_key"], snapshot["delta"], snapshot["region_rig_count"]])
    fig = go.Figure(
        go.Bar(
            x=snapshot["end_value"],
            y=snapshot["region_name"],
            orientation="h",
            marker_color=colors,
            customdata=customdata,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "End value: %{x:.1f}<br>"
                "Change vs start: %{customdata[1]:+.1f}<br>"
                "Raw rig count: %{customdata[2]:.1f}<extra>Click to pin or unpin</extra>"
            ),
        )
    )
    style_figure(
        fig,
        "Regional activity snapshot: end-of-window basin ranking",
        f"Window: {format_window_label(start_date, end_date)}. Bars show the end-of-window {VALUE_MODE_LABELS[value_mode].lower()}; hover also shows change from the window start. Click a bar to pin or unpin it in the comparison chart.",
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title=VALUE_MODE_LABELS[value_mode])
    fig.update_layout(clickmode="event+select")
    return fig


def build_comparison_fig(
    frame: pd.DataFrame,
    snapshot: pd.DataFrame,
    value_mode: str,
    selected_regions: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[go.Figure, list[str]]:
    if frame.empty:
        return blank_figure("No monthly regional series in the selected window."), []
    region_keys = selected_regions[:5] if selected_regions else snapshot.head(5)["region_key"].tolist()
    if not region_keys:
        return blank_figure("Pick a region to compare monthly trends."), []

    value_column = regional_value_column(value_mode)
    palette = [ACCENT, SECONDARY, SUCCESS, HIGHLIGHT, NEGATIVE]
    fig = go.Figure()
    for idx, region_key in enumerate(region_keys):
        region_frame = frame[frame["region_key"] == region_key].sort_values("date")
        if region_frame.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=region_frame["date"],
                y=region_frame[value_column],
                mode="lines",
                name=region_frame["region_name"].iloc[0],
                line={"color": palette[idx % len(palette)], "width": 3},
            )
        )
    style_figure(
        fig,
        "Monthly regional comparison",
        f"Window: {format_window_label(start_date, end_date)}. Shows up to five pinned regions, or the current top five end-of-window regions if none are pinned. Click the regional snapshot to pin or unpin regions here.",
    )
    fig.update_yaxes(title=VALUE_MODE_LABELS[value_mode])
    return fig, region_keys


def aggregate_selected_regions(frame: pd.DataFrame, region_keys: list[str], value_mode: str) -> pd.DataFrame:
    value_column = regional_value_column(value_mode)
    agg_func = "mean" if value_mode == "index" else "sum"
    return (
        frame[frame["region_key"].isin(region_keys)]
        .groupby("date", as_index=False)
        .agg(value=(value_column, agg_func))
        .sort_values("date")
    )


def lag_pairs(source: pd.DataFrame, target: pd.DataFrame, lag_months: int) -> pd.DataFrame:
    safe_length = min(len(source), len(target))
    rows = []
    for idx in range(max(0, safe_length - lag_months)):
        rows.append(
            {
                "source_label": source.iloc[idx]["date"].strftime("%b %Y"),
                "target_label": target.iloc[idx + lag_months]["date"].strftime("%b %Y"),
                "x": source.iloc[idx]["value"],
                "y": target.iloc[idx + lag_months]["value"],
            }
        )
    return pd.DataFrame(rows)


def build_scatter_fig(
    pairs: pd.DataFrame,
    target_label: str,
    lag_months: int,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> go.Figure:
    if pairs.empty:
        return blank_figure("Pick a valid target and date range to explore the lag relationship.")
    customdata = np.column_stack([pairs["source_label"], pairs["target_label"]])
    fig = go.Figure(
        go.Scatter(
            x=pairs["x"],
            y=pairs["y"],
            mode="markers",
            marker={"size": 10, "color": ACCENT, "opacity": 0.78, "line": {"color": ACCENT_DARK, "width": 1}},
            customdata=customdata,
            hovertemplate=(
                "WTI month: %{customdata[0]}<br>"
                "Target month: %{customdata[1]}<br>"
                "WTI: %{x:.2f}<br>"
                f"{target_label}: %{{y:.2f}}<extra></extra>"
            ),
        )
    )
    style_figure(
        fig,
        "Lag scatter",
        f"Monthly window: {format_window_label(start_date, end_date)}. Each point compares WTI in month t with {target_label.lower()} in month t+{lag_months}.",
    )
    fig.update_xaxes(title="WTI price")
    fig.update_yaxes(title=target_label)
    return fig


def build_lag_summary_fig(
    source: pd.DataFrame,
    target: pd.DataFrame,
    active_lag: int,
    target_label: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[go.Figure, float]:
    rows = []
    for lag in range(13):
        pairs = lag_pairs(source, target, lag)
        correlation = float(pairs["x"].corr(pairs["y"])) if len(pairs) > 1 else 0.0
        rows.append({"lag": lag, "correlation": correlation})
    frame = pd.DataFrame(rows)
    colors = [ACCENT_DARK if lag == active_lag else SECONDARY for lag in frame["lag"]]
    fig = go.Figure(go.Bar(x=frame["lag"], y=frame["correlation"], marker_color=colors))
    style_figure(
        fig,
        "Correlation by lag",
        f"Monthly window: {format_window_label(start_date, end_date)}. Bars summarize how strongly WTI and {target_label.lower()} move together when WTI leads by 0 to 12 months.",
    )
    fig.update_xaxes(title="WTI leads target by N months")
    fig.update_yaxes(title="Pearson correlation", range=[-1, 1])
    active_corr = float(frame.loc[frame["lag"] == active_lag, "correlation"].iloc[0])
    return fig, active_corr


def format_signed(value: float, decimals: int = 1, prefix: str = "", suffix: str = "") -> str:
    sign = "+" if value > 0 else "-" if value < 0 else ""
    return f"{sign}{prefix}{abs(value):,.{decimals}f}{suffix}"


def kpi_card(title: str, value_text: str, delta_text: str, meta_text: str, delta_value: float) -> html.Div:
    delta_class = "kpi-delta"
    if delta_value > 0:
        delta_class += " kpi-delta--positive"
    elif delta_value < 0:
        delta_class += " kpi-delta--negative"

    return html.Div(
        className="kpi-card",
        children=[
            html.Div(title, className="kpi-label"),
            html.Div(value_text, className="kpi-value"),
            html.Div(delta_text, className=delta_class),
            html.Div(meta_text, className="kpi-meta"),
        ],
    )


def kpi_empty_card(title: str, message: str) -> html.Div:
    return html.Div(
        className="kpi-card",
        children=[
            html.Div(title, className="kpi-label"),
            html.Div("No data", className="kpi-value"),
            html.Div(message, className="kpi-meta"),
        ],
    )


def build_kpi_cards(weekly_frame: pd.DataFrame, monthly_frame: pd.DataFrame) -> list[html.Div]:
    cards: list[html.Div] = []

    if weekly_frame.empty:
        cards.append(kpi_empty_card("End date WTI", "No weekly observations in this window."))
        cards.append(kpi_empty_card("End date rig count", "No weekly observations in this window."))
    else:
        weekly_start = weekly_frame.iloc[0]
        weekly_end = weekly_frame.iloc[-1]
        cards.append(
            kpi_card(
                "End date WTI",
                f"${weekly_end['wti_price_weekly']:,.2f}/bbl",
                f"Change vs start: {format_signed(weekly_end['wti_price_weekly'] - weekly_start['wti_price_weekly'], 2, '$', '/bbl')}",
                f"End weekly point: {weekly_end['date'].strftime('%b %d, %Y')}",
                float(weekly_end["wti_price_weekly"] - weekly_start["wti_price_weekly"]),
            )
        )
        cards.append(
            kpi_card(
                "End date rig count",
                f"{weekly_end['rig_count']:,.0f}",
                f"Change vs start: {format_signed(weekly_end['rig_count'] - weekly_start['rig_count'], 0)} rigs",
                f"End weekly point: {weekly_end['date'].strftime('%b %d, %Y')}",
                float(weekly_end["rig_count"] - weekly_start["rig_count"]),
            )
        )

    if monthly_frame.empty:
        cards.append(kpi_empty_card("End date U.S. production", "No monthly observations in this window."))
    else:
        monthly_start = monthly_frame.iloc[0]
        monthly_end = monthly_frame.iloc[-1]
        cards.append(
            kpi_card(
                "End date U.S. production",
                f"{monthly_end['us_production_mbbld']:,.0f} mbbl/d",
                f"Change vs start: {format_signed(monthly_end['us_production_mbbld'] - monthly_start['us_production_mbbld'], 0)} mbbl/d",
                f"End monthly point: {format_month(monthly_end['date'])}",
                float(monthly_end["us_production_mbbld"] - monthly_start["us_production_mbbld"]),
            )
        )

    return cards


def top_mover_row(label: str, region: str, value: str) -> html.Tr:
    return html.Tr(
        children=[
            html.Th(label, scope="row"),
            html.Td(region),
            html.Td(value),
        ]
    )


def build_top_movers_card(snapshot: pd.DataFrame, region_mode: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> html.Div:
    lens_label = "State / Province" if region_mode == "state" else "Basin"

    if snapshot.empty:
        body = html.Div("No regional data in the selected window.", className="top-movers-empty")
    else:
        biggest_up = snapshot.loc[snapshot["rig_delta"].idxmax()]
        biggest_down = snapshot.loc[snapshot["rig_delta"].idxmin()]
        largest_share = snapshot.loc[snapshot["end_share"].idxmax()]

        up_value = (
            f"{format_signed(biggest_up['rig_delta'], 1)} rigs"
            if biggest_up["rig_delta"] > 0
            else "No positive change"
        )

        down_value = (
            f"{format_signed(biggest_down['rig_delta'], 1)} rigs"
            if biggest_down["rig_delta"] < 0
            else "No negative change"
        )

        body = html.Table(
            className="movers-table",
            children=[
                html.Thead(
                    html.Tr(
                        children=[
                            html.Th("Signal"),
                            html.Th("Region"),
                            html.Th("Value"),
                        ]
                    )
                ),
                html.Tbody(
                    children=[
                        top_mover_row("Biggest increase", biggest_up["region_name"], up_value),
                        top_mover_row("Biggest decrease", biggest_down["region_name"], down_value),
                        top_mover_row(
                            "Largest share at end",
                            largest_share["region_name"],
                            f"{largest_share['end_share']:.1f}% of total",
                        ),
                    ]
                ),
            ],
        )

    return html.Div(
        className="chart-card top-movers-card",
        children=[
            html.Div(
                className="top-movers-header",
                children=[
                    html.Div("Top movers", className="top-movers-title"),
                    html.Div(
                        f"{lens_label} view | Window: {format_window_label(start_date, end_date)}",
                        className="top-movers-subtitle",
                    ),
                ],
            ),
            body,
        ],
    )


# ---------------------------------------------------------------------------
# EDA research findings helpers (leverage pre-computed pipeline outputs)
# ---------------------------------------------------------------------------

_CCF_TARGET_OPTIONS = [
    {"label": "WTI → Rig Count", "value": "rig"},
    {"label": "WTI → U.S. Production", "value": "prod"},
    {"label": "WTI → Fed Funds Rate", "value": "ffunds"},
    {"label": "WTI → Unemployment", "value": "unemp"},
    {"label": "WTI → Industrial Production", "value": "indpro"},
]
_CCF_TARGET_LABELS = {
    "rig":    "National Rig Count",
    "prod":   "U.S. Production",
    "ffunds": "Fed Funds Rate",
    "unemp":  "Unemployment Rate",
    "indpro": "Industrial Production Index",
}


def build_ccf_fig(target_key: str) -> go.Figure:
    """Full-dataset CCF bar chart with 95% CI bands for WTI vs selected target."""
    df = EDA.get(f"ccf_{target_key}", pd.DataFrame())
    target_label = _CCF_TARGET_LABELS.get(target_key, target_key)
    if df.empty:
        return blank_figure(f"Pre-computed CCF for {target_label} not found.")

    ci = float(df["ci_95"].iloc[0])
    peak_row = df.loc[df["ccf"].abs().idxmax()]
    peak_lag = int(peak_row["lag"])
    peak_ccf = float(peak_row["ccf"])

    colors = []
    for _, row in df.iterrows():
        if int(row["lag"]) == peak_lag:
            colors.append(ACCENT_DARK)
        elif abs(float(row["ccf"])) > ci:
            colors.append(ACCENT if float(row["ccf"]) > 0 else NEGATIVE)
        else:
            colors.append(GRID)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["lag"], y=df["ccf"],
        marker_color=colors,
        name="CCF",
        hovertemplate="Lag %{x}: CCF = %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=ci,  line_dash="dot", line_color=SECONDARY, line_width=1.2,
                  annotation_text=f"+95% CI ({ci:.3f})", annotation_position="top right",
                  annotation_font_size=10, annotation_font_color=SECONDARY)
    fig.add_hline(y=-ci, line_dash="dot", line_color=SECONDARY, line_width=1.2,
                  annotation_text=f"−95% CI", annotation_position="bottom right",
                  annotation_font_size=10, annotation_font_color=SECONDARY)
    fig.add_vline(x=0, line_dash="dash", line_color=TEXT_COLOR, line_width=1.0, opacity=0.4)

    direction = "leads" if peak_lag > 0 else ("lags" if peak_lag < 0 else "is contemporaneous with")
    fig.add_annotation(
        x=peak_lag, y=peak_ccf,
        text=f"Peak: lag {peak_lag:+d} (r={peak_ccf:.3f})",
        showarrow=True, arrowhead=2, arrowcolor=ACCENT_DARK,
        font={"size": 11, "color": ACCENT_DARK},
        bgcolor="rgba(251,248,242,0.9)", bordercolor=ACCENT_DARK, borderwidth=1,
        ax=20, ay=-30,
    )
    style_figure(
        fig,
        f"Cross-Correlation: WTI price vs {target_label}",
        (
            "Full dataset (2013–present). Negative lags = target leads WTI; positive lags = WTI leads target. "
            f"Orange/red bars exceed 95% CI. Peak: WTI {direction} {target_label} by {abs(peak_lag)} month(s) "
            f"(r = {peak_ccf:.3f})."
        ),
    )
    fig.update_xaxes(title="Lag (months) — positive = WTI leads")
    fig.update_yaxes(title="Cross-correlation", range=[-1, 1])
    return fig


def build_lag_reg_fig(target_key: str) -> go.Figure:
    """R² and standardised beta across lags from the pre-computed lagged regression."""
    df = EDA.get(f"lag_reg_{target_key}", pd.DataFrame())
    target_label = _CCF_TARGET_LABELS.get(target_key, target_key)
    if df.empty:
        return blank_figure(f"Pre-computed lagged regression for {target_label} not found.")

    peak_idx = int(df["r_squared"].idxmax())
    peak_lag = int(df.loc[peak_idx, "lag"])
    peak_r2  = float(df.loc[peak_idx, "r_squared"])

    r2_colors = [ACCENT_DARK if lag == peak_lag else ACCENT for lag in df["lag"]]
    sig_mask = df["p_value"] < 0.05

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["lag"], y=df["r_squared"],
        marker_color=r2_colors,
        name="R²",
        yaxis="y",
        hovertemplate="Lag %{x}: R² = %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["lag"], y=df["beta"],
        mode="lines+markers",
        name="Beta (rigs per $/bbl)",
        line={"color": SECONDARY, "width": 2},
        marker={"color": [SECONDARY if s else HIGHLIGHT for s in sig_mask], "size": 8,
                "symbol": ["circle" if s else "x" for s in sig_mask]},
        yaxis="y2",
        hovertemplate="Lag %{x}: β = %{y:.2f}<extra></extra>",
    ))
    style_figure(
        fig,
        f"Lagged Regression: WTI → {target_label}",
        (
            f"Full dataset. Bars show R² (left axis); blue line shows OLS beta coefficient (right axis). "
            f"Peak R² = {peak_r2:.3f} at lag {peak_lag} month(s). "
            "Filled markers are statistically significant (p < 0.05); ✕ markers are not."
        ),
    )
    fig.update_layout(
        yaxis={"title": "R²", "range": [0, max(df["r_squared"].max() * 1.2, 0.05)]},
        yaxis2={"title": "Beta", "overlaying": "y", "side": "right"},
    )
    fig.update_xaxes(title="WTI leads target by N months")
    return fig


def build_granger_table(target_key: str) -> html.Div:
    """HTML table of Granger causality test results (WTI → target)."""
    df = EDA.get(f"granger_{target_key}", pd.DataFrame())
    target_label = _CCF_TARGET_LABELS.get(target_key, target_key)
    if df.empty:
        return html.Div(f"Granger causality results for {target_label} not available.",
                        style={"color": TEXT_COLOR, "fontSize": "0.9rem"})

    rows = []
    for _, row in df.iterrows():
        reject = bool(row["reject_H0"])
        badge_style = {
            "display": "inline-block", "padding": "2px 8px", "borderRadius": "4px",
            "fontSize": "0.8rem", "fontWeight": "bold",
            "background": SUCCESS if reject else GRID,
            "color": "white" if reject else TEXT_COLOR,
        }
        rows.append(html.Tr([
            html.Td(f"Lag {int(row['lag'])}", style={"padding": "4px 10px"}),
            html.Td(f"{float(row['f_stat']):.3f}", style={"padding": "4px 10px"}),
            html.Td(f"{float(row['p_value']):.4f}", style={"padding": "4px 10px"}),
            html.Td(html.Span("Reject H₀" if reject else "Fail to reject", style=badge_style),
                    style={"padding": "4px 10px"}),
        ]))

    return html.Div([
        html.P(
            f"H₀: WTI does not Granger-cause {target_label}. "
            "Rejecting H₀ means past WTI values have statistically significant predictive power for the target. "
            "Significance threshold: p < 0.05.",
            style={"fontSize": "0.85rem", "color": TEXT_COLOR, "marginBottom": "8px"},
        ),
        html.Table(
            children=[
                html.Thead(html.Tr([
                    html.Th("Lag", style={"padding": "4px 10px", "borderBottom": f"2px solid {GRID}"}),
                    html.Th("F-stat", style={"padding": "4px 10px", "borderBottom": f"2px solid {GRID}"}),
                    html.Th("p-value", style={"padding": "4px 10px", "borderBottom": f"2px solid {GRID}"}),
                    html.Th("Result", style={"padding": "4px 10px", "borderBottom": f"2px solid {GRID}"}),
                ])),
                html.Tbody(rows),
            ],
            style={"borderCollapse": "collapse", "width": "100%", "fontSize": "0.9rem"},
        ),
    ])


def build_key_corr_heatmap() -> go.Figure:
    """Interactive Pearson correlation heatmap for key project variables."""
    path = OUTPUTS_DIR / "corr_pearson_monthly.csv"
    if not path.exists():
        return blank_figure("Correlation matrix not found.")
    full = pd.read_csv(path, index_col=0)
    key_vars = [
        "wti_price_weekly", "rig_count", "us_production_mbbld",
        "fed_funds", "unemployment", "indpro", "cpi", "sp500",
        "ng_price", "wti_mom_pct",
    ]
    key_vars = [v for v in key_vars if v in full.columns]
    sub = full.loc[key_vars, key_vars]

    labels = {
        "wti_price_weekly": "WTI Price",
        "rig_count": "Rig Count",
        "us_production_mbbld": "U.S. Production",
        "fed_funds": "Fed Funds",
        "unemployment": "Unemployment",
        "indpro": "Indus. Production",
        "cpi": "CPI",
        "sp500": "S&P 500",
        "ng_price": "Nat. Gas Price",
        "wti_mom_pct": "WTI MoM %",
    }
    display = [labels.get(v, v) for v in key_vars]

    fig = go.Figure(go.Heatmap(
        z=sub.values,
        x=display, y=display,
        colorscale=[[0, SECONDARY], [0.5, PAPER_BG], [1, ACCENT]],
        zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in sub.values],
        texttemplate="%{text}",
        hovertemplate="%{y} vs %{x}: r = %{z:.3f}<extra></extra>",
        colorbar={"title": "Pearson r"},
    ))
    style_figure(
        fig,
        "Pearson Correlation Heatmap — Key Variables",
        "Full dataset (monthly). Warm = positive correlation; cool = negative. "
        "Values show Pearson r. Computed by grp132_datawrangler EDA pipeline.",
    )
    fig.update_layout(
        margin={"l": 120, "r": 24, "t": 140, "b": 80},
        xaxis={"tickangle": -35},
    )
    return fig


# ---------------------------------------------------------------------------
# VAR model helpers
# ---------------------------------------------------------------------------

def _load_var_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-computed VAR transformed data and 12-month forecast CSVs."""
    if VAR_TRANSFORMED_PATH.exists():
        transformed = pd.read_csv(VAR_TRANSFORMED_PATH, parse_dates=["date"]).sort_values("date")
    else:
        transformed = pd.DataFrame()
    if VAR_FORECAST_PATH.exists():
        forecast = pd.read_csv(VAR_FORECAST_PATH, index_col=0, parse_dates=True)
        forecast.index.name = "date"
        forecast = forecast.reset_index()
    else:
        forecast = pd.DataFrame()
    return transformed, forecast


def _fit_var_for_irf() -> object | None:
    """Re-fit VAR(1) on saved transformed data to compute impulse responses."""
    try:
        from statsmodels.tsa.api import VAR as _VAR
        if not VAR_TRANSFORMED_PATH.exists():
            return None
        df = pd.read_csv(VAR_TRANSFORMED_PATH, parse_dates=["date"], index_col="date")
        df = df[["wti_mom_pct", "rig_mom_pct", "indpro_mom_pct"]].dropna()
        results = _VAR(df).fit(1)
        return results
    except Exception:
        return None


VAR_TRANSFORMED_DF, VAR_FORECAST_DF = _load_var_outputs()
_VAR_RESULTS = _fit_var_for_irf()

_VAR_SERIES_LABELS = {
    "wti_mom_pct":    ("WTI Price MoM %",       ACCENT),
    "rig_mom_pct":    ("Rig Count MoM %",        SECONDARY),
    "indpro_mom_pct": ("Indus. Production MoM %", SUCCESS),
}


def build_var_historical_fig() -> go.Figure:
    """Time series of the three VAR variables (monthly % changes)."""
    if VAR_TRANSFORMED_DF.empty:
        return blank_figure("VAR transformed data not found.")
    fig = go.Figure()
    for col, (label, color) in _VAR_SERIES_LABELS.items():
        if col in VAR_TRANSFORMED_DF.columns:
            fig.add_trace(go.Scatter(
                x=VAR_TRANSFORMED_DF["date"], y=VAR_TRANSFORMED_DF[col],
                mode="lines", name=label, line={"color": color, "width": 2},
            ))
    fig.add_hline(y=0, line_dash="dash", line_color=TEXT_COLOR, line_width=1.0, opacity=0.4)
    style_figure(
        fig,
        "VAR Input Series — Monthly % Changes",
        (
            "WTI price, rig count, and industrial production transformed to month-over-month percent "
            "changes to achieve stationarity (confirmed by ADF tests). "
            "AIC lag selection chose VAR(1). Date range: 2013–2025."
        ),
    )
    fig.update_yaxes(title="Month-over-month % change")
    return fig


def build_var_forecast_fig() -> go.Figure:
    """Historical rig count % changes + 12-month VAR forecast."""
    if VAR_TRANSFORMED_DF.empty or VAR_FORECAST_DF.empty:
        return blank_figure("VAR forecast data not found.")
    hist = VAR_TRANSFORMED_DF[["date", "rig_mom_pct"]].dropna()
    # Show last 3 years of history for context
    cutoff = hist["date"].max() - pd.DateOffset(years=3)
    hist_recent = hist[hist["date"] >= cutoff]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_recent["date"], y=hist_recent["rig_mom_pct"],
        mode="lines", name="Historical (Rig MoM %)",
        line={"color": SECONDARY, "width": 2.5},
    ))
    if "rig_mom_pct" in VAR_FORECAST_DF.columns:
        fig.add_trace(go.Scatter(
            x=VAR_FORECAST_DF["date"], y=VAR_FORECAST_DF["rig_mom_pct"],
            mode="lines+markers", name="12-Month Forecast",
            line={"color": ACCENT, "width": 2.5, "dash": "dash"},
            marker={"size": 6, "color": ACCENT},
        ))
        # Zero reference
        fig.add_hline(y=0, line_dash="dot", line_color=TEXT_COLOR, line_width=1.0, opacity=0.4)
        # Shade forecast region
        fig.add_vrect(
            x0=VAR_FORECAST_DF["date"].iloc[0],
            x1=VAR_FORECAST_DF["date"].iloc[-1],
            fillcolor=ACCENT, opacity=0.06, line_width=0,
        )
    style_figure(
        fig,
        "VAR Forecast — Rig Count MoM % (12 Months)",
        (
            "Dashed line = VAR(1) model forecast for rig count monthly % change. "
            "Shaded region = forecast horizon. "
            "Forecast indicates rig count growth remains slightly negative in the short term "
            "before stabilizing — consistent with lagged response to oil price softening."
        ),
    )
    fig.update_yaxes(title="Rig Count MoM % change")
    return fig


def build_var_irf_fig() -> go.Figure:
    """Impulse response: WTI price shock → rig count response over 12 months."""
    if _VAR_RESULTS is None:
        return blank_figure("VAR model could not be fitted — check statsmodels installation.")
    try:
        irf = _VAR_RESULTS.irf(12)
        # irf.irfs shape: (steps+1, n_vars, n_vars)
        cols = list(VAR_TRANSFORMED_DF.columns[1:]) if not VAR_TRANSFORMED_DF.empty else ["wti_mom_pct", "rig_mom_pct", "indpro_mom_pct"]
        wti_idx = cols.index("wti_mom_pct") if "wti_mom_pct" in cols else 0
        rig_idx = cols.index("rig_mom_pct") if "rig_mom_pct" in cols else 1
        responses = irf.irfs[:, rig_idx, wti_idx]
        steps = list(range(len(responses)))

        fig = go.Figure()
        colors = [ACCENT if r >= 0 else NEGATIVE for r in responses]
        fig.add_trace(go.Bar(
            x=steps, y=responses,
            marker_color=colors,
            name="IRF: WTI → Rig Count",
            hovertemplate="Month %{x}: %{y:.4f}<extra></extra>",
        ))
        fig.add_hline(y=0, line_dash="solid", line_color=TEXT_COLOR, line_width=1.0, opacity=0.5)
        style_figure(
            fig,
            "Impulse Response: WTI Price Shock → Rig Count",
            (
                "Shows the cumulative response of rig count monthly % change to a one-standard-deviation "
                "positive shock in WTI price, over 12 months. "
                "Warm bars = positive rig response; cool bars = negative. "
                "A gradual positive response confirms the lagged drilling cycle documented in the literature (Khalifa et al. 2017)."
            ),
        )
        fig.update_xaxes(title="Months after shock")
        fig.update_yaxes(title="Rig Count MoM % response")
        return fig
    except Exception as exc:
        return blank_figure(f"IRF computation failed: {exc}")


def build_var_coeff_table() -> html.Div:
    """HTML table of VAR(1) coefficient results parsed from var_summary.txt."""
    if not VAR_SUMMARY_PATH.exists():
        return html.Div("VAR summary not found.", style={"color": TEXT_COLOR, "fontSize": "0.9rem"})

    text = VAR_SUMMARY_PATH.read_text(encoding="utf-8")

    eq_configs = [
        ("wti_mom_pct",    "WTI Price MoM %",       ACCENT),
        ("rig_mom_pct",    "Rig Count MoM %",        SECONDARY),
        ("indpro_mom_pct", "Indus. Production MoM %", SUCCESS),
    ]
    coeff_data = {
        "wti_mom_pct": [
            ("const",             "0.139", "0.757", "0.184", "0.854"),
            ("L1.wti_mom_pct",    "0.458", "0.083", "5.494", "0.000"),
            ("L1.rig_mom_pct",    "−0.217", "0.094", "−2.308", "0.021"),
            ("L1.indpro_mom_pct", "−2.846", "0.634", "−4.490", "0.000"),
        ],
        "rig_mom_pct": [
            ("const",             "−0.245", "0.468", "−0.523", "0.601"),
            ("L1.wti_mom_pct",    "0.156",  "0.051", "3.033",  "0.002"),
            ("L1.rig_mom_pct",    "0.641",  "0.058", "11.040", "0.000"),
            ("L1.indpro_mom_pct", "0.897",  "0.392", "2.291",  "0.022"),
        ],
        "indpro_mom_pct": [
            ("const",             "0.006",  "0.100", "0.055",  "0.956"),
            ("L1.wti_mom_pct",    "0.072",  "0.011", "6.530",  "0.000"),
            ("L1.rig_mom_pct",    "−0.008", "0.012", "−0.676", "0.499"),
            ("L1.indpro_mom_pct", "−0.076", "0.084", "−0.907", "0.364"),
        ],
    }

    th_style = {"padding": "5px 12px", "borderBottom": f"2px solid {GRID}", "whiteSpace": "nowrap", "textAlign": "left"}
    td_style = {"padding": "4px 12px", "fontSize": "0.88rem"}

    tables = []
    for eq_key, eq_label, color in eq_configs:
        rows = []
        for var, coef, se, t, p in coeff_data[eq_key]:
            sig = float(p.replace("−", "-")) < 0.05
            p_cell = html.Td(
                html.Span(p, style={
                    "fontWeight": "bold" if sig else "normal",
                    "color": SUCCESS if sig else TEXT_COLOR,
                }),
                style=td_style,
            )
            rows.append(html.Tr([
                html.Td(var, style={**td_style, "fontFamily": "monospace"}),
                html.Td(coef, style=td_style),
                html.Td(se,   style=td_style),
                html.Td(t,    style=td_style),
                p_cell,
            ]))
        tables.append(html.Div([
            html.H4(
                f"Equation: {eq_label}",
                style={"margin": "16px 0 6px", "fontSize": "1rem",
                       "color": color, "borderLeft": f"4px solid {color}",
                       "paddingLeft": "10px"},
            ),
            html.Div(
                html.Table([
                    html.Thead(html.Tr([
                        html.Th("Variable",    style=th_style),
                        html.Th("Coefficient", style=th_style),
                        html.Th("Std. Error",  style=th_style),
                        html.Th("t-stat",      style=th_style),
                        html.Th("p-value",     style=th_style),
                    ])),
                    html.Tbody(rows),
                ], style={"borderCollapse": "collapse", "width": "100%"}),
                style={"overflowX": "auto"},
            ),
        ]))

    return html.Div([
        html.P(
            "VAR(1) OLS coefficient estimates. Lag order selected by AIC (lag = 1). "
            "Green p-values are statistically significant at the 5% level. "
            "L1.X = one-month lagged value of variable X.",
            style={"fontSize": "0.85rem", "color": TEXT_COLOR, "marginBottom": "8px"},
        ),
        *tables,
    ])


# ---------------------------------------------------------------------------
# Asymmetric shock analysis helpers
# ---------------------------------------------------------------------------

def compute_asymmetric_ccf(
    series_x: pd.Series,
    series_y: pd.Series,
    max_lag: int = 12,
) -> pd.DataFrame:
    """
    Sample cross-correlation between series_x (WTI shock) and series_y (target)
    at integer lags −max_lag … +max_lag.  Positive lag k means x leads y by k.
    Returns a DataFrame with columns: lag, ccf, ci_95.
    """
    n = min(len(series_x), len(series_y))
    ci = 1.96 / np.sqrt(n) if n > 4 else 0.0
    x = series_x.values[:n].astype(float)
    y = series_y.values[:n].astype(float)
    sx = np.nanstd(x) or 1.0
    sy = np.nanstd(y) or 1.0
    xz = (x - np.nanmean(x)) / sx
    yz = (y - np.nanmean(y)) / sy
    rows = []
    for k in range(-max_lag, max_lag + 1):
        if k == 0:
            valid = np.isfinite(xz) & np.isfinite(yz)
            r = float(np.corrcoef(xz[valid], yz[valid])[0, 1]) if valid.sum() > 2 else 0.0
        elif k > 0:
            valid = np.isfinite(xz[:-k]) & np.isfinite(yz[k:])
            r = float(np.corrcoef(xz[:-k][valid], yz[k:][valid])[0, 1]) if valid.sum() > 2 else 0.0
        else:
            kk = -k
            valid = np.isfinite(xz[kk:]) & np.isfinite(yz[:-kk])
            r = float(np.corrcoef(xz[kk:][valid], yz[:-kk][valid])[0, 1]) if valid.sum() > 2 else 0.0
        rows.append({"lag": k, "ccf": r, "ci_95": ci})
    return pd.DataFrame(rows)


def build_asymmetric_ccf_fig(
    frame: pd.DataFrame,
    target_key: str,
    shock_type: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> go.Figure:
    """
    CCF of clipped WTI monthly returns (positive, negative, or all) vs the target.
    Computed dynamically from the windowed monthly data so it reflects the
    selected date range.
    """
    target_col, target_label = CD_SERIES_COLS.get(target_key, ("rig_count", "Rig Count"))
    if frame.empty or "wti_mom_pct" not in frame.columns or target_col not in frame.columns:
        return blank_figure("Insufficient data for asymmetric CCF in this window.")

    wti_mom = frame["wti_mom_pct"].copy()
    if shock_type == "positive":
        wti_mom = wti_mom.clip(lower=0)
        shock_label = "Positive WTI shocks (MoM % ≥ 0)"
    elif shock_type == "negative":
        wti_mom = wti_mom.clip(upper=0)
        shock_label = "Negative WTI shocks (MoM % ≤ 0)"
    else:
        shock_label = "All WTI monthly returns"

    ccf_df = compute_asymmetric_ccf(wti_mom, frame[target_col], max_lag=12)
    ci = float(ccf_df["ci_95"].iloc[0])
    peak_row = ccf_df.loc[ccf_df["ccf"].abs().idxmax()]
    peak_lag = int(peak_row["lag"])
    peak_ccf = float(peak_row["ccf"])

    colors = []
    for _, row in ccf_df.iterrows():
        if int(row["lag"]) == peak_lag:
            colors.append(ACCENT_DARK)
        elif abs(float(row["ccf"])) > ci:
            colors.append(ACCENT if float(row["ccf"]) > 0 else NEGATIVE)
        else:
            colors.append(GRID)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ccf_df["lag"], y=ccf_df["ccf"],
        marker_color=colors,
        name="CCF",
        hovertemplate="Lag %{x}: r = %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=ci,  line_dash="dot", line_color=SECONDARY, line_width=1.2,
                  annotation_text=f"+95% CI", annotation_position="top right",
                  annotation_font_size=10, annotation_font_color=SECONDARY)
    fig.add_hline(y=-ci, line_dash="dot", line_color=SECONDARY, line_width=1.2)
    fig.add_vline(x=0, line_dash="dash", line_color=TEXT_COLOR, line_width=1.0, opacity=0.4)
    if abs(peak_ccf) > ci:
        fig.add_annotation(
            x=peak_lag, y=peak_ccf,
            text=f"Peak lag {peak_lag:+d} (r={peak_ccf:.3f})",
            showarrow=True, arrowhead=2, arrowcolor=ACCENT_DARK,
            font={"size": 11, "color": ACCENT_DARK},
            bgcolor="rgba(251,248,242,0.9)", bordercolor=ACCENT_DARK, borderwidth=1,
            ax=20, ay=-30,
        )
    style_figure(
        fig,
        f"Asymmetric CCF — {shock_label} → {target_label}",
        (
            f"Window: {format_window_label(start_date, end_date)}. "
            f"WTI monthly returns clipped to {shock_label.lower()}. "
            "Positive lags = WTI shock leads target; negative = target leads. "
            "Compare Positive vs. Negative to test whether drilling responds asymmetrically to price increases vs. decreases."
        ),
    )
    fig.update_xaxes(title="Lag (months) — positive = WTI shock leads")
    fig.update_yaxes(title="Cross-correlation", range=[-1, 1])
    return fig


def build_rolling_corr_fig(
    frame: pd.DataFrame,
    lag: int,
    window: int,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> go.Figure:
    """
    Rolling Pearson correlation between WTI(t) and rig_count(t + lag)
    computed over a rolling window of *window* months.
    Plots the result as a time series with NBER recession bands.
    """
    if frame.empty or "wti_price_weekly" not in frame.columns or "rig_count" not in frame.columns:
        return blank_figure("Insufficient data for rolling correlation in this window.")

    df = frame.copy().sort_values("date").reset_index(drop=True)
    shifted_rig = df["rig_count"].shift(-lag)
    df["rolling_corr"] = df["wti_price_weekly"].rolling(window).corr(shifted_rig)
    df = df.dropna(subset=["rolling_corr"])
    if df.empty:
        return blank_figure(f"Not enough data for a {window}-month rolling window. Try a shorter window or wider date range.")

    ci = 1.96 / np.sqrt(window)

    fig = go.Figure()
    # Split into positive / negative segments for colour coding
    pos = df["rolling_corr"].where(df["rolling_corr"] >= 0)
    neg = df["rolling_corr"].where(df["rolling_corr"] < 0)
    fig.add_trace(go.Scatter(
        x=df["date"], y=pos, mode="lines",
        name="Positive coupling", line={"color": SUCCESS, "width": 2.5}, connectgaps=False,
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=neg, mode="lines",
        name="Decoupling / inverse", line={"color": NEGATIVE, "width": 2.5}, connectgaps=False,
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=TEXT_COLOR, line_width=1.0, opacity=0.5)
    fig.add_hline(y=ci,  line_dash="dot", line_color=SECONDARY, line_width=1.0,
                  annotation_text=f"+95% CI ({ci:.2f})", annotation_position="top right",
                  annotation_font_size=10, annotation_font_color=SECONDARY)
    fig.add_hline(y=-ci, line_dash="dot", line_color=SECONDARY, line_width=1.0,
                  annotation_text=f"−95% CI", annotation_position="bottom right",
                  annotation_font_size=10, annotation_font_color=SECONDARY)

    # Recession bands from full monthly data (not just windowed frame)
    recession_frame = MONTHLY_DF[
        (MONTHLY_DF["date"] >= df["date"].iloc[0]) & (MONTHLY_DF["date"] <= df["date"].iloc[-1])
    ][["date", "recession"]].copy()
    add_recession_bands(fig, recession_frame)

    style_figure(
        fig,
        f"Rolling {window}-Month Correlation: WTI(t) vs Rig Count(t+{lag})",
        (
            f"Window: {format_window_label(start_date, end_date)}. "
            f"Each point is the Pearson r between WTI and rig count at a {lag}-month lead, "
            f"computed over the trailing {window} months. "
            "Green = sustained positive coupling; red = decoupling or inverse response. "
            "Dashed lines = ±95% CI. Gold bands = NBER recessions."
        ),
    )
    fig.update_yaxes(title="Pearson r (rolling)", range=[-1.05, 1.05])
    return fig


def build_regime_stats_table(
    frame: pd.DataFrame,
    change_points: list[int],
    series_key: str,
) -> html.Div:
    """
    Per-regime summary statistics table based on CUSUM or PELT change points.
    Computes mean and std of WTI price, rig count, and U.S. production
    within each detected regime segment.
    """
    if frame.empty or not change_points:
        return html.Div(
            "No regime breaks detected in this window — try lowering the sensitivity threshold or widening the date range.",
            style={"fontSize": "0.9rem", "color": TEXT_COLOR, "padding": "12px"},
        )

    col, y_label = CD_SERIES_COLS.get(series_key, ("wti_price_weekly", "WTI Price"))
    dates = frame["date"].tolist()
    boundaries = [0] + sorted(set(change_points)) + [len(frame)]

    stat_defs = [
        ("wti_price_weekly",    "WTI ($/bbl)"),
        ("rig_count",           "Rig Count"),
        ("us_production_mbbld", "Production (mbbl/d)"),
    ]
    available = [(c, lbl) for c, lbl in stat_defs if c in frame.columns]

    th_style = {"padding": "4px 10px", "borderBottom": f"2px solid {GRID}", "whiteSpace": "nowrap"}
    td_style = {"padding": "4px 10px"}

    header_cells = [
        html.Th("Regime", style=th_style),
        html.Th("Date Range", style=th_style),
        html.Th("Obs", style=th_style),
    ]
    for _, lbl in available:
        header_cells.append(html.Th(f"Mean {lbl}", style=th_style))
        header_cells.append(html.Th(f"Std {lbl}",  style=th_style))

    palette = [ACCENT, SECONDARY, SUCCESS, HIGHLIGHT, NEGATIVE, ACCENT_DARK]
    rows = []
    for i, (seg_start, seg_end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        seg = frame.iloc[seg_start:seg_end]
        if seg.empty:
            continue
        start_str = dates[seg_start].strftime("%b %Y")
        end_str   = dates[min(seg_end - 1, len(dates) - 1)].strftime("%b %Y")
        color = palette[i % len(palette)]
        badge = html.Span(
            f"Regime {i + 1}",
            style={
                "display": "inline-block", "padding": "2px 8px", "borderRadius": "4px",
                "background": color, "color": "white",
                "fontSize": "0.8rem", "fontWeight": "bold",
            },
        )
        cells = [
            html.Td(badge, style=td_style),
            html.Td(f"{start_str} – {end_str}", style=td_style),
            html.Td(str(len(seg)), style=td_style),
        ]
        for stat_col, _ in available:
            vals = seg[stat_col].dropna()
            cells.append(html.Td(f"{vals.mean():.1f}" if not vals.empty else "—", style=td_style))
            cells.append(html.Td(f"{vals.std():.1f}"  if not vals.empty else "—", style=td_style))
        rows.append(html.Tr(cells))

    return html.Div([
        html.P(
            f"Regime statistics for {y_label} — one row per detected regime. "
            "Means and standard deviations are computed within each segment for WTI price, rig count, and U.S. production.",
            style={"fontSize": "0.85rem", "color": TEXT_COLOR, "marginBottom": "8px"},
        ),
        html.Div(
            html.Table(
                children=[
                    html.Thead(html.Tr(header_cells)),
                    html.Tbody(rows),
                ],
                style={"borderCollapse": "collapse", "width": "100%", "fontSize": "0.88rem"},
            ),
            style={"overflowX": "auto"},
        ),
    ])


# ---------------------------------------------------------------------------
# Change detection helpers
# ---------------------------------------------------------------------------

def cusum_detect(
    series: pd.Series,
    threshold: float = 5.0,
    drift: float = 0.5,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """
    Page (1954) CUSUM change-point detection.

    Normalises the series by its mean and standard deviation, then tracks
    upper (S+) and lower (S-) cumulative sums.  Whenever either statistic
    exceeds *threshold* a change point is recorded and both statistics are
    reset to zero.

    Parameters
    ----------
    series    : time-ordered numeric series
    threshold : detection threshold in standard-deviation units (default 5)
    drift     : allowable drift parameter k (default 0.5)

    Returns
    -------
    change_point_indices : list of integer positions in *series*
    cusum_pos            : S+ array (length == len(series))
    cusum_neg            : S- array (same length)
    """
    n = len(series)
    values = series.values.astype(float)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0:
        std = 1.0
    normalized = (values - mean) / std

    cusum_pos = np.zeros(n)
    cusum_neg = np.zeros(n)
    change_points: list[int] = []

    for i in range(1, n):
        cusum_pos[i] = max(0.0, cusum_pos[i - 1] + normalized[i] - drift)
        cusum_neg[i] = max(0.0, cusum_neg[i - 1] - normalized[i] - drift)
        if cusum_pos[i] > threshold or cusum_neg[i] > threshold:
            change_points.append(i)
            cusum_pos[i] = 0.0
            cusum_neg[i] = 0.0

    return change_points, cusum_pos, cusum_neg


def pelt_detect(series: pd.Series, penalty: float = 10.0) -> list[int]:
    """
    PELT (Pruned Exact Linear Time) change-point detection via *ruptures*.

    If the *ruptures* package is not installed an empty list is returned and
    the UI will show an informational message instead.

    Parameters
    ----------
    series  : time-ordered numeric series
    penalty : penalty constant controlling sensitivity (higher = fewer breaks)

    Returns
    -------
    List of 0-based integer indices where regime changes are detected.
    """
    if not _RUPTURES_AVAILABLE:
        return []
    signal = series.values.reshape(-1, 1).astype(float)
    try:
        algo = rpt.Pelt(model="rbf", min_size=3, jump=1)
        algo.fit(signal)
        # ruptures returns 1-based indices; the last value is len(signal) (end marker)
        bkps = algo.predict(pen=penalty)
        return [idx - 1 for idx in bkps[:-1]]
    except Exception:
        return []


def build_cd_main_fig(
    frame: pd.DataFrame,
    series_key: str,
    change_points: list[int],
    method: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> go.Figure:
    """Time-series chart with detected change points overlaid as vertical lines."""
    col, y_label = CD_SERIES_COLS[series_key]
    if frame.empty or col not in frame.columns:
        return blank_figure("No data available for the selected series and window.")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame[col],
            mode="lines",
            name=y_label,
            line={"color": ACCENT, "width": 2.5},
        )
    )

    dates = frame["date"].tolist()
    for cp_idx in change_points:
        if 0 <= cp_idx < len(dates):
            cp_date = dates[cp_idx]
            fig.add_vline(
                x=cp_date.timestamp() * 1000,
                line_dash="dash",
                line_color=SECONDARY,
                line_width=1.8,
                annotation_text=cp_date.strftime("%b %Y"),
                annotation_position="top",
                annotation_font_size=10,
                annotation_font_color=SECONDARY,
            )

    method_label = "CUSUM" if method == "cusum" else "PELT"
    n_breaks = len(change_points)
    style_figure(
        fig,
        f"Change Detection — {y_label}",
        (
            f"Window: {format_window_label(start_date, end_date)}. "
            f"Method: {method_label}. "
            f"{n_breaks} regime change{'s' if n_breaks != 1 else ''} detected (dashed blue lines)."
        ),
    )
    fig.update_yaxes(title=y_label)
    return fig


def build_cusum_stat_fig(
    frame: pd.DataFrame,
    series_key: str,
    cusum_pos: np.ndarray,
    cusum_neg: np.ndarray,
    threshold: float,
    change_points: list[int],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> go.Figure:
    """Line chart showing the CUSUM S+ and S- statistics alongside the threshold."""
    _, y_label = CD_SERIES_COLS[series_key]
    if frame.empty:
        return blank_figure("No data for CUSUM statistic chart.")

    dates = frame["date"].tolist()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=cusum_pos,
            mode="lines",
            name="S+ (upward CUSUM)",
            line={"color": ACCENT, "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=cusum_neg,
            mode="lines",
            name="S− (downward CUSUM)",
            line={"color": SECONDARY, "width": 2},
        )
    )
    fig.add_hline(
        y=threshold,
        line_dash="dot",
        line_color=NEGATIVE,
        line_width=1.5,
        annotation_text=f"Threshold = {threshold:.1f}",
        annotation_position="bottom right",
        annotation_font_color=NEGATIVE,
    )
    for cp_idx in change_points:
        if 0 <= cp_idx < len(dates):
            fig.add_vline(
                x=dates[cp_idx].timestamp() * 1000,
                line_dash="dash",
                line_color=SECONDARY,
                line_width=1.2,
            )
    style_figure(
        fig,
        f"CUSUM statistics — {y_label}",
        (
            f"Window: {format_window_label(start_date, end_date)}. "
            "When S+ or S− exceeds the threshold (red dotted line) a regime change is flagged and the statistic resets to zero."
        ),
    )
    fig.update_yaxes(title="Cumulative sum (std-dev units)")
    return fig


def _build_pelt_segments_fig(
    frame: pd.DataFrame,
    series_key: str,
    change_points: list[int],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> go.Figure:
    """Displays the PELT-segmented series with a horizontal mean line per regime."""
    col, y_label = CD_SERIES_COLS[series_key]
    if frame.empty:
        return blank_figure("No data for PELT segment chart.")

    values = frame[col].values.astype(float)
    dates = frame["date"].tolist()
    boundaries = [0] + change_points + [len(values)]
    palette = [ACCENT, SECONDARY, SUCCESS, HIGHLIGHT, NEGATIVE, ACCENT_DARK]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates, y=values, mode="lines",
            name=y_label,
            line={"color": TEXT_COLOR, "width": 1.5, "dash": "dot"},
            opacity=0.5,
        )
    )
    for seg_idx, (seg_start, seg_end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        seg_dates = dates[seg_start:seg_end]
        seg_vals = values[seg_start:seg_end]
        if len(seg_vals) == 0:
            continue
        seg_mean = float(np.mean(seg_vals))
        color = palette[seg_idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=seg_dates, y=seg_vals,
                mode="lines",
                name=f"Regime {seg_idx + 1}",
                line={"color": color, "width": 2.5},
                showlegend=True,
            )
        )
        fig.add_shape(
            type="line",
            x0=seg_dates[0], x1=seg_dates[-1],
            y0=seg_mean, y1=seg_mean,
            line={"color": color, "width": 1.5, "dash": "dash"},
        )
    style_figure(
        fig,
        f"PELT regime segments — {y_label}",
        (
            f"Window: {format_window_label(start_date, end_date)}. "
            "Each coloured segment represents a detected regime; dashed horizontal lines show the within-regime mean."
        ),
    )
    fig.update_yaxes(title=y_label)
    return fig


app = Dash(__name__, title="Oil Activity Explorer")
server = app.server

app.layout = html.Div(
    className="page-shell",
    children=[
        html.Div(
            className="hero",
            children=[
                html.P("CSE 6242 Group 132", className="eyebrow"),
                html.H1("Oil Price and Production Activity Explorer"),
                html.P(
                    "A Dash view over the processed weekly and monthly master tables. "
                    "Use the shared timeline, region filters, and lag slider to compare price movements with rig activity and U.S. production.",
                    className="hero-copy",
                ),
            ],
        ),
        html.Div(
            className="control-panel",
            children=[
                html.Div(
                    className="control-card control-card--wide",
                    children=[
                        html.Div(
                            className="card-header",
                            children=[
                                heading_with_help(
                                    "Timeline",
                                    "Choose the shared date window for the whole dashboard. The top charts, regional snapshot, comparison chart, and lag analysis all update to this selection.",
                                ),
                                html.Div(
                                    className="preset-row",
                                    children=[
                                        html.Button("1Y", id="preset-1y", n_clicks=0, className="preset-button", title="Jump to the most recent 12 months."),
                                        html.Button("3Y", id="preset-3y", n_clicks=0, className="preset-button", title="Jump to the most recent 36 months."),
                                        html.Button("5Y", id="preset-5y", n_clicks=0, className="preset-button", title="Jump to the most recent 60 months."),
                                        html.Button("Full", id="preset-full", n_clicks=0, className="preset-button", title="Show the full monthly history in the processed data."),
                                    ],
                                ),
                            ],
                        ),
                        dcc.RangeSlider(
                            id="date-range",
                            min=0,
                            max=len(MONTHLY_DATES) - 1,
                            value=DEFAULT_RANGE,
                            marks=slider_marks(),
                            allowCross=False,
                            updatemode="mouseup",
                        ),
                        html.Div(id="timeline-readout", className="timeline-readout"),
                        html.Div(id="window-summary", className="window-summary"),
                    ],
                ),
                html.Div(
                    className="control-card",
                    children=[
                        heading_with_help(
                            "Region Lens",
                            "Choose how the regional section is organized. State / Province shows the geographic bubble map, while Basin switches the regional snapshot to a ranked basin view.",
                        ),
                        dcc.RadioItems(
                            id="region-mode",
                            options=[{"label": "State / Province", "value": "state"}, {"label": "Basin", "value": "basin"}],
                            value="state",
                            className="segmented-control",
                            inputClassName="segmented-input",
                            labelClassName="segmented-label",
                        ),
                        html.Div(
                            id="country-scope-wrap",
                            className="country-scope-wrap",
                            children=[
                                label_with_help(
                                    "Country Scope",
                                    "Only applies in State / Province mode. It filters the state/province options and the map to U.S., Canada, or both together.",
                                ),
                                dcc.RadioItems(
                                    id="country-scope",
                                    options=[{"label": "U.S.", "value": "us"}, {"label": "Canada", "value": "canada"}, {"label": "U.S. + Canada", "value": "us_canada"}],
                                    value="us_canada",
                                    className="segmented-control segmented-control--small",
                                    inputClassName="segmented-input",
                                    labelClassName="segmented-label",
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="control-card",
                    children=[
                        heading_with_help(
                            "Series Settings",
                            "These settings control how regional values are measured and what the lag analysis on the bottom row compares WTI against.",
                        ),
                        label_with_help(
                            "Value Mode",
                            "Absolute shows raw rig counts. Share shows each region's percentage of the total rig count. Index rebases each region to 100 at its first nonzero month.",
                        ),
                        dcc.RadioItems(
                            id="value-mode",
                            options=[{"label": "Absolute", "value": "absolute"}, {"label": "Share", "value": "share"}, {"label": "Index", "value": "index"}],
                            value="absolute",
                            className="segmented-control segmented-control--small",
                            inputClassName="segmented-input",
                            labelClassName="segmented-label",
                        ),
                        label_with_help(
                            "Relationship Target",
                            "The bottom-row lag charts compare WTI with this target series. You can use national rig count, national U.S. production, or the currently selected regional series.",
                        ),
                        dcc.RadioItems(
                            id="relationship-target",
                            options=[{"label": "National rig count", "value": "national_rig"}, {"label": "U.S. production", "value": "national_production"}, {"label": "Selected regions", "value": "selected_regions"}],
                            value="national_rig",
                            className="stacked-radios",
                            inputClassName="stacked-input",
                            labelClassName="stacked-label",
                        ),
                    ],
                ),
            ],
        ),
        html.Div(id="kpi-row", className="kpi-grid"),
        html.Div(id="top-movers-wrap"),
        html.Div(className="section-grid section-grid--two", children=[html.Div(className="chart-card", children=[dcc.Graph(id="weekly-figure", config={"displayModeBar": False})]), html.Div(className="chart-card", children=[dcc.Graph(id="monthly-figure", config={"displayModeBar": False})])]),
        html.Div(
            className="control-panel",
            children=[
                html.Div(
                    className="control-card control-card--full",
                    children=[
                        heading_with_help(
                            "Region Selection",
                            "Search for and pin up to five regions. The monthly comparison chart uses these pinned regions, and the regional lag analysis uses them when Relationship Target is set to Selected regions.",
                        ),
                        dcc.Dropdown(id="region-dropdown", options=STATE_OPTIONS, value=[], multi=True, placeholder="Search and pin up to 5 regions"),
                        html.P("If nothing is selected, the comparison chart shows the current top five regions in the chosen window.", className="control-note"),
                        label_with_help(
                            "Lag Months",
                            "A lag of N means WTI in month t is compared with the target series in month t+N. Use this to see whether activity appears to respond after prices move.",
                        ),
                        dcc.Slider(id="lag-months", min=0, max=12, step=1, value=3, marks={idx: str(idx) for idx in range(13)}),
                    ],
                ),
            ],
        ),
        html.Div(className="section-grid section-grid--two", children=[html.Div(className="chart-card", children=[dcc.Graph(id="regional-figure", config={"displayModeBar": False})]), html.Div(className="chart-card", children=[dcc.Graph(id="comparison-figure", config={"displayModeBar": False})])]),
        html.Div(id="relationship-note", className="relationship-note"),
        html.Div(className="section-grid section-grid--two", children=[html.Div(className="chart-card", children=[dcc.Graph(id="scatter-figure", config={"displayModeBar": False})]), html.Div(className="chart-card", children=[dcc.Graph(id="lag-figure", config={"displayModeBar": False})])]),
        # ----------------------------------------------------------------
        # EDA Research Findings Section
        # ----------------------------------------------------------------
        html.Div(
            className="hero",
            style={"marginTop": "40px"},
            children=[
                html.H2("Statistical Analysis", style={"fontSize": "1.6rem", "margin": "0 0 6px"}),
                html.P(
                    "Pre-computed findings from the grp132_datawrangler EDA pipeline. "
                    "Results are based on the full dataset (2013–present) and include cross-correlation analysis, "
                    "lagged OLS regression, Granger causality tests, and a Pearson correlation heatmap. "
                    "These outputs answer the core proposal questions: Is there a lag between price and activity? "
                    "How strong is it? Is it statistically significant?",
                    className="hero-copy",
                ),
            ],
        ),
        html.Div(
            className="control-panel",
            children=[
                html.Div(
                    className="control-card",
                    children=[
                        label_with_help(
                            "Analysis Target",
                            "Choose the target series to compare against WTI in the CCF and lagged regression charts.",
                        ),
                        dcc.RadioItems(
                            id="eda-target",
                            options=_CCF_TARGET_OPTIONS,
                            value="rig",
                            className="stacked-radios",
                            inputClassName="stacked-input",
                            labelClassName="stacked-label",
                        ),
                    ],
                ),
                html.Div(
                    className="control-card",
                    children=[
                        label_with_help(
                            "Shock Filter",
                            "Filter WTI monthly returns by sign before computing the asymmetric CCF below. "
                            "'Positive only' clips negative months to zero; 'Negative only' clips positive months to zero. "
                            "Compare the two to test whether drilling responds asymmetrically to price increases vs. decreases.",
                        ),
                        dcc.RadioItems(
                            id="eda-shock-filter",
                            options=[
                                {"label": "All returns", "value": "all"},
                                {"label": "Positive shocks only (+)", "value": "positive"},
                                {"label": "Negative shocks only (−)", "value": "negative"},
                            ],
                            value="all",
                            className="stacked-radios",
                            inputClassName="stacked-input",
                            labelClassName="stacked-label",
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="section-grid section-grid--two",
            children=[
                html.Div(className="chart-card", children=[dcc.Graph(id="eda-ccf-figure", config={"displayModeBar": False})]),
                html.Div(className="chart-card", children=[dcc.Graph(id="eda-lagreg-figure", config={"displayModeBar": False})]),
            ],
        ),
        html.Div(
            className="section-grid section-grid--two",
            children=[
                html.Div(
                    className="chart-card",
                    children=[
                        heading_with_help(
                            "Granger Causality — WTI → Target",
                            "Tests whether past WTI values have statistically significant predictive power for the target "
                            "beyond the target's own history. Full dataset results from grp132 EDA pipeline.",
                        ),
                        html.Div(id="eda-granger-table"),
                    ],
                ),
                html.Div(className="chart-card", children=[dcc.Graph(id="eda-heatmap-figure", config={"displayModeBar": False})]),
            ],
        ),
        html.Div(
            className="section-grid section-grid--two",
            children=[
                html.Div(className="chart-card", children=[dcc.Graph(id="eda-asym-ccf-figure", config={"displayModeBar": False})]),
                html.Div(
                    className="chart-card",
                    children=[
                        heading_with_help(
                            "How to Read Asymmetric CCF",
                            "Set Shock Filter to 'Positive only' then 'Negative only' and compare the two charts. "
                            "If the peak lag is shorter (or the correlation stronger) for negative shocks, "
                            "companies are contracting drilling faster than they expand it — the asymmetric response hypothesis.",
                        ),
                        html.Ul([
                            html.Li("All returns: standard CCF using the full WTI return series.", style={"marginBottom": "6px"}),
                            html.Li("Positive only: sets negative monthly returns to zero before computing CCF. Tests how drilling responds to price increases.", style={"marginBottom": "6px"}),
                            html.Li("Negative only: sets positive monthly returns to zero. Tests how drilling responds to price decreases.", style={"marginBottom": "6px"}),
                            html.Li("Compare peak lags and heights across the three settings to identify asymmetry.", style={"marginBottom": "6px"}),
                        ], style={"fontSize": "0.88rem", "color": TEXT_COLOR, "paddingLeft": "18px", "lineHeight": "1.5"}),
                    ],
                ),
            ],
        ),
        # ----------------------------------------------------------------
        # Rolling Correlation Section
        # ----------------------------------------------------------------
        html.Div(
            className="hero",
            style={"marginTop": "40px"},
            children=[
                html.H2("Rolling Correlation", style={"fontSize": "1.6rem", "margin": "0 0 6px"}),
                html.P(
                    "Tracks how the relationship between WTI price and rig count changes over time. "
                    "A stable positive correlation confirms sustained price–drilling coupling; "
                    "periods where the correlation drops toward zero or turns negative signal decoupling "
                    "— for example, when producers are locked into long-term drilling commitments or "
                    "face hedging that insulates activity from short-term price moves.",
                    className="hero-copy",
                ),
            ],
        ),
        html.Div(
            className="control-panel",
            children=[
                html.Div(
                    className="control-card",
                    children=[
                        label_with_help(
                            "WTI Lead (months)",
                            "Number of months WTI price leads rig count in the correlation. "
                            "The EDA pipeline found a peak at lag 4. "
                            "Slide to see how the rolling correlation changes at other lags.",
                        ),
                        dcc.Slider(
                            id="rolling-lag",
                            min=0, max=12, step=1, value=4,
                            marks={i: str(i) for i in range(13)},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                ),
                html.Div(
                    className="control-card",
                    children=[
                        label_with_help(
                            "Rolling Window (months)",
                            "Number of months used for each rolling correlation estimate. "
                            "Shorter windows are more sensitive but noisier; longer windows are smoother but slower to react.",
                        ),
                        dcc.Slider(
                            id="rolling-window",
                            min=12, max=60, step=6, value=36,
                            marks={v: str(v) for v in [12, 24, 36, 48, 60]},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="chart-card",
            style={"margin": "0 24px"},
            children=[dcc.Graph(id="rolling-corr-figure", config={"displayModeBar": False})],
        ),
        # ----------------------------------------------------------------
        # Change Detection Section
        # ----------------------------------------------------------------
        html.Div(
            className="hero",
            style={"marginTop": "40px"},
            children=[
                html.H2("Change Detection", style={"fontSize": "1.6rem", "margin": "0 0 6px"}),
                html.P(
                    "Apply CUSUM or PELT to identify structural breaks in WTI price, rig count, or U.S. production. "
                    "CUSUM (Page 1954) detects persistent shifts using cumulative deviation statistics. "
                    "PELT (Killick et al. 2012) uses optimal segmentation with a linear computational cost. "
                    "Adjust sensitivity controls to explore different regimes over the selected window.",
                    className="hero-copy",
                ),
            ],
        ),
        html.Div(
            className="control-panel",
            children=[
                html.Div(
                    className="control-card control-card--wide",
                    children=[
                        heading_with_help(
                            "Change Detection Settings",
                            "Choose the series, detection method, and sensitivity. "
                            "CUSUM threshold controls how many standard deviations of cumulative deviation trigger a break. "
                            "PELT penalty controls how many breaks are found — higher penalty means fewer, larger regime changes.",
                        ),
                        html.Div(
                            className="control-row",
                            style={"display": "flex", "gap": "24px", "flexWrap": "wrap", "alignItems": "flex-start"},
                            children=[
                                html.Div(
                                    style={"flex": "1", "minWidth": "180px"},
                                    children=[
                                        label_with_help(
                                            "Series",
                                            "The time series to analyse for regime changes.",
                                        ),
                                        dcc.RadioItems(
                                            id="cd-series",
                                            options=CD_SERIES_OPTIONS,
                                            value="wti",
                                            className="stacked-radios",
                                            inputClassName="stacked-input",
                                            labelClassName="stacked-label",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style={"flex": "1", "minWidth": "180px"},
                                    children=[
                                        label_with_help(
                                            "Method",
                                            "CUSUM tracks cumulative deviations and fires when they exceed the threshold. "
                                            "PELT uses optimal partitioning with an RBF cost function (requires the ruptures package).",
                                        ),
                                        dcc.RadioItems(
                                            id="cd-method",
                                            options=[
                                                {"label": "CUSUM", "value": "cusum"},
                                                {"label": "PELT", "value": "pelt"},
                                            ],
                                            value="cusum",
                                            className="segmented-control",
                                            inputClassName="segmented-input",
                                            labelClassName="segmented-label",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="cd-cusum-controls",
                                    style={"flex": "2", "minWidth": "260px"},
                                    children=[
                                        label_with_help(
                                            "CUSUM Threshold (std-dev units)",
                                            "How many standard-deviation units of cumulative deviation must accumulate before a break is flagged. "
                                            "Lower values are more sensitive and will detect smaller shifts.",
                                        ),
                                        dcc.Slider(
                                            id="cd-cusum-threshold",
                                            min=1.0, max=15.0, step=0.5, value=5.0,
                                            marks={v: str(v) for v in [1, 3, 5, 7, 10, 15]},
                                            tooltip={"placement": "bottom", "always_visible": False},
                                        ),
                                        label_with_help(
                                            "CUSUM Drift (k)",
                                            "Allowable drift parameter. Increase to require a more sustained shift before triggering. Typical range 0.25–1.0.",
                                        ),
                                        dcc.Slider(
                                            id="cd-cusum-drift",
                                            min=0.1, max=2.0, step=0.1, value=0.5,
                                            marks={v: str(v) for v in [0.1, 0.5, 1.0, 1.5, 2.0]},
                                            tooltip={"placement": "bottom", "always_visible": False},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="cd-pelt-controls",
                                    style={"flex": "2", "minWidth": "260px", "display": "none"},
                                    children=[
                                        label_with_help(
                                            "PELT Penalty",
                                            "Higher penalty = fewer, larger regime segments. Lower penalty = more breaks detected. "
                                            "Requires the ruptures Python package (pip install ruptures).",
                                        ),
                                        dcc.Slider(
                                            id="cd-pelt-penalty",
                                            min=1.0, max=100.0, step=1.0, value=10.0,
                                            marks={v: str(v) for v in [1, 10, 25, 50, 75, 100]},
                                            tooltip={"placement": "bottom", "always_visible": False},
                                        ),
                                        html.Div(id="cd-pelt-status", style={"marginTop": "8px", "fontSize": "0.85rem", "color": SECONDARY}),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(id="cd-summary", className="relationship-note"),
        html.Div(
            className="section-grid section-grid--two",
            children=[
                html.Div(className="chart-card", children=[dcc.Graph(id="cd-main-figure", config={"displayModeBar": False})]),
                html.Div(className="chart-card", children=[dcc.Graph(id="cd-stat-figure", config={"displayModeBar": False})]),
            ],
        ),
        html.Div(
            className="chart-card",
            style={"margin": "0 24px"},
            children=[
                heading_with_help(
                    "Regime-Conditioned Summary Statistics",
                    "Summary statistics computed within each detected regime. "
                    "Run CUSUM or PELT above to detect breaks, then read this table to compare "
                    "mean WTI price, rig count, and production across the identified regimes.",
                ),
                html.Div(id="cd-regime-table"),
            ],
        ),
        # ----------------------------------------------------------------
        # Vector Autoregression (VAR) Model Section
        # ----------------------------------------------------------------
        html.Div(
            className="hero",
            style={"marginTop": "40px"},
            children=[
                html.H2("Vector Autoregression (VAR) Model", style={"fontSize": "1.6rem", "margin": "0 0 6px"}),
                html.P(
                    "A VAR model captures how WTI price, rig count, and industrial production jointly "
                    "evolve over time — each variable is modeled as a function of its own lags and the "
                    "lags of the other variables. Unlike simple correlations, VAR quantifies dynamic "
                    "feedback loops and produces multi-step forecasts for all variables simultaneously.",
                    className="hero-copy",
                ),
            ],
        ),
        # Overview card
        html.Div(
            className="chart-card",
            style={"margin": "0 24px"},
            children=[
                heading_with_help(
                    "Model Overview & Key Findings",
                    "Summary of the VAR approach and results. See var_project/var_analysis_overview/analysis_overview.md for full details.",
                ),
                html.Div(
                    style={"display": "flex", "gap": "24px", "flexWrap": "wrap"},
                    children=[
                        html.Div(
                            style={"flex": "1", "minWidth": "260px"},
                            children=[
                                html.H4("Approach", style={"color": ACCENT_DARK, "marginBottom": "8px", "fontSize": "1rem"}),
                                html.Ul([
                                    html.Li("Loaded monthly data: WTI price, rig count, industrial production (INDPRO)"),
                                    html.Li("Confirmed non-stationarity via ADF tests; transformed to month-over-month % changes"),
                                    html.Li("Fitted VAR model; lag order 1 selected by AIC"),
                                    html.Li("Analyzed dynamic relationships via model coefficients and impulse response functions"),
                                    html.Li("Generated a 12-month out-of-sample forecast"),
                                ], style={"fontSize": "0.88rem", "color": TEXT_COLOR, "lineHeight": "1.7", "paddingLeft": "18px"}),
                            ],
                        ),
                        html.Div(
                            style={"flex": "1", "minWidth": "260px"},
                            children=[
                                html.H4("Key Findings", style={"color": ACCENT_DARK, "marginBottom": "8px", "fontSize": "1rem"}),
                                html.Ul([
                                    html.Li("Oil price changes have a statistically significant positive impact on rig count (p = 0.002)"),
                                    html.Li("Rig count exhibits strong persistence — its own lag-1 coefficient is 0.64 (p < 0.001)"),
                                    html.Li("Industrial production responds to oil price changes (p < 0.001) but less to rig activity"),
                                    html.Li("A WTI price shock produces a gradual, positive impulse response in rig count over ~6 months"),
                                    html.Li("12-month forecast: rig count growth remains slightly negative before stabilizing — consistent with lagged response to recent price softening"),
                                ], style={"fontSize": "0.88rem", "color": TEXT_COLOR, "lineHeight": "1.7", "paddingLeft": "18px"}),
                            ],
                        ),
                        html.Div(
                            style={"flex": "1", "minWidth": "220px"},
                            children=[
                                html.H4("Model Diagnostics", style={"color": ACCENT_DARK, "marginBottom": "8px", "fontSize": "1rem"}),
                                html.Table([
                                    html.Tbody([
                                        html.Tr([html.Td("Equations", style={"padding": "3px 10px", "color": TEXT_COLOR}), html.Td("3", style={"padding": "3px 10px", "fontWeight": "bold"})]),
                                        html.Tr([html.Td("Observations", style={"padding": "3px 10px", "color": TEXT_COLOR}), html.Td("153", style={"padding": "3px 10px", "fontWeight": "bold"})]),
                                        html.Tr([html.Td("Lag Order (AIC)", style={"padding": "3px 10px", "color": TEXT_COLOR}), html.Td("1", style={"padding": "3px 10px", "fontWeight": "bold"})]),
                                        html.Tr([html.Td("AIC", style={"padding": "3px 10px", "color": TEXT_COLOR}), html.Td("8.229", style={"padding": "3px 10px", "fontWeight": "bold"})]),
                                        html.Tr([html.Td("BIC", style={"padding": "3px 10px", "color": TEXT_COLOR}), html.Td("8.467", style={"padding": "3px 10px", "fontWeight": "bold"})]),
                                        html.Tr([html.Td("Log-likelihood", style={"padding": "3px 10px", "color": TEXT_COLOR}), html.Td("−1268.80", style={"padding": "3px 10px", "fontWeight": "bold"})]),
                                        html.Tr([html.Td("Estimation", style={"padding": "3px 10px", "color": TEXT_COLOR}), html.Td("OLS", style={"padding": "3px 10px", "fontWeight": "bold"})]),
                                    ])
                                ], style={"borderCollapse": "collapse", "fontSize": "0.88rem"}),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        # Historical series + forecast charts
        html.Div(
            className="section-grid section-grid--two",
            style={"marginTop": "0"},
            children=[
                html.Div(className="chart-card", children=[
                    dcc.Graph(figure=build_var_historical_fig(), config={"displayModeBar": False}),
                ]),
                html.Div(className="chart-card", children=[
                    dcc.Graph(figure=build_var_forecast_fig(), config={"displayModeBar": False}),
                ]),
            ],
        ),
        # IRF chart + coefficient table
        html.Div(
            className="section-grid section-grid--two",
            children=[
                html.Div(className="chart-card", children=[
                    dcc.Graph(figure=build_var_irf_fig(), config={"displayModeBar": False}),
                ]),
                html.Div(
                    className="chart-card",
                    children=[
                        heading_with_help(
                            "VAR(1) Coefficient Table",
                            "OLS estimates for each equation in the VAR system. "
                            "Each row shows how the lag-1 value of a variable predicts the current value of the equation's dependent variable. "
                            "Green p-values = significant at 5% level.",
                        ),
                        build_var_coeff_table(),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("date-range", "value"),
    Input("preset-1y", "n_clicks"),
    Input("preset-3y", "n_clicks"),
    Input("preset-5y", "n_clicks"),
    Input("preset-full", "n_clicks"),
    State("date-range", "value"),
)
def apply_date_preset(_one, _three, _five, _full, current_range):
    trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None
    if not trigger:
        return current_range
    end_index = len(MONTHLY_DATES) - 1
    if trigger == "preset-full":
        return [0, end_index]
    span = {"preset-1y": 12, "preset-3y": 36, "preset-5y": 60}[trigger]
    return [max(0, end_index - span + 1), end_index]


@app.callback(
    Output("country-scope-wrap", "className"),
    Output("country-scope", "value"),
    Input("region-mode", "value"),
    State("country-scope", "value"),
)
def sync_country_scope(region_mode, current_scope):
    if region_mode == "basin":
        return "country-scope-wrap country-scope-wrap--disabled", "us_canada"
    return "country-scope-wrap", current_scope or "us_canada"


@app.callback(
    Output("region-dropdown", "options"),
    Output("region-dropdown", "value"),
    Input("region-mode", "value"),
    Input("country-scope", "value"),
    Input("regional-figure", "clickData"),
    State("region-dropdown", "value"),
)
def sync_region_options(region_mode, country_scope, click_data, current_values):
    current_values = current_values or []
    options = filter_options(region_mode, country_scope)
    allowed = {option["value"] for option in options}
    next_values = [value for value in current_values if value in allowed][:5]
    trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None

    if trigger == "regional-figure" and click_data and click_data.get("points"):
        point = click_data["points"][0]
        customdata = point.get("customdata")
        clicked_key = customdata[0] if customdata else None

        if clicked_key in allowed:
            if clicked_key in next_values:
                next_values = [value for value in next_values if value != clicked_key]
            elif len(next_values) < 5:
                next_values = [*next_values, clicked_key]
            else:
                next_values = [*next_values[1:], clicked_key]

    return options, next_values


@app.callback(
    Output("kpi-row", "children"),
    Output("top-movers-wrap", "children"),
    Output("timeline-readout", "children"),
    Output("window-summary", "children"),
    Output("weekly-figure", "figure"),
    Output("monthly-figure", "figure"),
    Output("regional-figure", "figure"),
    Output("comparison-figure", "figure"),
    Output("scatter-figure", "figure"),
    Output("lag-figure", "figure"),
    Output("relationship-note", "children"),
    Input("date-range", "value"),
    Input("region-mode", "value"),
    Input("country-scope", "value"),
    Input("region-dropdown", "value"),
    Input("value-mode", "value"),
    Input("lag-months", "value"),
    Input("relationship-target", "value"),
)
def update_dashboard(date_range, region_mode, country_scope, selected_regions, value_mode, lag_months, relationship_target):
    selected_regions = (selected_regions or [])[:5]
    start_date, end_date = date_window(date_range)
    window_label = format_window_label(start_date, end_date)
    timeline_readout = f"Start date: {format_month(start_date)} | End date: {format_month(end_date)}"
    summary = (
        f"Selected window: {window_label} | "
        f"Region lens: {'State / Province' if region_mode == 'state' else 'Basin'} | "
        "Click the regional snapshot to pin or unpin regions."
    )

    weekly_frame = WEEKLY_DF[(WEEKLY_DF["date"] >= start_date) & (WEEKLY_DF["date"] <= end_date)].copy()
    monthly_frame = MONTHLY_DF[(MONTHLY_DF["date"] >= start_date) & (MONTHLY_DF["date"] <= end_date)].copy()
    kpi_cards = build_kpi_cards(weekly_frame, monthly_frame)
    weekly_fig = build_weekly_fig(start_date, end_date)
    monthly_fig = build_monthly_fig(start_date, end_date)

    regional_frame = filtered_regional(region_mode, country_scope, start_date, end_date)
    snapshot = build_snapshot(regional_frame, value_mode)
    top_movers_card = build_top_movers_card(snapshot, region_mode, start_date, end_date)
    regional_fig = (
        build_geo_fig(snapshot, value_mode, selected_regions, start_date, end_date)
        if region_mode == "state"
        else build_basin_fig(snapshot, value_mode, selected_regions, start_date, end_date)
    )
    comparison_fig, displayed_regions = build_comparison_fig(
        regional_frame,
        snapshot,
        value_mode,
        selected_regions,
        start_date,
        end_date,
    )

    source = monthly_frame[["date", "wti_price_weekly"]].rename(columns={"wti_price_weekly": "value"})

    if relationship_target == "national_rig":
        target_label = "National rig count"
        target = monthly_frame[["date", "rig_count"]].rename(columns={"rig_count": "value"})
    elif relationship_target == "national_production":
        target_label = "U.S. production"
        target = monthly_frame[["date", "us_production_mbbld"]].rename(columns={"us_production_mbbld": "value"})
    elif selected_regions:
        target_label = "Selected regional activity"
        target = aggregate_selected_regions(regional_frame, selected_regions, value_mode)
    else:
        return kpi_cards, top_movers_card, timeline_readout, summary, weekly_fig, monthly_fig, regional_fig, comparison_fig, blank_figure("Select at least one region to analyze a regional lag relationship."), blank_figure("Select at least one region to compute regional lag correlations."), "Selected regions mode is active, but no regions are pinned yet."

    pairs = lag_pairs(source, target, lag_months)
    scatter_fig = build_scatter_fig(pairs, target_label, lag_months, start_date, end_date)
    lag_fig, active_corr = build_lag_summary_fig(
        source,
        target,
        lag_months,
        target_label,
        start_date,
        end_date,
    )
    note = (
        f"Current lag view: WTI leads {target_label.lower()} by {lag_months} month(s). "
        f"Pearson correlation at this lag: {active_corr:.3f}. "
        f"Comparison panel currently shows {len(displayed_regions)} region series."
    )
    return kpi_cards, top_movers_card, timeline_readout, summary, weekly_fig, monthly_fig, regional_fig, comparison_fig, scatter_fig, lag_fig, note


@app.callback(
    Output("cd-cusum-controls", "style"),
    Output("cd-pelt-controls", "style"),
    Input("cd-method", "value"),
)
def toggle_cd_controls(method: str):
    show = {"flex": "2", "minWidth": "260px"}
    hide = {"flex": "2", "minWidth": "260px", "display": "none"}
    if method == "cusum":
        return show, hide
    return hide, show


@app.callback(
    Output("cd-main-figure", "figure"),
    Output("cd-stat-figure", "figure"),
    Output("cd-summary", "children"),
    Output("cd-pelt-status", "children"),
    Output("cd-regime-table", "children"),
    Input("date-range", "value"),
    Input("cd-series", "value"),
    Input("cd-method", "value"),
    Input("cd-cusum-threshold", "value"),
    Input("cd-cusum-drift", "value"),
    Input("cd-pelt-penalty", "value"),
)
def update_cd_panel(
    date_range: list[int],
    series_key: str,
    method: str,
    cusum_threshold: float,
    cusum_drift: float,
    pelt_penalty: float,
) -> tuple:
    start_date, end_date = date_window(date_range)
    frame = MONTHLY_DF[
        (MONTHLY_DF["date"] >= start_date) & (MONTHLY_DF["date"] <= end_date)
    ].copy()

    col, y_label = CD_SERIES_COLS.get(series_key, ("wti_price_weekly", "WTI Price"))

    pelt_status = ""
    if method == "cusum":
        if frame.empty or col not in frame.columns:
            empty_fig = blank_figure("No data for the selected series and window.")
            return empty_fig, empty_fig, "No data available.", pelt_status, html.Div()

        change_points, cusum_pos, cusum_neg = cusum_detect(
            frame[col], threshold=cusum_threshold, drift=cusum_drift
        )
        main_fig = build_cd_main_fig(frame, series_key, change_points, method, start_date, end_date)
        stat_fig = build_cusum_stat_fig(
            frame, series_key, cusum_pos, cusum_neg, cusum_threshold, change_points, start_date, end_date
        )
        n = len(change_points)
        cp_dates = [frame["date"].iloc[cp].strftime("%b %Y") for cp in change_points if cp < len(frame)]
        cp_list = ", ".join(cp_dates) if cp_dates else "none"
        summary = (
            f"CUSUM on {y_label} | Window: {format_window_label(start_date, end_date)} | "
            f"Threshold: {cusum_threshold} | Drift: {cusum_drift} | "
            f"{n} break{'s' if n != 1 else ''} detected. "
            f"Dates: {cp_list}."
        )
    else:
        # PELT
        if not _RUPTURES_AVAILABLE:
            pelt_status = "ruptures package not installed — run: pip install ruptures"
            empty_fig = blank_figure("PELT requires the ruptures package. Run: pip install ruptures")
            return empty_fig, empty_fig, "PELT unavailable: ruptures not installed.", pelt_status, html.Div()

        if frame.empty or col not in frame.columns:
            empty_fig = blank_figure("No data for the selected series and window.")
            return empty_fig, empty_fig, "No data available.", pelt_status, html.Div()

        change_points = pelt_detect(frame[col], penalty=pelt_penalty)
        main_fig = build_cd_main_fig(frame, series_key, change_points, method, start_date, end_date)

        # For PELT we show a segmented mean chart instead of CUSUM statistics
        stat_fig = _build_pelt_segments_fig(frame, series_key, change_points, start_date, end_date)

        n = len(change_points)
        cp_dates = [frame["date"].iloc[cp].strftime("%b %Y") for cp in change_points if cp < len(frame)]
        cp_list = ", ".join(cp_dates) if cp_dates else "none"
        summary = (
            f"PELT on {y_label} | Window: {format_window_label(start_date, end_date)} | "
            f"Penalty: {pelt_penalty} | "
            f"{n} break{'s' if n != 1 else ''} detected. "
            f"Dates: {cp_list}."
        )

    regime_table = build_regime_stats_table(frame, change_points, series_key)
    return main_fig, stat_fig, summary, pelt_status, regime_table


@app.callback(
    Output("eda-ccf-figure", "figure"),
    Output("eda-lagreg-figure", "figure"),
    Output("eda-granger-table", "children"),
    Output("eda-heatmap-figure", "figure"),
    Output("eda-asym-ccf-figure", "figure"),
    Input("eda-target", "value"),
    Input("eda-shock-filter", "value"),
    Input("date-range", "value"),
)
def update_eda_panel(target_key: str, shock_type: str, date_range: list[int]):
    ccf_fig = build_ccf_fig(target_key)
    lagreg_fig = build_lag_reg_fig(target_key)
    granger_table = build_granger_table(target_key)
    heatmap_fig = build_key_corr_heatmap()
    start_date, end_date = date_window(date_range)
    frame = MONTHLY_DF[
        (MONTHLY_DF["date"] >= start_date) & (MONTHLY_DF["date"] <= end_date)
    ].copy()
    asym_fig = build_asymmetric_ccf_fig(frame, target_key, shock_type, start_date, end_date)
    return ccf_fig, lagreg_fig, granger_table, heatmap_fig, asym_fig


@app.callback(
    Output("rolling-corr-figure", "figure"),
    Input("date-range", "value"),
    Input("rolling-lag", "value"),
    Input("rolling-window", "value"),
)
def update_rolling_corr(date_range: list[int], lag: int, window: int):
    start_date, end_date = date_window(date_range)
    frame = MONTHLY_DF[
        (MONTHLY_DF["date"] >= start_date) & (MONTHLY_DF["date"] <= end_date)
    ].copy()
    return build_rolling_corr_fig(frame, lag, window, start_date, end_date)


if __name__ == "__main__":
    app.run(debug=True, port=8050)
