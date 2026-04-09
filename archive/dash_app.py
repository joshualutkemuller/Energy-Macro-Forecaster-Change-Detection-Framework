from __future__ import annotations

from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html

from region_meta import SPECIAL_LABELS, STATE_METADATA


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "grp132_datawrangler" / "data" / "processed"
MONTHLY_PATH = DATA_DIR / "master_monthly.csv"
WEEKLY_PATH = DATA_DIR / "master_weekly.csv"

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


if __name__ == "__main__":
    app.run(debug=True, port=8050)
