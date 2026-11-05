"""Shared look and feel: colours, number formats, chart styling and small page pieces."""
from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Colours follow one validated palette. Text always uses the ink colours, never a series colour.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
NEUTRAL = "#f0efec"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE = SERIES[0]
ORANGE = SERIES[1]
RAISES_RISK = "#d03b3b"
LOWERS_RISK = "#2a78d6"
HEATMAP_SCALE = [[0.0, "#cde2fb"], [0.33, "#6da7ec"], [0.66, "#256abf"], [1.0, "#0d366b"]]

# Status colours are reserved for risk state and always appear with a text label.
RISK_COLOURS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#ec835a", "Very High": "#d03b3b"}
RISK_ICONS = {"Low": ":material/check_circle:", "Medium": ":material/info:", "High": ":material/warning:", "Very High": ":material/error:"}

PAGE_STYLE = """
<style>
  .block-container { padding-top: 2.2rem; }
  .card { background: #fcfcfb; border: 1px solid rgba(11,11,11,0.10); border-radius: 8px; padding: 14px 16px; height: 100%; }
  .card h4 { margin: 0 0 6px 0; padding: 0; font-size: 0.95rem; font-weight: 600; color: #0b0b0b; }
  .card p { margin: 0; font-size: 0.875rem; line-height: 1.45; color: #52514e; }
  .card .so-what { margin-top: 8px; color: #0b0b0b; }
  .eyebrow { font-size: 0.75rem; font-weight: 600; color: #52514e; margin: 0 0 4px 0; }
  .risk-pill { display: inline-flex; align-items: center; gap: 6px; font-weight: 600; color: #0b0b0b; }
  .risk-dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
  .source-line { font-size: 0.8rem; color: #52514e; }
  [data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
"""


_MARKDOWN_SPECIAL = re.compile(r"([\\`*_\[\]<>~$])")


def plain(text: str) -> str:
    """Show text literally in Streamlit Markdown.

    Without this, two $ signs in one sentence render as a LaTeX formula, and text from an
    LLM could add links or images to the page.
    """
    return _MARKDOWN_SPECIAL.sub(r"\\\1", str(text))


def money(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if abs(value) >= 1e6:
        return f"${value / 1e6:,.1f}M"
    return f"${value:,.0f}"


def percent(value: float, digits: int = 1) -> str:
    return "n/a" if pd.isna(value) else f"{value:.{digits}%}"


def style_figure(fig: go.Figure, height: int = 320, show_legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=13, color=INK_SECONDARY),
        margin=dict(l=8, r=28, t=36 if show_legend else 12, b=8),
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=BASELINE, font=dict(color=INK, size=13)),
        bargap=0.5,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=BASELINE, zeroline=False, automargin=True, ticks="", title_font=dict(size=12))
    fig.update_yaxes(gridcolor=GRID, linecolor=BASELINE, zeroline=False, automargin=True, ticks="", title_font=dict(size=12))
    return fig


def show_chart(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})


def horizontal_bars(
    labels, values, text, hover=None, colour=BLUE, x_title=None, x_format=None, reference=None, reference_label=None
) -> go.Figure:
    """Thin horizontal bars with the value at the tip, largest at the top of the list given."""
    fig = go.Figure(
        go.Bar(
            x=values, y=labels, orientation="h", text=text, textposition="outside", cliponaxis=False,
            marker=dict(color=colour, cornerradius=4),
            hovertext=hover if hover is not None else text, hoverinfo="text",
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    # Leave room past the longest bar, so its value label is never clipped.
    fig.update_xaxes(title_text=x_title, tickformat=x_format, range=[0, float(max(values)) * 1.2])
    if reference is not None:
        fig.add_vline(x=reference, line=dict(color=INK_MUTED, width=1), layer="below")
    if reference_label:
        fig.add_annotation(
            x=reference, y=1.0, yref="paper", yanchor="bottom", text=reference_label, showarrow=False,
            font=dict(size=12, color=INK_SECONDARY),
        )
    return style_figure(fig, height=max(170, 56 + 38 * len(labels)))


def repayment_timeline(months: pd.DataFrame, peer_median: float, height: int = 300) -> go.Figure:
    """Twelve months of days past due, with the peer median as a reference line."""
    colours = {"On-Time": BASELINE, "Late": RISK_COLOURS["Medium"], "Missed": RISK_COLOURS["Very High"]}
    fig = go.Figure()
    for status, colour in colours.items():
        part = months[months["payment_status"] == status]
        fig.add_bar(
            # On-time months sit at 0, so they get a small stub to stay visible. The hover shows the true value.
            x=part["payment_month"], y=part["days_past_due"].clip(lower=1 if status == "On-Time" else 0),
            width=0.55, name=status, marker=dict(color=colour, cornerradius=4), customdata=part["days_past_due"],
            hovertemplate="Month %{x}<br>" + status + "<br>%{customdata} days past due<extra></extra>",
        )
    fig.add_scatter(
        x=[0.4, 12.6], y=[peer_median, peer_median], mode="lines", name=f"Peer median: {peer_median:.1f} days",
        line=dict(color=INK_SECONDARY, width=2), hoverinfo="skip",
    )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(dtick=1, range=[0.4, 12.6], showgrid=False, title_text="Month since origination")
    fig.update_yaxes(title_text="Days past due", rangemode="tozero")
    return style_figure(fig, height=height, show_legend=True)


def plain_table(table: pd.DataFrame) -> None:
    """A static table whose long text wraps instead of being cut off."""
    st.table(table, hide_index=True)


def text_card(title: str, body: str, so_what: str | None = None, eyebrow: str | None = None) -> str:
    top = f'<p class="eyebrow">{eyebrow}</p>' if eyebrow else ""
    bottom = f'<p class="so-what"><strong>So what:</strong> {so_what}</p>' if so_what else ""
    return f'<div class="card">{top}<h4>{title}</h4><p>{body}</p>{bottom}</div>'


def feature_statement(trigger: str, false_positive: str, false_negative: str, human_step: str) -> None:
    """The four things the brief requires each AI feature to state in the dashboard."""
    columns = st.columns(4)
    items = [
        ("What triggers it", trigger),
        ("A false positive", false_positive),
        ("A false negative", false_negative),
        ("The human step", human_step),
    ]
    for column, (title, body) in zip(columns, items):
        column.markdown(text_card(title, body), unsafe_allow_html=True)


def risk_pill(band: str) -> str:
    return (
        f'<span class="risk-pill"><span class="risk-dot" style="background:{RISK_COLOURS.get(band, INK_MUTED)}"></span>'
        f"{band}</span>"
    )


def open_account(account_id: str) -> None:
    """Jump to the Account view for one account."""
    st.session_state["selected_account"] = account_id
    st.session_state["pending_page"] = "Account"
    st.rerun()


def as_table_expander(table: pd.DataFrame, label: str = "View as a table") -> None:
    with st.expander(label):
        st.dataframe(table, width="stretch", hide_index=True)
