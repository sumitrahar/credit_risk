"""Portfolio view: where the risk and the money sit, and how that changes across dimensions."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import (
    BLUE,
    HEATMAP_SCALE,
    NEUTRAL,
    RISK_COLOURS,
    SERIES,
    as_table_expander,
    horizontal_bars,
    money,
    open_account,
    percent,
    plain,
    show_chart,
    style_figure,
    text_card,
)
from src.analytics import cross_tab, month_on_book_curve, portfolio_totals, segment_summary
from src.config import MIN_SEGMENT_ACCOUNTS, RISK_BAND_LABELS

DIMENSIONS = {
    "Loan type": "loan_type",
    "Employment": "employment_type",
    "Income band": "income_band",
    "Age band": "age_band",
    "Region": "region",
    "City tier": "city_tier",
    "Education": "education_level",
    "Housing": "housing_type",
    "Sector": "occupation_sector",
    "Disbursal channel": "disbursal_channel",
    "Loan purpose": "loan_purpose",
    "Risk band": "risk_band",
}
TREND_SPLITS = {
    "No split": None,
    "Loan type": "loan_type",
    "Employment": "employment_type",
    "Income band": "income_band",
    "Age band": "age_band",
    "Region": "region",
}


def _kpis(frame: pd.DataFrame) -> None:
    totals = portfolio_totals(frame)
    columns = st.columns(6)
    columns[0].metric("Accounts", f"{totals['accounts']:,}", help="Loan accounts matching the filters.")
    columns[1].metric("Originated", money(totals["originated"]), help="Sum of original loan amounts.")
    columns[2].metric(
        "Latest balance", money(totals["latest_balance"]),
        help="Outstanding balance at month 12. Approximate: see Data Notes.",
    )
    columns[3].metric("Default rate", percent(totals["default_rate"]), help="Accounts with target_default = 1.")
    columns[4].metric("Defaulted amount", money(totals["defaulted_amount"]), help="Original amount of defaulted loans.")
    columns[5].metric("In review list", f"{int(frame['flagged'].sum()):,}", help="Accounts flagged by two or more anomaly detectors.")


def _risk_bands(frame: pd.DataFrame) -> None:
    table = (
        frame.groupby("risk_band")
        .agg(
            accounts=("account_id", "size"),
            defaults=("target_default", "sum"),
            mean_pd=("pd_score", "mean"),
            originated=("loan_amount", "sum"),
            latest_balance=("latest_balance", "sum"),
        )
        .reindex(RISK_BAND_LABELS)
        .dropna(subset=["accounts"])
    )
    table["default_rate"] = table["defaults"] / table["accounts"]

    left, right = st.columns(2)
    with left:
        st.markdown("**Exposure by risk band**")
        st.caption("Latest balance in each band set by the model's predicted probability of default.")
        fig = horizontal_bars(
            list(table.index), table["latest_balance"], [money(v) for v in table["latest_balance"]],
            hover=[f"{band}<br>{money(row.latest_balance)} latest balance<br>{money(row.originated)} originated<br>{row.accounts:,.0f} accounts"
                   for band, row in table.iterrows()],
            colour=[RISK_COLOURS[band] for band in table.index], x_format="$~s",
        )
        show_chart(fig, key="risk_exposure")
    with right:
        st.markdown("**Default rate by risk band**")
        st.caption("Observed defaults. Each band should sit close to its predicted rate, which is in the hover.")
        fig = horizontal_bars(
            list(table.index), table["default_rate"], [percent(v) for v in table["default_rate"]],
            hover=[f"{band}<br>{percent(row.default_rate)} observed<br>{percent(row.mean_pd)} predicted<br>{row.accounts:,.0f} accounts"
                   for band, row in table.iterrows()],
            colour=[RISK_COLOURS[band] for band in table.index], x_format=".0%",
        )
        show_chart(fig, key="risk_rate")
    as_table_expander(
        table.reset_index().assign(
            default_rate=lambda t: t["default_rate"].map(percent),
            mean_pd=lambda t: t["mean_pd"].map(percent),
            originated=lambda t: t["originated"].map(money),
            latest_balance=lambda t: t["latest_balance"].map(money),
            accounts=lambda t: t["accounts"].map("{:,.0f}".format),
            defaults=lambda t: t["defaults"].map("{:,.0f}".format),
        ).rename(columns={
            "risk_band": "Risk band", "accounts": "Accounts", "defaults": "Defaults", "mean_pd": "Predicted default rate",
            "originated": "Originated", "latest_balance": "Latest balance", "default_rate": "Observed default rate",
        }),
    )


def _by_dimension(frame: pd.DataFrame) -> None:
    st.markdown("**Default rate and exposure by dimension**")
    choice = st.selectbox("Dimension", list(DIMENSIONS), key="dimension_choice", label_visibility="collapsed")
    column = DIMENSIONS[choice]
    table = segment_summary(frame, column)
    average = frame["target_default"].mean()

    left, right = st.columns(2)
    hover = [
        f"{name}<br>{percent(row.default_rate)} default rate<br>{row.defaults:,.0f} of {row.accounts:,.0f} accounts<br>"
        f"{money(row.originated)} originated ({percent(row.share_of_originated)})"
        for name, row in table.iterrows()
    ]
    with left:
        st.caption(f"Default rate. The line marks the {percent(average)} average for this selection.")
        fig = horizontal_bars(
            list(table.index), table["default_rate"], [percent(v) for v in table["default_rate"]],
            hover=hover, x_format=".0%", reference=average,
        )
        show_chart(fig, key="dimension_rate")
    with right:
        st.caption("Originated exposure.")
        fig = horizontal_bars(
            list(table.index), table["originated"], [money(v) for v in table["originated"]],
            hover=hover, x_format="$~s",
        )
        show_chart(fig, key="dimension_exposure")
    as_table_expander(
        table.reset_index().assign(
            default_rate=lambda t: t["default_rate"].map(lambda v: percent(v, 2)),
            originated=lambda t: t["originated"].map(money),
            defaulted_amount=lambda t: t["defaulted_amount"].map(money),
            latest_balance=lambda t: t["latest_balance"].map(money),
            share_of_originated=lambda t: t["share_of_originated"].map(percent),
            share_of_defaulted_amount=lambda t: t["share_of_defaulted_amount"].map(percent),
        ).rename(columns={
            column: choice, "accounts": "Accounts", "defaults": "Defaults", "originated": "Originated",
            "defaulted_amount": "Defaulted amount", "latest_balance": "Latest balance", "default_rate": "Default rate",
            "share_of_originated": "Share of originated", "share_of_defaulted_amount": "Share of defaulted amount",
        }),
    )


def _heatmap(frame: pd.DataFrame) -> None:
    st.markdown("**Cross-dimensional default rate**")
    st.caption(f"Darker means a higher default rate. Grey cells have fewer than {MIN_SEGMENT_ACCOUNTS} accounts, so no rate is shown.")
    left, right = st.columns(2)
    names = [n for n in DIMENSIONS if n not in ("Loan purpose", "Sector", "Risk band")]
    rows_name = left.selectbox("Rows", names, index=names.index("Employment"), key="heat_rows")
    other = [n for n in names if n != rows_name]
    default_column = "Income band" if "Income band" in other else other[0]
    columns_name = right.selectbox("Columns", other, index=other.index(default_column), key="heat_columns")

    rates, counts = cross_tab(frame, DIMENSIONS[rows_name], DIMENSIONS[columns_name])
    visible = rates.where(counts >= MIN_SEGMENT_ACCOUNTS)
    hover = [
        [
            (f"{r} / {c}<br>{percent(rates.loc[r, c])} default rate<br>{counts.loc[r, c]:,} accounts"
             if counts.loc[r, c] >= MIN_SEGMENT_ACCOUNTS
             else f"{r} / {c}<br>{counts.loc[r, c]:,} accounts: too few to show")
            for c in rates.columns
        ]
        for r in rates.index
    ]
    text = [[percent(v, 1) if pd.notna(v) else "" for v in row] for row in visible.to_numpy()]
    fig = go.Figure(
        go.Heatmap(
            z=visible.to_numpy(), x=[str(c) for c in rates.columns], y=[str(r) for r in rates.index],
            colorscale=HEATMAP_SCALE, text=text, texttemplate="%{text}", hovertext=hover, hoverinfo="text",
            xgap=2, ygap=2, colorbar=dict(title="Default rate", tickformat=".0%", thickness=12, outlinewidth=0),
        )
    )
    fig.update_yaxes(autorange="reversed", type="category", showgrid=False)
    fig.update_xaxes(type="category", side="top", showgrid=False)
    style_figure(fig, height=max(260, 80 + 52 * len(rates.index)))
    fig.update_layout(plot_bgcolor=NEUTRAL, margin=dict(t=30))
    show_chart(fig, key="cross_heatmap")


def _trends(frame: pd.DataFrame, repayment: pd.DataFrame) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("**Default rate by origination year**")
        split_name = st.selectbox("Split by", list(TREND_SPLITS), key="trend_split", label_visibility="collapsed")
        split = TREND_SPLITS[split_name]
        fig = go.Figure()
        if split is None:
            grouped = frame.groupby("origination_year").agg(rate=("target_default", "mean"), predicted=("pd_score", "mean"), n=("account_id", "size"))
            fig.add_scatter(x=grouped.index, y=grouped["rate"], mode="lines+markers", name="Observed", line=dict(color=BLUE, width=2), marker=dict(size=8, line=dict(color="#fcfcfb", width=2)),
                            customdata=grouped["n"], hovertemplate="%{x}: %{y:.1%} observed<br>%{customdata:,} accounts<extra></extra>")
            fig.add_scatter(x=grouped.index, y=grouped["predicted"], mode="lines+markers", name="Predicted by the model", line=dict(color=SERIES[1], width=2), marker=dict(size=8, line=dict(color="#fcfcfb", width=2)),
                            hovertemplate="%{x}: %{y:.1%} predicted<extra></extra>")
        else:
            # Each value keeps its colour whatever the filters leave on screen.
            everything = sorted(st.session_state["_all_values"][split])
            colours = {value: SERIES[i % len(SERIES)] for i, value in enumerate(everything)}
            for value, part in frame.groupby(split):
                grouped = part.groupby("origination_year").agg(rate=("target_default", "mean"), n=("account_id", "size"))
                fig.add_scatter(x=grouped.index, y=grouped["rate"], mode="lines+markers", name=str(value), line=dict(color=colours[value], width=2), marker=dict(size=8, line=dict(color="#fcfcfb", width=2)),
                                customdata=grouped["n"], hovertemplate=f"{value}<br>%{{x}}: %{{y:.1%}}<br>%{{customdata:,}} accounts<extra></extra>")
        fig.update_yaxes(tickformat=".0%", rangemode="tozero", title_text="Default rate")
        fig.update_xaxes(dtick=1, showgrid=False)
        style_figure(fig, height=340, show_legend=True)
        show_chart(fig, key="vintage_trend")
    with right:
        st.markdown("**Delinquency by month since origination**")
        st.caption("Share of accounts late or missed in each of the 12 repayment months.")
        curve = month_on_book_curve(repayment, frame)
        fig = go.Figure()
        fig.add_scatter(x=curve.index, y=curve["late_rate"], mode="lines+markers", name="Late", line=dict(color=BLUE, width=2), marker=dict(size=8, line=dict(color="#fcfcfb", width=2)), hovertemplate="Month %{x}: %{y:.1%} late<extra></extra>")
        fig.add_scatter(x=curve.index, y=curve["missed_rate"], mode="lines+markers", name="Missed", line=dict(color=SERIES[1], width=2), marker=dict(size=8, line=dict(color="#fcfcfb", width=2)), hovertemplate="Month %{x}: %{y:.1%} missed<extra></extra>")
        fig.update_yaxes(tickformat=".0%", rangemode="tozero", title_text="Share of accounts")
        fig.update_xaxes(dtick=1, showgrid=False, title_text="Month since origination")
        style_figure(fig, height=340, show_legend=True)
        show_chart(fig, key="mob_curve")


def _insights(insights: list[dict]) -> None:
    st.subheader("What the portfolio is telling us")
    st.caption("Written for the whole portfolio, so these do not change with the filters. Every figure is computed at build time.")
    first_row = st.columns(3)
    second_row = st.columns(2)
    for position, (column, insight) in enumerate(zip([*first_row, *second_row], insights), start=1):
        column.markdown(
            text_card(insight["title"], insight["finding"], insight["so_what"], eyebrow=f"Insight {position}"),
            unsafe_allow_html=True,
        )
        column.write("")


def _account_table(frame: pd.DataFrame) -> None:
    st.markdown("**Highest-risk accounts in this selection**")
    st.caption("Select a row to open that account.")
    top = frame.sort_values("pd_score", ascending=False).head(200)
    display = pd.DataFrame(
        {
            "Account": top["account_id"],
            "Loan type": top["loan_type"],
            "Employment": top["employment_type"],
            "Income band": top["income_band"],
            "Latest balance": top["latest_balance"],
            "Predicted default": top["pd_score"] * 100,
            "Risk band": top["risk_band"],
            "Outcome": top["default_status"],
            "In review list": top["flagged"].map({1: "Yes", 0: "No"}),
        }
    )
    event = st.dataframe(
        display, width="stretch", hide_index=True, height=320, on_select="rerun", selection_mode="single-row", key="portfolio_accounts",
        column_config={
            "Latest balance": st.column_config.NumberColumn(format="$%,.0f"),
            "Predicted default": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )
    if event.selection.rows:
        open_account(display.iloc[event.selection.rows[0]]["Account"])


def render(context: dict) -> None:
    frame = context["filtered"]
    portfolio = context["portfolio"]
    st.session_state["_all_values"] = {split: portfolio[split].unique().tolist() for split in TREND_SPLITS.values() if split}

    st.title("Portfolio")
    if context["active_filters"]:
        st.caption(f"Showing {len(frame):,} of {len(portfolio):,} accounts. " + plain(" · ".join(context["active_filters"])))
    else:
        st.caption(f"Showing all {len(portfolio):,} accounts. Use the filters in the sidebar to narrow this down.")

    if frame.empty:
        st.warning("No accounts match these filters. Remove one to continue.")
        return

    _kpis(frame)
    st.divider()
    _insights(context["insights"])
    st.divider()
    _risk_bands(frame)
    st.divider()
    _by_dimension(frame)
    st.divider()
    _heatmap(frame)
    st.divider()
    _trends(frame, context["repayment"])
    st.divider()
    _account_table(frame)
