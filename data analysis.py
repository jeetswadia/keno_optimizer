"""
Massachusetts Keno - Interactive Analysis Dashboard
====================================================
Loads historical MA Keno draw data, dynamically reconstructs precise draw times
(400 draws/day starting at 5:04 AM, 3 minutes apart), and provides:
  - Sidebar filters (date range, day of week, hour of day)
  - Frequency analysis with optional weekly trend at a specific hour
  - ROI / Backtesting calculator using official MA Keno payout tables

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DRAWS_PER_DAY = 400
FIRST_DRAW_TIME = "05:04:00"        # 5:04 AM ET
DRAW_INTERVAL_MINUTES = 3            # 3 min between draws -> 400th @ 1:01 AM next day
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]

# Default file path (kept for convenience; users can also upload)
DEFAULT_DATA_PATH = "Keno_data_year.xlsx - getDrawsByDateRange_startDate=2.csv"

# ---------------------------------------------------------------------------
# Official MA Keno payout tables (base $1 wager)
# Source: MA State Lottery / lottery.net published prize tables
# Key = spots played, value = {numbers_matched: prize_in_dollars}
# Note: 10, 11, 12-spot games include a "Match 0" prize.
# ---------------------------------------------------------------------------

PAYOUTS = {
    1:  {1: 2.50},
    2:  {2: 5, 1: 1},
    3:  {3: 25, 2: 2.50},
    4:  {4: 100, 3: 4, 2: 1},
    5:  {5: 450, 4: 20, 3: 2},
    6:  {6: 1600, 5: 50, 4: 7, 3: 1},
    7:  {7: 5000, 6: 100, 5: 20, 4: 3, 3: 1},
    8:  {8: 15000, 7: 1000, 6: 50, 5: 10, 4: 2},
    9:  {9: 40000, 8: 4000, 7: 200, 6: 25, 5: 5, 4: 1},
    10: {10: 100000, 9: 10000, 8: 500, 7: 80, 6: 20, 5: 2, 0: 2},
    11: {11: 500000, 10: 15000, 9: 1500, 8: 250, 7: 50, 6: 10, 5: 1, 0: 2},
    12: {12: 1000000, 11: 25000, 10: 2500, 9: 1000, 8: 150, 7: 25, 6: 5, 0: 4},
}

# Bonus is NOT available for 10/11/12 spot games per MA rules.
BONUS_ELIGIBLE_SPOTS = set(range(1, 10))  # 1-9 only

# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading and processing draw data...")
def load_and_prep(file_bytes_or_path) -> pd.DataFrame:
    """
    Load the raw Keno CSV and compute precise per-draw timestamps.

    Logic:
      * Group rows by drawDate.
      * Sort each group by drawNumber ascending.
      * Assign drawTime = drawDate @ 05:04:00 + (position_in_day * 3 minutes).
        With 400 draws @ 3 min spacing, the last draw lands at the next day's 01:01 AM.
      * Derive Hour (0-23) and DayOfWeek (Monday-Sunday) from drawTime.
      * Pre-parse winningNumbers into a list[int] column for fast downstream use.

    Accepts either a filesystem path or an uploaded-file BytesIO/buffer.
    """
    df = pd.read_csv(file_bytes_or_path)

    # --- Defensive column normalization ---------------------------------
    df.columns = [c.strip() for c in df.columns]
    required = {"drawNumber", "bonus", "drawDate", "winningNumbers"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # --- Type coercion --------------------------------------------------
    df["drawNumber"] = pd.to_numeric(df["drawNumber"], errors="coerce").astype("Int64")
    df["bonus"] = pd.to_numeric(df["bonus"], errors="coerce").fillna(1).astype(int)
    df["drawDate"] = pd.to_datetime(df["drawDate"], errors="coerce").dt.normalize()

    df = df.dropna(subset=["drawNumber", "drawDate", "winningNumbers"]).copy()
    df["drawNumber"] = df["drawNumber"].astype(int)

    # --- Build drawTime per the 5:04 AM + n*3min rule -------------------
    df = df.sort_values(["drawDate", "drawNumber"]).reset_index(drop=True)

    # Position of the row within its day (0-indexed)
    df["_pos"] = df.groupby("drawDate").cumcount()

    base_offset = pd.Timedelta(hours=5, minutes=4)
    df["drawTime"] = (
        df["drawDate"]
        + base_offset
        + pd.to_timedelta(df["_pos"] * DRAW_INTERVAL_MINUTES, unit="m")
    )
    df = df.drop(columns="_pos")

    # --- Time features --------------------------------------------------
    df["Hour"] = df["drawTime"].dt.hour.astype(int)
    df["DayOfWeek"] = df["drawTime"].dt.day_name()

    # --- Parse winning numbers once -------------------------------------
    def _parse_nums(s):
        if isinstance(s, list):
            return s
        try:
            return [int(x) for x in str(s).split(",") if str(x).strip()]
        except Exception:
            return []

    df["winningNumbersList"] = df["winningNumbers"].map(_parse_nums)

    return df


# ---------------------------------------------------------------------------
# Analytical helpers
# ---------------------------------------------------------------------------

def number_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with 'Number' and 'Frequency' across all draws in df."""
    counter: Counter = Counter()
    for nums in df["winningNumbersList"]:
        counter.update(nums)
    freq = (
        pd.DataFrame({"Number": list(counter.keys()),
                      "Frequency": list(counter.values())})
        .sort_values("Number")
        .reset_index(drop=True)
    )
    return freq


def weekly_hour_trend(df: pd.DataFrame, hour: int, top_k: int = 10) -> pd.DataFrame:
    """
    For a given hour-of-day, return frequency of the top_k overall numbers
    broken down by day-of-week. Long-format: Number / DayOfWeek / Frequency.
    """
    sub = df[df["Hour"] == hour]
    if sub.empty:
        return pd.DataFrame(columns=["Number", "DayOfWeek", "Frequency"])

    overall_counter: Counter = Counter()
    for nums in sub["winningNumbersList"]:
        overall_counter.update(nums)
    top_numbers = [n for n, _ in overall_counter.most_common(top_k)]

    rows = []
    for day in DAY_ORDER:
        day_sub = sub[sub["DayOfWeek"] == day]
        day_counter: Counter = Counter()
        for nums in day_sub["winningNumbersList"]:
            day_counter.update(nums)
        for n in top_numbers:
            rows.append({"Number": n, "DayOfWeek": day,
                         "Frequency": int(day_counter.get(n, 0))})
    out = pd.DataFrame(rows)
    out["DayOfWeek"] = pd.Categorical(out["DayOfWeek"], categories=DAY_ORDER, ordered=True)
    return out


def simulate_roi(
    df: pd.DataFrame,
    selected_numbers: list[int],
    n_draws: int,
) -> dict:
    """
    Backtest selected_numbers against the first `n_draws` rows of df.

    Wager logic:
      * Spots 1-9 (Bonus eligible): $1 base + $1 bonus = $2/draw.
        Base prize multiplied by historical 'bonus' multiplier for that draw.
      * Spots 10-12 (Bonus ineligible): $1 base only = $1/draw. No multiplier.

    Returns a dict with totals, ROI, hit histogram, and per-draw details.
    """
    spots = len(selected_numbers)
    selected_set = set(selected_numbers)
    payout_table = PAYOUTS.get(spots, {})
    bonus_eligible = spots in BONUS_ELIGIBLE_SPOTS

    cost_per_draw = 2.0 if bonus_eligible else 1.0

    # Take the first n draws of the (already filtered, time-sorted) dataset
    sample = df.head(n_draws)
    actual_n = len(sample)

    total_wagered = actual_n * cost_per_draw
    total_won = 0.0
    hit_counter: Counter = Counter()
    biggest_win = 0.0

    for _, row in sample.iterrows():
        winning_set = set(row["winningNumbersList"])
        matches = len(selected_set & winning_set)
        hit_counter[matches] += 1

        base_prize = payout_table.get(matches, 0)
        if base_prize == 0:
            continue

        if bonus_eligible:
            mult = int(row["bonus"]) if pd.notna(row["bonus"]) else 1
            # Per MA rules the bonus multiplier replaces (not stacks with) the
            # base prize when a Bonus number is drawn. The 'bonus' column in
            # the official feed is 1 when no multiplier hit, else 3/4/5/10.
            mult = mult if mult >= 1 else 1
            prize = base_prize * mult
        else:
            prize = base_prize

        total_won += prize
        biggest_win = max(biggest_win, prize)

    net = total_won - total_wagered
    roi_pct = (net / total_wagered * 100.0) if total_wagered > 0 else 0.0

    return {
        "spots": spots,
        "bonus_eligible": bonus_eligible,
        "cost_per_draw": cost_per_draw,
        "actual_n": actual_n,
        "total_wagered": total_wagered,
        "total_won": total_won,
        "net": net,
        "roi_pct": roi_pct,
        "biggest_win": biggest_win,
        "hits": dict(sorted(hit_counter.items(), reverse=True)),
        "payout_table": payout_table,
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="MA Keno Analysis Dashboard", layout="wide",
                   page_icon="🎰")

st.title("🎰 Massachusetts Keno - Analysis & ROI Dashboard")
st.caption("Historical frequency analysis and backtesting on real MA Keno draws. "
           "For research/entertainment only - past performance does not predict "
           "future draws.")

# --- Data source -----------------------------------------------------------
with st.sidebar:
    st.header("📁 Data Source")
    uploaded = st.file_uploader("Upload Keno CSV", type=["csv"])
    use_default = st.checkbox(
        "Use default path", value=(uploaded is None),
        help=f"Looks for `{DEFAULT_DATA_PATH}` in the working directory.",
    )

# Load data
df = None
load_err = None
try:
    if uploaded is not None:
        df = load_and_prep(uploaded)
    elif use_default and Path(DEFAULT_DATA_PATH).exists():
        df = load_and_prep(DEFAULT_DATA_PATH)
    else:
        st.info("Upload a Keno CSV in the sidebar to begin.")
        st.stop()
except Exception as e:
    load_err = str(e)
    st.error(f"Failed to load data: {load_err}")
    st.stop()

# --- Sidebar filters -------------------------------------------------------
with st.sidebar:
    st.header("🔎 Filters")

    min_date = df["drawDate"].min().date()
    max_date = df["drawDate"].max().date()
    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        d_start, d_end = date_range
    else:
        d_start, d_end = min_date, max_date

    days_selected = st.multiselect(
        "Day of week",
        options=DAY_ORDER,
        default=DAY_ORDER,
    )

    hour_mode = st.radio(
        "Hour of day",
        options=["All hours", "Specific hour"],
        horizontal=True,
    )
    specific_hour = None
    if hour_mode == "Specific hour":
        specific_hour = st.slider("Hour (0-23)", 0, 23, 17)

# --- Apply filters ---------------------------------------------------------
mask = (
    (df["drawDate"].dt.date >= d_start)
    & (df["drawDate"].dt.date <= d_end)
    & (df["DayOfWeek"].isin(days_selected if days_selected else DAY_ORDER))
)
if specific_hour is not None:
    mask &= (df["Hour"] == specific_hour)

filtered = df.loc[mask].reset_index(drop=True)

# --- Top metrics -----------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total draws (raw)", f"{len(df):,}")
c2.metric("Filtered draws", f"{len(filtered):,}")
c3.metric("Date span", f"{min_date} → {max_date}")
c4.metric(
    "Avg bonus multiplier",
    f"{filtered['bonus'].mean():.2f}x" if len(filtered) else "—",
)

st.divider()

# --- Tabs ------------------------------------------------------------------
tab_freq, tab_roi, tab_data = st.tabs(
    ["📊 Frequency Analysis", "💰 ROI Calculator", "🗂 Data Preview"]
)

# ===========================================================================
# TAB 1: Frequency analysis
# ===========================================================================
with tab_freq:
    if filtered.empty:
        st.warning("No draws match your filters.")
    else:
        st.subheader("Number Frequency")

        freq_df = number_frequency(filtered)
        # Ensure all 1..80 are present even if zero
        all_nums = pd.DataFrame({"Number": list(range(1, 81))})
        freq_df = all_nums.merge(freq_df, on="Number", how="left").fillna(0)
        freq_df["Frequency"] = freq_df["Frequency"].astype(int)

        avg_freq = freq_df["Frequency"].mean()
        freq_df["DeltaFromAvg"] = freq_df["Frequency"] - avg_freq

        fig = px.bar(
            freq_df,
            x="Number",
            y="Frequency",
            color="DeltaFromAvg",
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            hover_data={"Number": True, "Frequency": True,
                        "DeltaFromAvg": ":.1f"},
            title=f"Draw frequency for numbers 1-80 "
                  f"({len(filtered):,} draws in current filter)",
        )
        fig.update_layout(xaxis=dict(dtick=5), height=420,
                          coloraxis_colorbar=dict(title="Δ vs avg"))
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**🔥 Hottest 10**")
            st.dataframe(
                freq_df.sort_values("Frequency", ascending=False).head(10)
                       .reset_index(drop=True),
                use_container_width=True,
            )
        with col_b:
            st.markdown("**❄️ Coldest 10**")
            st.dataframe(
                freq_df.sort_values("Frequency", ascending=True).head(10)
                       .reset_index(drop=True),
                use_container_width=True,
            )

        # Weekly trend chart - only when a specific hour is selected
        if specific_hour is not None:
            st.divider()
            st.subheader(
                f"📅 Weekly Trend at Hour {specific_hour:02d}:00 - "
                f"{(specific_hour % 12) or 12}"
                f"{' AM' if specific_hour < 12 else ' PM'}"
            )

            top_k = st.slider("Show top N most-drawn numbers at this hour",
                              5, 20, 10, key="topk_hour")
            trend_df = weekly_hour_trend(filtered, specific_hour, top_k=top_k)
            if trend_df.empty:
                st.info("No draws at this hour within the current filter.")
            else:
                # Heatmap view
                heat = alt.Chart(trend_df).mark_rect().encode(
                    x=alt.X("DayOfWeek:N", sort=DAY_ORDER, title="Day of week"),
                    y=alt.Y("Number:O", sort="-x", title="Number"),
                    color=alt.Color("Frequency:Q",
                                    scale=alt.Scale(scheme="viridis"),
                                    legend=alt.Legend(title="Hits")),
                    tooltip=["Number", "DayOfWeek", "Frequency"],
                ).properties(
                    height=max(300, 28 * top_k),
                    title=f"Top {top_k} numbers at hour {specific_hour}:00, "
                          f"by day of week",
                )
                st.altair_chart(heat, use_container_width=True)

                # Grouped bar view
                bar = alt.Chart(trend_df).mark_bar().encode(
                    x=alt.X("DayOfWeek:N", sort=DAY_ORDER, title=None),
                    y=alt.Y("Frequency:Q"),
                    color=alt.Color("DayOfWeek:N", sort=DAY_ORDER,
                                    legend=None),
                    column=alt.Column("Number:O", title="Number"),
                    tooltip=["Number", "DayOfWeek", "Frequency"],
                ).properties(width=70, height=180)
                st.altair_chart(bar, use_container_width=False)

# ===========================================================================
# TAB 2: ROI Calculator
# ===========================================================================
with tab_roi:
    st.subheader("💰 Historical ROI / Backtesting")
    st.caption("Picks are matched against draws from your *currently filtered* "
               "dataset (sidebar filters apply).")

    if filtered.empty:
        st.warning("No draws in current filter - widen your filters to backtest.")
    else:
        col_l, col_r = st.columns([2, 1])

        with col_l:
            picked = st.multiselect(
                "Select 1-12 numbers (your spots)",
                options=list(range(1, 81)),
                default=[3, 16, 22, 34, 47],
                max_selections=12,
            )
        with col_r:
            max_n = len(filtered)
            n_draws = st.number_input(
                f"Number of draws to simulate (max {max_n:,})",
                min_value=1,
                max_value=max_n,
                value=min(1000, max_n),
                step=100,
            )

        if not picked:
            st.info("Pick at least one number to run the backtest.")
        else:
            spots = len(picked)
            bonus_ok = spots in BONUS_ELIGIBLE_SPOTS

            st.markdown(
                f"**Spot game:** {spots}-spot &nbsp;|&nbsp; "
                f"**Bonus wager:** {'$1 (eligible)' if bonus_ok else '— (10/11/12-spot games are not Bonus eligible per MA rules)'} "
                f"&nbsp;|&nbsp; **Cost/draw:** "
                f"${2.00 if bonus_ok else 1.00:.2f}"
            )

            results = simulate_roi(filtered, picked, int(n_draws))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total wagered", f"${results['total_wagered']:,.2f}")
            m2.metric("Total won", f"${results['total_won']:,.2f}")
            m3.metric(
                "Net profit / loss",
                f"${results['net']:,.2f}",
                delta=f"{results['roi_pct']:+.2f}% ROI",
                delta_color="normal",
            )
            m4.metric("Biggest single win", f"${results['biggest_win']:,.2f}")

            st.markdown("#### 🎯 Match breakdown")
            hit_rows = []
            payout_table = results["payout_table"]
            for matched, count in sorted(results["hits"].items(), reverse=True):
                base = payout_table.get(matched, 0)
                hit_rows.append({
                    "Numbers matched": matched,
                    "Times hit": count,
                    "Base prize ($, per hit)": f"${base:,.2f}" if base else "—",
                    "Wins this bucket": "Yes" if base else "No",
                })
            hits_df = pd.DataFrame(hit_rows)
            st.dataframe(hits_df, use_container_width=True, hide_index=True)

            with st.expander(f"📋 Payout table for {spots}-spot game"):
                pt_rows = [
                    {"Match": k, "Base prize ($)": f"${v:,.2f}"}
                    for k, v in sorted(payout_table.items(), reverse=True)
                ]
                st.table(pd.DataFrame(pt_rows))
                if bonus_ok:
                    st.caption(
                        "Bonus column from the historical feed is applied as a "
                        "multiplier (1, 3, 4, 5, or 10) to the base prize on "
                        "each draw."
                    )
                else:
                    st.caption(
                        "Bonus is not offered on 10/11/12-spot games per MA rules."
                    )

            st.warning(
                "⚠️ Reminder: this backtest reflects what *would have* happened "
                "on these specific historical draws. Keno draws are independent "
                "and random; historical hot/cold patterns have no predictive "
                "value for future draws."
            )

# ===========================================================================
# TAB 3: Data preview
# ===========================================================================
with tab_data:
    st.subheader("Filtered data preview")
    st.caption(f"Showing first 500 of {len(filtered):,} filtered rows.")
    preview_cols = ["drawNumber", "drawDate", "drawTime", "Hour", "DayOfWeek",
                    "bonus", "winningNumbers"]
    st.dataframe(
        filtered[preview_cols].head(500),
        use_container_width=True,
        hide_index=True,
    )

    csv_buf = io.StringIO()
    filtered[preview_cols].to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download filtered data as CSV",
        data=csv_buf.getvalue(),
        file_name="E:\Radhika AIM\AI Project\Keno_data_year_april18.csv",
        mime="text/csv",
    )