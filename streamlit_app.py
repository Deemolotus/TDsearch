"""Streamlit app: multi-indicator US / Canada equity scanner."""
from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Literal

import pandas as pd
import streamlit as st

from market_lists import (
    CA_MODE_OPTIONS,
    US_MODE_OPTIONS,
    market_defaults,
    resolve_tickers,
)
from scan_runner import ScanParams, ScanResult, ScanUpdate, run_scan
from td_scanner_core import ScoreWeights

MarketKey = Literal["us", "ca"]

SIGNAL_COLORS = {
    "TD9": "color: #4ade80",
    "TD13": "color: #fb923c",
    "TD15": "color: #ffd700",
}

CSV_KEYS = [
    "ticker", "source", "signal", "sig_date", "days_ago", "bars_ago", "price",
    "avg_vol", "cmf", "cmf_turn_pos", "sqz", "sqz_turn_pos", "supertrend_bull",
    "st_line", "cycle_up", "cycle_slope", "cycle_r2", "cycle_strength",
    "cycle_error", "score", "score_td", "score_cmf", "score_sqz", "score_st",
    "score_cycle", "reasons", "missing",
]


def _init_session_state() -> None:
    for key in ("us", "ca"):
        st.session_state.setdefault(f"{key}_stop", False)
        st.session_state.setdefault(f"{key}_list_cache", {})
        st.session_state.setdefault(f"{key}_results", [])
        st.session_state.setdefault(f"{key}_stats", None)
        st.session_state.setdefault(f"{key}_scanning", False)
        st.session_state.setdefault(f"{key}_last_summary", "")


def _format_eta(done: int, total: int, elapsed: float) -> str:
    if done <= 10 or total <= done:
        return ""
    remaining = elapsed / done * (total - done)
    return f"ETA ~ {int(remaining // 60)}m{int(remaining % 60):02d}s"


def _results_to_csv(results: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_KEYS, extrasaction="ignore")
    writer.writeheader()
    for row in results:
        writer.writerow({k: row.get(k, "") for k in CSV_KEYS})
    return buf.getvalue()


def _style_results_df(df: pd.DataFrame):
    def _color_signal(val: str) -> str:
        return SIGNAL_COLORS.get(str(val), "")

    if df.empty:
        return df
    return df.style.map(_color_signal, subset=["Signal"])


def _render_results(market: MarketKey, price_label: str) -> None:
    results = st.session_state[f"{market}_results"]
    stats = st.session_state[f"{market}_stats"]
    summary = st.session_state.get(f"{market}_last_summary", "")

    if summary:
        st.info(summary)

    if not results:
        st.caption("No candidates yet. Run a scan to see ranked results.")
        return

    rows = []
    for i, r in enumerate(results, 1):
        rows.append({
            "Rank": i,
            "Score": r.get("score", ""),
            "Ticker": r["ticker"],
            "Source": r.get("source", ""),
            "Signal": r["signal"],
            "Signal Date": r["sig_date"],
            "Bars Ago": r.get("bars_ago", r.get("days_ago", "")),
            f"Price ({price_label})": r["price"],
            "CMF": r.get("cmf", ""),
            "CMF−→+": r.get("cmf_turn_pos", ""),
            "SQZ": r.get("sqz", ""),
            "SQZ−→+": r.get("sqz_turn_pos", ""),
            "ST↑": r.get("supertrend_bull", ""),
            "Cycle↑": r.get("cycle_up", ""),
            "Cycle R²": r.get("cycle_r2", ""),
            "Reasons": r.get("reasons", ""),
            "Missing": r.get("missing", ""),
        })
    df = pd.DataFrame(rows)
    st.dataframe(_style_results_df(df), use_container_width=True, hide_index=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    prefix = "multi_scan_ca" if market == "ca" else "multi_scan_us"
    st.download_button(
        label="Download CSV",
        data=_results_to_csv(results),
        file_name=f"{prefix}_{stamp}.csv",
        mime="text/csv",
        key=f"{market}_download",
    )

    if stats is not None:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Download failures", len(stats.failed_download))
        c2.metric("Analyse failures", len(stats.failed_analyse))
        c3.metric("Liquidity filtered", stats.liquidity_skipped)
        c4.metric("No TD filtered", getattr(stats, "no_td_skipped", 0))
        c5.metric("Cycle attempted", getattr(stats, "cycle_attempted", 0))
        c6.metric("Candidates", len(results))


def _run_scan_for_market(market: MarketKey, params: ScanParams, tickers: list[str]) -> None:
    stop_key = f"{market}_stop"
    st.session_state[stop_key] = False
    st.session_state[f"{market}_scanning"] = True
    st.session_state[f"{market}_results"] = []
    st.session_state[f"{market}_stats"] = None
    st.session_state[f"{market}_last_summary"] = ""

    progress = st.progress(0.0, text="Starting scan…")
    status = st.empty()

    final: ScanResult | None = None
    for update in run_scan(tickers, params, lambda: st.session_state.get(stop_key, False)):
        if isinstance(update, ScanUpdate):
            pct = update.done / max(update.total, 1)
            eta = _format_eta(update.done, update.total, update.elapsed_sec)
            phase_label = "Stage 1 (TD+indicators)" if update.phase == "stage1" else "Stage 2 (cycle)"
            progress.progress(
                pct,
                text=(
                    f"{phase_label}: {update.done}/{update.total}  |  "
                    f"candidates: {update.found_count}  {eta}"
                ),
            )
            status.caption(
                f"Elapsed {int(update.elapsed_sec // 60)}m{int(update.elapsed_sec % 60):02d}s"
            )
        else:
            final = update

    progress.empty()
    status.empty()
    st.session_state[f"{market}_scanning"] = False

    if final is None:
        return

    st.session_state[f"{market}_results"] = final.results
    st.session_state[f"{market}_stats"] = final.stats
    stopped = st.session_state.get(stop_key, False)
    st_label = "Stopped" if stopped else "Complete"
    st.session_state[f"{market}_last_summary"] = (
        f"{st_label} — {len(final.results)} candidates  |  "
        f"download failed {len(final.stats.failed_download)}  |  "
        f"analyse failed {len(final.stats.failed_analyse)}  |  "
        f"liquidity {final.stats.liquidity_skipped}  |  "
        f"no TD {final.stats.no_td_skipped}  |  "
        f"cycle {final.stats.cycle_attempted} "
        f"(failed {final.stats.cycle_failed})  |  "
        f"elapsed {int(final.elapsed_sec // 60)}m{int(final.elapsed_sec % 60):02d}s"
    )


def render_scanner_tab(market: MarketKey) -> None:
    mode_options = US_MODE_OPTIONS if market == "us" else CA_MODE_OPTIONS
    default_price, default_vol, price_label = market_defaults(market)
    cache_key = f"{market}_list_cache"
    stop_key = f"{market}_stop"
    scanning = st.session_state.get(f"{market}_scanning", False)

    if market == "ca":
        st.caption("TSX tickers use `.TO`, TSXV tickers use `.V` (yfinance format).")
    else:
        st.caption(
            "Hard filters: liquidity + TD buy in window. Soft score: CMF −→+, "
            "SQZ (拥挤值) −→+, SuperTrend↑, cycle projection↑. "
            "CMF is a price/volume proxy, not true institutional flow."
        )

    with st.expander("Scan settings", expanded=True):
        mode = st.radio(
            "Scan scope",
            options=list(mode_options.keys()),
            format_func=lambda k: mode_options[k],
            horizontal=True,
            key=f"{market}_mode",
            disabled=scanning,
        )

        custom_text = ""
        if mode == "custom":
            placeholder = (
                "SHOP.TO, ATH.V"
                if market == "ca"
                else "AAPL, MSFT, NVDA"
            )
            custom_text = st.text_input(
                "Custom tickers (comma-separated)",
                placeholder=placeholder,
                key=f"{market}_custom",
                disabled=scanning,
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            signal_days = st.number_input(
                "TD window (trading days ≈ 2 weeks)",
                min_value=1,
                max_value=60,
                value=10,
                key=f"{market}_signal_days",
                disabled=scanning,
            )
            min_price = st.number_input(
                f"Min price ({price_label})",
                min_value=0.0,
                value=float(default_price),
                step=0.5,
                key=f"{market}_min_price",
                disabled=scanning,
            )
            turn_lookback = st.number_input(
                "CMF / SQZ turn lookback (bars)",
                min_value=2,
                max_value=40,
                value=10,
                key=f"{market}_turn_lookback",
                disabled=scanning,
            )
        with c2:
            min_vol = st.number_input(
                "Min avg daily volume",
                min_value=0,
                value=int(default_vol),
                step=10_000,
                key=f"{market}_min_vol",
                disabled=scanning,
            )
            batch_size = st.number_input(
                "Batch size",
                min_value=5,
                max_value=100,
                value=20,
                key=f"{market}_batch_size",
                disabled=scanning,
            )
            cycle_shortlist = st.number_input(
                "Cycle shortlist size",
                min_value=5,
                max_value=200,
                value=50,
                key=f"{market}_cycle_shortlist",
                disabled=scanning,
            )
        with c3:
            batch_delay = st.number_input(
                "Batch delay (seconds)",
                min_value=0.1,
                max_value=5.0,
                value=0.4,
                step=0.1,
                key=f"{market}_batch_delay",
                disabled=scanning,
            )
            liquidity_filter = st.checkbox(
                "Liquidity filter (latest bar)",
                value=True,
                key=f"{market}_liquidity",
                disabled=scanning,
            )
            enable_cycle = st.checkbox(
                "Stage-2 cycle ranking",
                value=True,
                key=f"{market}_enable_cycle",
                disabled=scanning,
            )

    with st.expander("Score weights (sum normalized to 100)", expanded=False):
        w1, w2, w3, w4, w5 = st.columns(5)
        with w1:
            w_td = st.number_input("TD", min_value=0.0, max_value=100.0, value=25.0, key=f"{market}_w_td", disabled=scanning)
        with w2:
            w_cmf = st.number_input("CMF", min_value=0.0, max_value=100.0, value=20.0, key=f"{market}_w_cmf", disabled=scanning)
        with w3:
            w_sqz = st.number_input("SQZ", min_value=0.0, max_value=100.0, value=20.0, key=f"{market}_w_sqz", disabled=scanning)
        with w4:
            w_st = st.number_input("SuperTrend", min_value=0.0, max_value=100.0, value=15.0, key=f"{market}_w_st", disabled=scanning)
        with w5:
            w_cycle = st.number_input("Cycle", min_value=0.0, max_value=100.0, value=20.0, key=f"{market}_w_cycle", disabled=scanning)

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
    with btn_col1:
        start = st.button(
            "Start scan",
            type="primary",
            key=f"{market}_start",
            disabled=scanning,
        )
    with btn_col2:
        if st.button("Stop", key=f"{market}_stop_btn", disabled=not scanning):
            st.session_state[stop_key] = True
            st.rerun()
    with btn_col3:
        if st.button("Refresh list cache", key=f"{market}_refresh", disabled=scanning):
            st.session_state[cache_key] = {}
            st.success("Exchange list cache cleared.")

    if start:
        status_box = st.empty()

        def _status(msg: str) -> None:
            status_box.info(msg)

        tickers, sources = resolve_tickers(
            market,
            mode,
            custom_text,
            st.session_state[cache_key],
            status_callback=_status,
        )
        status_box.empty()

        if not tickers:
            st.warning("No tickers to scan. Choose a scope or enter custom tickers.")
        else:
            params = ScanParams(
                signal_days=int(signal_days),
                min_price=float(min_price),
                min_avg_vol=float(min_vol),
                batch_size=int(batch_size),
                batch_delay=float(batch_delay),
                apply_liquidity_filter=liquidity_filter,
                turn_lookback=int(turn_lookback),
                cycle_shortlist=int(cycle_shortlist),
                enable_cycle=enable_cycle,
                weights=ScoreWeights(
                    td=float(w_td),
                    cmf=float(w_cmf),
                    sqz=float(w_sqz),
                    supertrend=float(w_st),
                    cycle=float(w_cycle),
                ),
                source_by_ticker=sources,
            )
            _run_scan_for_market(market, params, tickers)
            st.rerun()

    st.divider()
    st.subheader("Ranked candidates")
    _render_results(market, price_label)


def main() -> None:
    st.set_page_config(
        page_title="Multi-Indicator Scanner",
        page_icon="📈",
        layout="wide",
    )
    _init_session_state()

    st.title("Multi-Indicator Equity Scanner")
    st.markdown(
        "Hybrid screener: **hard-filter** on liquidity + recent **TD9/13/15** buy, "
        "then **rank** by CMF turn (proxy money flow), Squeeze Momentum turn (拥挤值), "
        "SuperTrend, and cycle projection. Sorted by composite score (0–100)."
    )

    tab_us, tab_ca = st.tabs(["US Scanner", "Canada Scanner"])
    with tab_us:
        render_scanner_tab("us")
    with tab_ca:
        render_scanner_tab("ca")


if __name__ == "__main__":
    main()
