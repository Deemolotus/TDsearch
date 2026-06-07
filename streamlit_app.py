"""Streamlit app for US and Canada TD buy signal scanners."""
from __future__ import annotations

import csv
import io
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from market_lists import (
    CA_MODE_OPTIONS,
    US_MODE_OPTIONS,
    Market,
    market_defaults,
    resolve_tickers,
)
from scan_runner import ScanParams, ScanResult, ScanUpdate, run_scan

MarketKey = Literal["us", "ca"]

SIGNAL_COLORS = {
    "TD9": "color: #4ade80",
    "TD13": "color: #fb923c",
    "TD15": "color: #ffd700",
}


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
    keys = ["ticker", "signal", "sig_date", "days_ago", "price"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys)
    writer.writeheader()
    for row in results:
        writer.writerow({k: row[k] for k in keys})
    return buf.getvalue()


def _style_results_df(df: pd.DataFrame) -> pd.DataFrame:
    def _color_signal(val: str) -> str:
        return SIGNAL_COLORS.get(str(val), "")

    if df.empty:
        return df
    styled = df.style.map(_color_signal, subset=["Signal"])
    return styled


def _render_results(market: MarketKey, price_label: str) -> None:
    results = st.session_state[f"{market}_results"]
    stats = st.session_state[f"{market}_stats"]
    summary = st.session_state.get(f"{market}_last_summary", "")

    if summary:
        st.info(summary)

    if not results:
        st.caption("No signals yet. Run a scan to see results.")
        return

    rows = []
    for i, r in enumerate(results, 1):
        rows.append({
            "Rank": i,
            "Ticker": r["ticker"],
            "Signal": r["signal"],
            "Signal Date": r["sig_date"],
            "Days Ago": r["days_ago"],
            f"Price ({price_label})": r["price"],
        })
    df = pd.DataFrame(rows)
    st.dataframe(_style_results_df(df), use_container_width=True, hide_index=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    prefix = "td_buy_ca" if market == "ca" else "td_buy"
    st.download_button(
        label="Download CSV",
        data=_results_to_csv(results),
        file_name=f"{prefix}_{stamp}.csv",
        mime="text/csv",
        key=f"{market}_download",
    )

    if stats is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Download failures", len(stats.failed_download))
        c2.metric("Analyse failures", len(stats.failed_analyse))
        c3.metric("Liquidity filtered", stats.liquidity_skipped)
        c4.metric("Signals found", len(results))


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
            progress.progress(
                pct,
                text=f"Scanning {update.done}/{update.total}  |  signals: {update.found_count}  {eta}",
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
        f"{st_label} — {len(final.results)} signals  |  "
        f"download failed {len(final.stats.failed_download)}  |  "
        f"analyse failed {len(final.stats.failed_analyse)}  |  "
        f"liquidity filtered {final.stats.liquidity_skipped}  |  "
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
                "Signal window (trading days)",
                min_value=1,
                max_value=60,
                value=20,
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
                "Liquidity filter (signal day)",
                value=True,
                key=f"{market}_liquidity",
                disabled=scanning,
            )

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

        tickers = resolve_tickers(
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
            )
            _run_scan_for_market(market, params, tickers)
            st.rerun()

    st.divider()
    st.subheader("Results")
    _render_results(market, price_label)


def main() -> None:
    st.set_page_config(
        page_title="TD Buy Scanner",
        page_icon="📈",
        layout="wide",
    )
    _init_session_state()

    st.title("TD Buy Signal Scanner")
    st.markdown(
        "Scan for **TD9 / TD13 / TD15** buy signals in the last N trading days. "
        "Results sorted by signal date (newest first)."
    )

    tab_us, tab_ca = st.tabs(["US Scanner", "Canada Scanner"])
    with tab_us:
        render_scanner_tab("us")
    with tab_ca:
        render_scanner_tab("ca")


if __name__ == "__main__":
    main()
