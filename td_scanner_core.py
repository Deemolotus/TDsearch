"""
Shared TD buy scanner engine (US + Canada).
Aligned with nya_cycle_combined.download_ohlc / td_sequential.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

MAX_RETRIES = 3


@dataclass
class ScanStats:
    failed_download: list[str] = field(default_factory=list)
    failed_analyse: list[str] = field(default_factory=list)
    liquidity_skipped: int = 0


def _sq_col(series) -> pd.Series:
    s = series
    return (s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s).dropna()


def _is_multi_ticker_df(df: pd.DataFrame) -> bool:
    """yfinance may return MultiIndex columns even for a one-element ticker list."""
    return isinstance(df.columns, pd.MultiIndex)


def _parse_symbol_from_df(df: pd.DataFrame, sym: str) -> tuple[pd.Series, pd.Series, pd.Series] | None:
    """Extract and align Close/Low/Volume for one symbol (nya download_ohlc pattern)."""
    try:
        if _is_multi_ticker_df(df):
            close = df[sym]["Close"]
            low = df[sym]["Low"]
            vol = df[sym]["Volume"]
        else:
            close = df["Close"]
            low = df["Low"]
            vol = df["Volume"]
        close = _sq_col(close)
        if close.empty:
            return None
        idx = close.index
        low = _sq_col(low).reindex(idx)
        vol = _sq_col(vol).reindex(idx)
        return close, low, vol
    except (KeyError, TypeError, ValueError):
        return None


def _download_batch_once(
    tickers: list[str], lookback: int,
) -> dict[str, tuple[pd.Series, pd.Series, pd.Series]]:
    """Single yfinance batch attempt; returns successfully parsed symbols only."""
    end = pd.Timestamp.today().normalize()
    start = (end - pd.offsets.BDay(lookback + 15)).date().isoformat()
    end_str = (end + pd.offsets.BDay(1)).date().isoformat()
    result: dict[str, tuple[pd.Series, pd.Series, pd.Series]] = {}

    for attempt in range(MAX_RETRIES):
        try:
            df = yf.download(
                tickers, start=start, end=end_str,
                auto_adjust=False, progress=False,
                group_by="ticker", threads=True,
            )
            if df is None or df.empty:
                time.sleep(2 ** attempt + random.uniform(0.5, 1.5))
                continue
            for sym in tickers:
                parsed = _parse_symbol_from_df(df, sym)
                if parsed is None:
                    continue
                close, low, vol = parsed
                if len(close) >= 40:
                    result[sym] = (
                        close.tail(lookback),
                        low.tail(lookback),
                        vol.tail(lookback),
                    )
            return result
        except Exception:
            time.sleep(2 ** attempt + random.uniform(0.5, 1.5))
    return result


def batch_download(
    tickers: list[str], lookback: int,
) -> dict[str, tuple[pd.Series, pd.Series, pd.Series]]:
    """
    Download OHLCV for tickers. Retries missing symbols in smaller batches / singles.
    Close/Low/Volume share the same index before tail (matches nya_cycle_combined).
    """
    if not tickers:
        return {}

    result = _download_batch_once(tickers, lookback)
    missing = [t for t in tickers if t not in result]
    if not missing:
        return result

    # Retry missing: pairs then singles
    for i in range(0, len(missing), 2):
        chunk = missing[i : i + 2]
        extra = _download_batch_once(chunk, lookback)
        result.update(extra)

    still_missing = [t for t in tickers if t not in result]
    for sym in still_missing:
        extra = _download_batch_once([sym], lookback)
        result.update(extra)

    return result


def find_td_buy_signals(close: pd.Series, low: pd.Series, signal_days: int) -> list[dict]:
    """
    Global TD buy state machine (same as nya_cycle_combined.td_sequential buy side).
    Date window: last signal_days bars on the actual index (inclusive).
    """
    c = close.values.astype(float)
    lo = low.reindex(close.index).values.astype(float)
    dates = close.index
    n = len(c)
    if n < 20:
        return []

    cutoff_idx = max(0, n - signal_days)
    cutoff = dates[cutoff_idx]
    signals: list[dict] = []

    buy_cnt = sell_cnt = 0
    cd_dir: str | None = None
    cd_count = 0

    for i in range(n):
        if i >= 4:
            if c[i] < c[i - 4]:
                buy_cnt += 1
                sell_cnt = 0
            elif c[i] > c[i - 4]:
                sell_cnt += 1
                buy_cnt = 0
            else:
                buy_cnt = 0
                sell_cnt = 0

            if buy_cnt == 9:
                if cd_dir == "sell":
                    cd_count = 0
                cd_dir, cd_count = "buy", 0
                if dates[i] >= cutoff:
                    signals.append({"type": "TD9", "date": dates[i].date(), "idx": i})

            elif sell_cnt == 9:
                if cd_dir == "buy":
                    cd_count = 0
                cd_dir, cd_count = "sell", 0

        if cd_dir is None or i < 2:
            continue

        if cd_dir == "buy":
            if c[i] <= lo[i - 2]:
                cd_count += 1
                if cd_count == 13 and dates[i] >= cutoff:
                    signals.append({"type": "TD13", "date": dates[i].date(), "idx": i})
                if cd_count == 15:
                    if dates[i] >= cutoff:
                        signals.append({"type": "TD15", "date": dates[i].date(), "idx": i})
                    cd_dir = None

    return signals


def _avg_vol_at(vol: pd.Series, end_idx: int) -> float:
    start = max(0, end_idx - 19)
    window = vol.iloc[start : end_idx + 1]
    if window.empty:
        return 0.0
    return float(window.mean())


def analyse(
    ticker: str,
    close: pd.Series,
    low: pd.Series,
    vol: pd.Series,
    signal_days: int,
    min_price: float,
    min_vol: float,
    apply_liquidity_filter: bool = True,
) -> tuple[list[dict], int]:
    """
    Return (result rows, count of signals dropped by liquidity filter).
    Liquidity is evaluated at each signal's bar, not at today's close.
    """
    try:
        sigs = find_td_buy_signals(close, low, signal_days)
        today = date.today()
        rows: list[dict] = []
        liquidity_skipped = 0

        for s in sigs:
            idx = s["idx"]
            sig_price = float(close.iloc[idx])
            sig_vol = _avg_vol_at(vol, idx)

            if apply_liquidity_filter and (sig_price < min_price or sig_vol < min_vol):
                liquidity_skipped += 1
                continue

            rows.append({
                "ticker": ticker,
                "signal": s["type"],
                "sig_date": str(s["date"]),
                "days_ago": (today - s["date"]).days,
                "price": f"{sig_price:.2f}",
            })
        return rows, liquidity_skipped
    except Exception as exc:
        raise RuntimeError(f"{ticker}: {exc}") from exc
