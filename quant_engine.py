"""Quant engine extracted from nya_V6.py (no GUI dependencies)."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import periodogram, find_peaks

_BASE = Path(__file__).resolve().parent
DEFAULT_SYMBOL      = "HON"
MIN_LOOKBACK_BARS   = 678
MAX_LOOKBACK_BARS   = 900
LOOKBACK_STEP       = 10
DEFAULT_FUTURE_DAYS = 65
DEFAULT_PERIODS     = [127.0, 227.0, 78.0]#[27.0, 54.0, 81.0]
PHASING_SCALE       = 209
DEFAULT_HTML        = str(_BASE / "tmp" / "cycle_analysis.html")
YF_CACHE_DIR        = _BASE / "tmp" / "yf_cache"
PASS_COLOR          = "#59c36a"
FAIL_COLOR          = "#e36a6a"
ROBUST_SIGMA_FACTOR = 1.4826

DEFAULT_SLIDING_WINDOW = 252
DEFAULT_TREND_DEGREE   = 2
WINDOW_MIN_WEIGHT      = 0.35
WINDOW_WEIGHT_POWER    = 1.35

YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(YF_CACHE_DIR.resolve()))

# ══════════════════════════════════════════════════════════════════════════════
#  QUANT ENGINE: Indicators
# ══════════════════════════════════════════════════════════════════════════════

def chaikin_money_flow(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series, n: int = 20):
    mf_mult = ((close - low) - (high - close)) / (high - low + 1e-9)
    mf_vol = mf_mult * volume
    cmf = mf_vol.rolling(n).sum() / (volume.rolling(n).sum() + 1e-9)
    return cmf

def calculate_supertrend(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, multiplier: float = 3.0):
    df = pd.DataFrame({'h': high, 'l': low, 'c': close}).dropna()
    h, l, c = df['h'].values, df['l'].values, df['c'].values
    n = len(c)

    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(window=period).mean().values

    hl2 = (h + l) / 2
    basic_ub = hl2 + (multiplier * atr)
    basic_lb = hl2 - (multiplier * atr)

    final_ub = np.full(n, np.nan)
    final_lb = np.full(n, np.nan)
    trend = np.ones(n)

    if n > period:
        final_ub[period] = basic_ub[period]
        final_lb[period] = basic_lb[period]

    for i in range(period + 1, n):
        if basic_ub[i] < final_ub[i-1] or c[i-1] > final_ub[i-1]: final_ub[i] = basic_ub[i]
        else: final_ub[i] = final_ub[i-1]

        if basic_lb[i] > final_lb[i-1] or c[i-1] < final_lb[i-1]: final_lb[i] = basic_lb[i]
        else: final_lb[i] = final_lb[i-1]

        if trend[i-1] == 1 and c[i] < final_lb[i]: trend[i] = -1
        elif trend[i-1] == -1 and c[i] > final_ub[i]: trend[i] = 1
        else: trend[i] = trend[i-1]

    st_line = np.where(trend == 1, final_lb, final_ub)
    res_st = np.full(len(close), np.nan); res_tr = np.full(len(close), 1)
    res_st[-n:] = st_line; res_tr[-n:] = trend

    return res_tr, res_st

@dataclass
class TDSignal:
    idx: int; date: pd.Timestamp; kind: str; count: int
    perfect: bool = False; price: float = 0.0

def td_sequential(close: pd.Series, high: pd.Series, low: pd.Series, cd_target: int = 13, cd_extended: int = 15):
    n = len(close)
    c, h, lo = close.values.astype(float), high.values.astype(float), low.values.astype(float)
    setup_arr = np.zeros(n, dtype=int); countdown_arr = np.zeros(n, dtype=int)
    signals: list[TDSignal] = []
    buy_cnt = sell_cnt = 0; cd_dir = None; cd_count = 0

    for i in range(n):
        if i >= 4:
            if c[i] < c[i-4]: buy_cnt += 1; sell_cnt = 0
            elif c[i] > c[i-4]: sell_cnt += 1; buy_cnt = 0
            else: buy_cnt = 0; sell_cnt = 0
            setup_arr[i] = -buy_cnt if buy_cnt > 0 else sell_cnt

            if buy_cnt == 9:
                perfect = i >= 3 and min(lo[i], lo[i-1]) <= min(lo[i-2], lo[i-3])
                signals.append(TDSignal(i, close.index[i], 'buy9', 9, perfect, c[i]))
                if cd_dir == 'sell': cd_count = 0
                cd_dir, cd_count = 'buy', 0
            elif sell_cnt == 9:
                perfect = i >= 3 and max(h[i], h[i-1]) >= max(h[i-2], h[i-3])
                signals.append(TDSignal(i, close.index[i], 'sell9', 9, perfect, c[i]))
                if cd_dir == 'buy': cd_count = 0
                cd_dir, cd_count = 'sell', 0

        if cd_dir is None or i < 2: continue
        if cd_dir == 'buy':
            if c[i] <= lo[i-2]:
                cd_count += 1; countdown_arr[i] = -cd_count
                if cd_count == cd_target: signals.append(TDSignal(i, close.index[i], 'buy13', cd_count, False, c[i]))
                if cd_count == cd_extended: signals.append(TDSignal(i, close.index[i], 'buy15', cd_count, False, c[i])); cd_dir = None
            else: countdown_arr[i] = -cd_count
        elif cd_dir == 'sell':
            if c[i] >= h[i-2]:
                cd_count += 1; countdown_arr[i] = cd_count
                if cd_count == cd_target: signals.append(TDSignal(i, close.index[i], 'sell13', cd_count, False, c[i]))
                if cd_count == cd_extended: signals.append(TDSignal(i, close.index[i], 'sell15', cd_count, False, c[i])); cd_dir = None
            else: countdown_arr[i] = cd_count

    return setup_arr, countdown_arr, signals

def squeeze_momentum(close: pd.Series, high: pd.Series, low: pd.Series, bb_length=20, bb_mult=2.0, kc_length=20, kc_mult=1.5):
    c, h, lo = close.values.astype(float), high.values.astype(float), low.values.astype(float)
    n = len(c)
    s = pd.Series(c)
    basis = s.rolling(bb_length).mean().values
    dev = s.rolling(bb_length).std(ddof=0).values
    upperBB, lowerBB = basis + bb_mult * dev, basis - bb_mult * dev

    ma = s.rolling(kc_length).mean().values
    prev_c = np.empty_like(c); prev_c[0] = c[0]; prev_c[1:] = c[:-1]
    tr = np.maximum(h - lo, np.maximum(np.abs(h - prev_c), np.abs(lo - prev_c)))
    rangema = pd.Series(tr).rolling(kc_length).mean().values
    upperKC, lowerKC = ma + kc_mult * rangema, ma - kc_mult * rangema

    sqzOn = (lowerBB > lowerKC) & (upperBB < upperKC)
    sqzOff = (lowerBB < lowerKC) & (upperBB > upperKC)
    noSqz = ~sqzOn & ~sqzOff

    highest_h = pd.Series(h).rolling(kc_length).max().values
    lowest_l = pd.Series(lo).rolling(kc_length).min().values
    delta = c - (((highest_h + lowest_l) / 2.0) + ma) / 2.0

    val = np.full(n, np.nan)
    x = np.arange(kc_length, dtype=float)
    for i in range(kc_length - 1, n):
        y = delta[i - kc_length + 1: i + 1]
        if np.any(np.isnan(y)): continue
        coef = np.polyfit(x, y, 1)
        val[i] = np.polyval(coef, kc_length - 1)

    bar_colors, dot_colors = [], []
    for i in range(n):
        v = val[i]
        if np.isnan(v): bar_colors.append('rgba(100,100,100,0.3)'); continue
        prev = val[i - 1] if i > 0 and not np.isnan(val[i - 1]) else v
        if v > 0: bar_colors.append('#00e676' if v > prev else '#26a69a')
        else: bar_colors.append('#ef5350' if v < prev else '#b71c1c')

    for i in range(n):
        if noSqz[i]: dot_colors.append('#29b6f6')
        elif sqzOn[i]: dot_colors.append('#212121')
        else: dot_colors.append('#9e9e9e')

    return val, bar_colors, dot_colors, sqzOn, sqzOff, noSqz

# ══════════════════════════════════════════════════════════════════════════════
#  CYCLE ANALYSIS CORE
# ══════════════════════════════════════════════════════════════════════════════
class DataFetchError(ValueError):
    """Raised when Yahoo Finance returns no usable OHLCV bars."""


def _normalize_yf_frame(data: pd.DataFrame | None) -> pd.DataFrame:
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        return pd.DataFrame()
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _series_from_frame(data: pd.DataFrame, col: str) -> pd.Series:
    lookup = {c.lower(): c for c in data.columns}
    key = col.lower()
    if key not in lookup:
        raise DataFetchError(f"Missing '{col}' column. Got: {list(data.columns)}")
    s = data[lookup[key]]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.dropna()


def download_ohlcv(symbol, start, end_inclusive):
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise DataFetchError("Ticker symbol is required.")

    fetch_end = (pd.Timestamp(end_inclusive) + pd.Timedelta(days=1)).date().isoformat()
    data = _normalize_yf_frame(
        yf.download(symbol, start=start, end=fetch_end, auto_adjust=False, progress=False, threads=False)
    )
    if data.empty:
        data = _normalize_yf_frame(
            yf.Ticker(symbol).history(start=start, end=fetch_end, auto_adjust=False)
        )

    if data.empty:
        raise DataFetchError(
            f"No OHLCV data for '{symbol}' ({start} → {end_inclusive}). "
            "Check the ticker symbol or retry later — Yahoo Finance may be rate-limiting."
        )

    close = _series_from_frame(data, "Close")
    if len(close) < 3:
        raise DataFetchError(
            f"Only {len(close)} bar(s) returned for '{symbol}' ({start} → {end_inclusive}). "
            "Need at least 3 trading days of data."
        )

    idx = close.index
    return (
        close,
        _series_from_frame(data, "High").reindex(idx).dropna(),
        _series_from_frame(data, "Low").reindex(idx).dropna(),
        _series_from_frame(data, "Volume").reindex(idx).dropna(),
    )

def scale_composite_to_price(close, composite):
    ins = composite[:len(close)]; lp,hp = np.quantile(close.values,[.05,.95]); lc,hc = np.quantile(ins,[.05,.95])
    if np.isclose(hc,lc):
        lo,hi = float(close.min())*.98, float(close.max())*1.02
        return (composite-composite.min())/(composite.max()-composite.min())*(hi-lo)+lo
    slope = float((hp-lp)/(hc-lc)); return composite*slope+float(lp-slope*lc)

def compose_full_series(coefficients, full_length, periods):
    t = np.arange(full_length, dtype=float)
    return np.sum([np.column_stack([np.sin(2*np.pi*t/p), np.cos(2*np.pi*t/p)]) @ coefficients[p] for p in periods], axis=0)

def detrend_series(values: np.ndarray, degree: int = DEFAULT_TREND_DEGREE):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise DataFetchError("Cannot detrend an empty price series — no market data was loaded.")
    t = np.arange(len(values), dtype=float)
    if len(values) < 3: return values - np.polyval(np.polyfit(t, values, 1), t), np.polyval(np.polyfit(t, values, 1), t)
    degree = max(1, min(int(degree), 3, len(values) - 1))
    trend = np.polyval(np.polyfit(t, values, degree), t)
    return values - trend, trend

def _window_slice(n: int, window_bars: int | None):
    if window_bars is None or window_bars <= 0 or window_bars >= n: return slice(0, n)
    return slice(n - int(window_bars), n)

def _recent_weights(n: int):
    if n <= 1: return np.ones(max(n, 1), dtype=float)
    return np.power(np.linspace(WINDOW_MIN_WEIGHT, 1.0, n, dtype=float), WINDOW_WEIGHT_POWER)

def build_overlay(close, periods, future_days, sliding_window=DEFAULT_SLIDING_WINDOW, trend_degree=DEFAULT_TREND_DEGREE):
    t = np.arange(len(close), dtype=float)
    residual, trend = detrend_series(close.values.astype(float), degree=trend_degree)
    future_index = pd.bdate_range(close.index[-1] + pd.offsets.BDay(1), periods=future_days)
    full_index = close.index.append(future_index)
    fit_slice = _window_slice(len(close), sliding_window)
    fit_t = t[fit_slice]; fit_residual = residual[fit_slice]; fit_weights = _recent_weights(len(fit_t))

    coefficients, cycle_stats, components_fit = {}, {}, []
    rs = float(np.std(fit_residual))
    mad = float(np.median(np.abs(fit_residual - np.median(fit_residual))))
    robust_sigma = mad * ROBUST_SIGMA_FACTOR if mad else rs
    total_ss = float(np.sum((fit_residual - fit_residual.mean()) ** 2))

    for p in periods:
        angle_fit = 2 * np.pi * fit_t / p
        D_fit = np.column_stack([np.sin(angle_fit), np.cos(angle_fit)])
        w = np.sqrt(np.asarray(fit_weights, dtype=float))
        coef, *_ = np.linalg.lstsq(D_fit * w[:, None], fit_residual * w, rcond=None)
        coefficients[p] = coef
        components_fit.append(D_fit @ coef)
        amp = float(np.sqrt(coef[0] ** 2 + coef[1] ** 2))
        cur = float(np.sin(2*np.pi*t[-1]/p)*coef[0] + np.cos(2*np.pi*t[-1]/p)*coef[1])
        cycle_stats[p] = {
            "strength": amp / rs if rs else 0.0,
            "robust_strength": amp / robust_sigma if robust_sigma else 0.0,
            "phase_position": cur / amp if amp else 0.0,
        }
        cycle_stats[p]["effective_score"] = cycle_stats[p]["strength"] * (1.0 - abs(cycle_stats[p]["phase_position"]))

    combined_fit = np.sum(components_fit, axis=0) if components_fit else np.zeros_like(fit_residual)
    combined_r2 = max(0.0, 1.0 - float(np.sum((fit_residual - combined_fit)**2)) / total_ss) if total_ss else 0.0
    strongest = sorted(periods, key=lambda p: cycle_stats[p]["strength"], reverse=True)[:2]
    p_score = int(round(PHASING_SCALE * np.average([cycle_stats[p]["phase_position"] for p in strongest], weights=[cycle_stats[p]["strength"] for p in strongest])))
    overlay = pd.Series(scale_composite_to_price(close, compose_full_series(coefficients, len(full_index), periods)), index=full_index)
    return overlay, cycle_stats, p_score, combined_r2, coefficients

def days_to_next_turn(period: float, coef: np.ndarray, t_now: float, kind: str = "trough") -> float:
    """Trading-day index offset to the next harmonic peak or trough for one fitted component."""
    a, b = float(coef[0]), float(coef[1])
    phi = np.arctan2(b, a)
    w = 2 * np.pi / period
    target = 1.5 * np.pi if kind == "trough" else 0.5 * np.pi
    k = int(np.floor((t_now * w + phi - target) / (2 * np.pi))) + 1
    for _ in range(64):
        t_next = (target - phi + 2 * np.pi * k) / w
        if t_next > t_now:
            return float(t_next - t_now)
        k += 1
    return float("nan")

def projection_extrema(overlay: pd.Series, close: pd.Series, min_distance: int = 5) -> list[dict]:
    """Local peaks/troughs on the dashed projection segment of the scaled cycle line."""
    proj = overlay.iloc[len(close) :]
    if len(proj) < 3:
        return []
    y = proj.values.astype(float)
    events: list[dict] = []
    for kind, idxs in (("trough", find_peaks(-y, distance=min_distance)[0]), ("peak", find_peaks(y, distance=min_distance)[0])):
        for i in idxs:
            events.append({"kind": kind, "date": proj.index[i], "value": float(y[i])})
    events.sort(key=lambda e: e["date"])
    return events

def build_cycle_timing_report(
    close: pd.Series,
    periods: list[float],
    cycle_stats: dict,
    coefficients: dict,
    overlay: pd.Series,
    ranked_periods: list[float],
    combined_r2: float,
) -> str:
    t_now = float(len(close) - 1)
    last_dt = close.index[-1]
    lines = ["", "── Cycle timing (harmonic projection) ──"]

    for label, p in [("Dominant", ranked_periods[0]), ("2nd", ranked_periods[1] if len(ranked_periods) > 1 else None)]:
        if p is None:
            continue
        coef = coefficients[p]
        strength = cycle_stats[p]["robust_strength"]
        phase = cycle_stats[p]["phase_position"]
        for kind in ("trough", "peak"):
            bd = days_to_next_turn(p, coef, t_now, kind)
            if not np.isfinite(bd) or bd > 400:
                continue
            dt = last_dt + pd.offsets.BDay(int(round(bd)))
            lines.append(
                f"  {label} {int(round(p))}d → next {kind}: {dt.date()} "
                f"(~{bd:.0f} sessions, strength {strength:.2f}, phase {phase:+.2f})"
            )

    extrema = projection_extrema(overlay, close)
    if extrema:
        troughs = [e for e in extrema if e["kind"] == "trough"]
        peaks = [e for e in extrema if e["kind"] == "peak"]
        if troughs:
            main = min(troughs, key=lambda e: e["value"])
            lines.append(
                f"  Composite line (scaled) main dip: {main['date'].date()} "
                f"(lowest point on dashed projection)"
            )
            jun_aug = [e for e in troughs if e["date"].month in (6, 7, 8)]
            if jun_aug:
                window = ", ".join(str(e["date"].date()) for e in jun_aug)
                lines.append(f"  Jun–Aug projection trough(s) on cycle line: {window}")
        if peaks:
            top = max(peaks, key=lambda e: e["value"])
            lines.append(f"  Composite line next crest: {top['date'].date()}")

    if combined_r2 < 0.35:
        lines.append("  Note: R² < 0.35 — treat dates as rough; confirm on chart.")
    lines.append("  (Dates are model turns on detrended cycles, not guaranteed price lows.)")
    return "\n".join(lines)

def metrics_for_window(
    close, periods, future_days,
    sliding_window: int = DEFAULT_SLIDING_WINDOW,
    trend_degree: int = DEFAULT_TREND_DEGREE,
):
    overlay, cs, p_score, r2, coefficients = build_overlay(
        close, periods, future_days, sliding_window=sliding_window, trend_degree=trend_degree
    )
    ranked = sorted(periods, key=lambda p: cs[p]["effective_score"], reverse=True)
    return {
        "overlay": overlay, "cycle_stats": cs, "p_score": p_score, "combined_r2": r2,
        "ranked_periods": ranked,
        "top_strength": float(max(s["robust_strength"] for s in cs.values())),
        "coefficients": coefficients,
        "timing_report": build_cycle_timing_report(close, periods, cs, coefficients, overlay, ranked, r2),
        "projection_extrema": projection_extrema(overlay, close),
    }

# ─── Dynamic Frequency Extraction (DFE) ───────────────────────────────────────
DFE_MIN_PERIOD = 15.0
DFE_MIN_SEP_DAYS = 10.0
DFE_PROMINENCE_FRAC = 0.05
DFE_HARMONIC_TOL = 0.08

def _is_harmonic(candidate: float, existing: list[float], tol: float = DFE_HARMONIC_TOL) -> bool:
    for e in existing:
        for ratio in (2.0, 0.5, 3.0, 1.0 / 3.0):
            if abs(candidate - e * ratio) / max(e, 1.0) < tol:
                return True
    return False

def extract_dominant_periods(
    residual: np.ndarray,
    n_periods: int = 3,
    fs: float = 1.0,
    min_period: float = DFE_MIN_PERIOD,
    max_period: float | None = None,
    min_sep_days: float = DFE_MIN_SEP_DAYS,
    prominence_frac: float = DFE_PROMINENCE_FRAC,
) -> tuple[list[float], bool]:
    """PSD peak pick → top cycle lengths in trading days. Returns (periods, used_fallback)."""
    x = np.asarray(residual, dtype=float)
    n = len(x)
    if n < int(2 * min_period):
        return list(DEFAULT_PERIODS[:n_periods]), True

    max_period = max_period or n / 2.5
    x = x - np.mean(x)
    freqs, psd = periodogram(x, fs=fs, detrend=False)

    valid = freqs > 0
    freqs, psd = freqs[valid], psd[valid]
    periods = 1.0 / freqs

    mask = (periods >= min_period) & (periods <= max_period)
    periods, psd = periods[mask], psd[mask]
    if len(periods) == 0:
        return list(DEFAULT_PERIODS[:n_periods]), True

    prom = max(float(psd.max()) * prominence_frac, 1e-12)
    peak_idx, _ = find_peaks(psd, prominence=prom)
    if len(peak_idx) == 0:
        return list(DEFAULT_PERIODS[:n_periods]), True

    order = peak_idx[np.argsort(psd[peak_idx])[::-1]]
    chosen: list[float] = []
    for idx in order:
        p = float(periods[idx])
        if any(abs(p - c) < min_sep_days for c in chosen):
            continue
        if _is_harmonic(p, chosen):
            continue
        chosen.append(p)
        if len(chosen) >= n_periods:
            break

    if len(chosen) < n_periods:
        for fallback_p in DEFAULT_PERIODS:
            if len(chosen) >= n_periods:
                break
            if any(abs(fallback_p - c) < min_sep_days for c in chosen):
                continue
            if _is_harmonic(fallback_p, chosen):
                continue
            chosen.append(float(fallback_p))
        if len(chosen) < n_periods:
            return list(DEFAULT_PERIODS[:n_periods]), True

    chosen.sort(reverse=True)
    return chosen[:n_periods], False

def resolve_periods(
    close: pd.Series,
    manual: list[float] | None = None,
    auto: bool = True,
    sliding_window: int = DEFAULT_SLIDING_WINDOW,
    trend_degree: int = DEFAULT_TREND_DEGREE,
) -> tuple[list[float], str]:
    if not auto and manual:
        return list(manual), "manual"
    residual, _ = detrend_series(close.values.astype(float), degree=trend_degree)
    fit_residual = residual[_window_slice(len(close), sliding_window)]
    periods, used_fallback = extract_dominant_periods(fit_residual)
    return periods, "fallback" if used_fallback else "dfe"

def compute_min_lookback(periods: list[float]) -> int:
    return max(MIN_LOOKBACK_BARS, int(np.ceil(3.0 * max(periods))))

def _slice_ohlcv(close, high, low, volume, n_bars: int):
    n_bars = min(int(n_bars), len(close))
    if n_bars >= len(close):
        return close, high, low, volume
    sl = slice(-n_bars, None)
    return close.iloc[sl], high.iloc[sl], low.iloc[sl], volume.iloc[sl]

def choose_lookback_bars(
    close, high, low, volume,
    periods, future_days,
    log_cb=None,
    min_bars: int | None = None,
    max_bars: int = MAX_LOOKBACK_BARS,
    step: int = LOOKBACK_STEP,
    sliding_window: int = DEFAULT_SLIDING_WINDOW,
    trend_degree: int = DEFAULT_TREND_DEGREE,
) -> int:
    def _log(m):
        if log_cb:
            log_cb(m)

    min_bars = min_bars if min_bars is not None else compute_min_lookback(periods)
    min_bars = min(max(1, min_bars), len(close))
    max_bars = min(max_bars, len(close))
    if min_bars >= max_bars:
        return min_bars

    c, h, l, v = _slice_ohlcv(close, high, low, volume, min_bars)
    base_m = metrics_for_window(
        c, periods, future_days, sliding_window=sliding_window, trend_degree=trend_degree
    )
    base_rank = sorted(periods, key=lambda p: base_m["cycle_stats"][p]["strength"], reverse=True)
    base_p = int(base_m["p_score"])
    base_r2 = float(base_m["combined_r2"])
    base_top = float(base_m["top_strength"])
    chosen = min_bars

    for lb in range(min_bars + step, max_bars + 1, step):
        if lb > len(close):
            break
        _log(f"  Testing {lb} bars…")
        c, h, l, v = _slice_ohlcv(close, high, low, volume, lb)
        m = metrics_for_window(
            c, periods, future_days, sliding_window=sliding_window, trend_degree=trend_degree
        )
        rank = sorted(periods, key=lambda p: m["cycle_stats"][p]["strength"], reverse=True)
        if (
            rank == base_rank
            and abs(float(m["top_strength"]) - base_top) <= 0.02
            and abs(float(m["combined_r2"]) - base_r2) <= 0.01
            and abs(int(m["p_score"]) - base_p) <= 5
        ):
            chosen = lb
        else:
            break
    return chosen

def resolve_cycle_config(
    close, high, low, volume,
    *,
    periods_manual: list[float] | None = None,
    auto_dfe: bool = True,
    future_days: int = DEFAULT_FUTURE_DAYS,
    explicit_lookback: int | None = None,
    auto_lookback: bool = False,
    log_cb=None,
    sliding_window: int = DEFAULT_SLIDING_WINDOW,
    trend_degree: int = DEFAULT_TREND_DEGREE,
) -> dict:
    messages: list[str] = []

    if auto_dfe:
        periods, source = resolve_periods(
            close, manual=periods_manual, auto=True,
            sliding_window=sliding_window, trend_degree=trend_degree,
        )
    elif periods_manual:
        periods = list(periods_manual)
        source = "manual"
    else:
        periods = list(DEFAULT_PERIODS)
        source = "manual"

    if auto_lookback:
        min_lb = compute_min_lookback(periods)
        if min_lb > len(close):
            messages.append(
                f"Warning: period-driven min lookback ({min_lb}) exceeds available bars ({len(close)}); clamping."
            )
        lookback = choose_lookback_bars(
            close, high, low, volume, periods, future_days,
            log_cb=log_cb,
            min_bars=min(min_lb, len(close)),
            max_bars=min(MAX_LOOKBACK_BARS, len(close)),
            sliding_window=sliding_window,
            trend_degree=trend_degree,
        )
    elif explicit_lookback is not None:
        lookback = min(int(explicit_lookback), len(close))
    else:
        lookback = len(close)

    close, high, low, volume = _slice_ohlcv(close, high, low, volume, lookback)

    if auto_dfe:
        periods, source = resolve_periods(
            close, manual=periods_manual, auto=True,
            sliding_window=sliding_window, trend_degree=trend_degree,
        )

    min_required = compute_min_lookback(periods)
    if lookback < min_required:
        messages.append(
            f"Warning: lookback {lookback} bars < 3× longest period ({min_required} bars recommended)."
        )

    return {
        "periods": periods,
        "periods_source": source,
        "lookback_bars": lookback,
        "close": close,
        "high": high,
        "low": low,
        "volume": volume,
        "messages": messages,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  TRADE STATE ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def describe_cycle_outlook(close: pd.Series, cycle_metrics: dict | None) -> dict:
    """
    Summarize the harmonic cycle projection direction for the trade report.

    Returns a dict with:
      direction : 'up' | 'down' | 'flat' | 'unknown'
      slope_pct : projection slope per session, in % of current price (relative move)
      horizon_pct : total move of the cycle line across the projection, in % of price
      r2, confidence, next_turn_kind, next_turn_days, lines (list[str])
    The projection is a detrended-cycle line scaled to the price range — it indicates
    cycle PHASE direction, NOT a price target or probability.
    """
    result = {
        "direction": "unknown", "slope_pct": 0.0, "horizon_pct": 0.0,
        "r2": 0.0, "confidence": "无", "next_turn_kind": None,
        "next_turn_days": None, "lines": [],
    }
    if not cycle_metrics:
        return result

    overlay = cycle_metrics.get("overlay")
    r2 = float(cycle_metrics.get("combined_r2", 0.0))
    result["r2"] = r2
    if overlay is None or len(overlay) <= len(close) + 1:
        result["lines"].append("周期模型: 无未来投影数据，方向未知。")
        return result

    hist_n = len(close)
    proj = overlay.iloc[hist_n:]
    y = np.asarray(proj.values, dtype=float)
    last_px = float(close.iloc[-1]) or 1.0

    slope = float(np.polyfit(np.arange(len(y)), y, 1)[0])
    slope_pct = slope / last_px * 100.0
    horizon_pct = (y[-1] - y[0]) / last_px * 100.0
    result["slope_pct"] = slope_pct
    result["horizon_pct"] = horizon_pct

    if slope_pct > 0.02:
        direction, arrow = "up", "向上 ↑"
    elif slope_pct < -0.02:
        direction, arrow = "down", "向下 ↓"
    else:
        direction, arrow = "flat", "走平 →"
    result["direction"] = direction

    if r2 >= 0.5:
        conf = "高"
    elif r2 >= 0.35:
        conf = "中"
    else:
        conf = "低(仅参考)"
    result["confidence"] = conf

    # Next projected turn (peak/trough) from the dashed projection line
    next_turn_txt = ""
    extrema = cycle_metrics.get("projection_extrema")
    if extrema is None:
        extrema = projection_extrema(overlay, close)
    if extrema:
        nxt = extrema[0]
        kind_cn = "见底(周期低点)" if nxt["kind"] == "trough" else "见顶(周期高点)"
        result["next_turn_kind"] = nxt["kind"]
        try:
            days = int(np.busday_count(close.index[-1].date(), nxt["date"].date()))
            result["next_turn_days"] = days
            next_turn_txt = f" 预计约 {days} 个交易日后{kind_cn} ({nxt['date'].date()})。"
        except Exception:
            next_turn_txt = f" 下一转折: {kind_cn}。"

    result["lines"].append("── 周期模型 (谐波投影) ──")
    result["lines"].append(
        f"周期方向: {arrow} (未来投影斜率 {slope_pct:+.3f}%/日, "
        f"投影区间累计 {horizon_pct:+.1f}%; 拟合 R²={r2:.2f}, 置信度: {conf})"
    )
    if next_turn_txt:
        result["lines"].append("周期节奏:" + next_turn_txt)
    result["lines"].append(
        "说明: 周期线为去趋势谐波按价格分位缩放的结果，只表示周期相位的上/下方向，"
        "不是价格目标或上涨概率。"
    )
    return result


def analyze_trade_state(close: pd.Series, low: pd.Series, high: pd.Series, trend: np.ndarray, st_line: np.ndarray, td_signals: list, sqz_val: np.ndarray, cmf_val: pd.Series, cycle_metrics: dict | None = None):
    current_price = float(close.iloc[-1])
    st_support_resist = st_line[-1]
    current_trend = 1 if current_price > st_support_resist else -1

    recent_td_buy = any(s.kind == 'buy9' and (len(close) - s.idx) <= 3 for s in td_signals)
    recent_td_sell = any(s.kind == 'sell9' and (len(close) - s.idx) <= 3 for s in td_signals)
    current_sqz, prev_sqz = sqz_val[-1], (sqz_val[-2] if len(sqz_val) > 1 else sqz_val[-1])
    cmf_current = cmf_val.iloc[-1]

    tr = np.maximum(high.values - low.values, np.maximum(np.abs(high.values - np.roll(close.values, 1)), np.abs(low.values - np.roll(close.values, 1))))
    current_atr = pd.Series(tr).rolling(14).mean().iloc[-1]

    cycle = describe_cycle_outlook(close, cycle_metrics)
    cycle_dir = cycle["direction"]

    report = [f"\n▼▼▼量化决策▼▼▼ (最新价: {current_price:.2f})", "═" * 50]
    report.append(f"资金流向 (CMF): {'流入(机构吸筹)' if cmf_current > 0 else '流出(机构派发)'} (CMF: {cmf_current:.3f})")

    if current_trend == 1:
        report.append("当前状态: [右侧交易] - 趋势向上，大势安全。")
        if recent_td_sell:
            report.append("警报: 出现 TD9 卖出极值！且面临可能的技术阻力。")
            report.append("动作: 禁止加仓！持仓者应在此处减仓锁定利润。")
        elif current_sqz > prev_sqz and current_sqz > 0:
            if cmf_current > 0:
                report.append("动能: 量价齐升 (SQZ绿柱 + CMF流入)。")
                if cycle_dir == "up":
                    report.append("动作: 【正金字塔加仓】趋势+资金+周期三重共振，可果断突破加仓。")
                elif cycle_dir == "down":
                    report.append("动作: 趋势与资金向好，但周期已进入下行相位，加仓宜轻仓/分批，防止追高。")
                else:
                    report.append("动作: 【正金字塔加仓】趋势与资金共振，可果断突破加仓。")
            else:
                report.append("警报: [顶背离风险] 价格上涨但机构资金(CMF)显示流出！")
                report.append("动作: 停止买入，持股观望，随时准备跌破支撑时离场。")
        else:
            report.append("动能: 动量减弱或震荡阶段。")
            if cycle_dir == "up":
                report.append("动作: 缩量回踩期，但周期仍向上，回踩支撑不破可低吸。")
            elif cycle_dir == "down":
                report.append("动作: 缩量回踩且周期下行，观望为主，等待周期见底再介入。")
            else:
                report.append("动作: 缩量回踩期。重点关注下方支撑。")
        report.append(f"止损/防守线: {st_support_resist:.2f} (SuperTrend 支撑)")

    elif current_trend == -1:
        if recent_td_buy:
            td_low = min([low.iloc[s.idx] for s in td_signals if s.kind == 'buy9' and (len(close) - s.idx) <= 3])
            left_stop_loss = td_low - current_atr
            report.append("当前状态: [左侧极限] - 趋势向下，但出现反转信号。")
            if cmf_current > 0: report.append("异动: 主力资金(CMF)正在逆势吸筹！")
            if cycle_dir == "up":
                report.append("印证: 周期已转向上行相位，与 TD 反转信号形成共振，反弹概率增加。")
            elif cycle_dir == "down":
                report.append("注意: 周期仍处下行相位，反弹或为逃命波，严格控制仓位。")
            report.append("动作: 启动【漏斗建仓法】抢反弹，先打 30% 试探底仓。")
            report.append(f"极限止损线: {left_stop_loss:.2f} (破位无条件平仓)")
        else:
            report.append("当前状态: [绝对空头] - 趋势向下，接飞刀的高风险区。")
            if cycle_dir == "up":
                report.append("留意: 趋势仍空但周期相位已开始向上，可加入自选观察，等趋势转右侧再动手。")
            else:
                report.append("动作: 【空仓观望】。绝不盲目抄底。")

    for line in cycle["lines"]:
        report.append(line)

    report.append("═" * 50 + "\n")
    return "\n".join(report)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY CHART BUILDER
# ══════════════════════════════════════════════════════════════════════════════
_TD_COLORS = {'buy9':'#00e676', 'sell9':'#ff1744', 'buy13':'#00b0ff', 'sell13':'#ff9100', 'buy15':'#76ff03', 'sell15':'#f50057'}
_TD_MARKER = {'buy9':'triangle-up', 'sell9':'triangle-down', 'buy13':'triangle-up', 'sell13':'triangle-down', 'buy15':'triangle-up', 'sell15':'triangle-down'}
_TD_SIZE   = {'buy9':12, 'sell9':12, 'buy13':14, 'sell13':14, 'buy15':16, 'sell15':16}

def build_plotly_chart(
    close, overlay, cycle_stats, symbol, periods, high, low,
    show_td, show_td_setup, show_td_cd, show_td_nums, cd_tgt, cd_ext,
    show_sqz, sqz_bb_len, sqz_bb_mult, sqz_kc_len, sqz_kc_mult,
    show_st, st_period, st_mult, cmf_val,
    projection_extrema: list[dict] | None = None,
) -> Any:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    top_s = cycle_stats[max(periods, key=lambda p: cycle_stats[p]["robust_strength"])]["robust_strength"]
    title_html = f'<b>{symbol}</b> <span style="color:#aaa">| Top Strength: {top_s:.2f}</span>'

    # 根据开关动态计算行数，保持 UI 整洁
    rows = 1
    row_heights = [0.6]
    if show_sqz: rows += 1; row_heights.append(0.2)
    # CMF always on in v7
    rows += 1; row_heights.append(0.2)

    # 动态分配高度
    if rows == 3: row_heights = [0.6, 0.2, 0.2]
    elif rows == 2: row_heights = [0.75, 0.25]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=row_heights, vertical_spacing=0.03)
    curr_row = 1

    # --- ROW 1: 主图 ---
    ov_hist, ov_proj = overlay.iloc[:len(close)+1], overlay.iloc[len(close):]
    fig.add_trace(go.Scatter(x=ov_hist.index, y=ov_hist.values, name="Cycle", line=dict(color="#e040fb", width=2.0)), row=curr_row, col=1)
    fig.add_trace(go.Scatter(x=ov_proj.index, y=ov_proj.values, name="Projection", line=dict(color="#e040fb", width=2.0, dash="dash")), row=curr_row, col=1)

    if projection_extrema:
        troughs = [e for e in projection_extrema if e["kind"] == "trough"]
        peaks = [e for e in projection_extrema if e["kind"] == "peak"]
        if troughs:
            fig.add_trace(go.Scatter(
                x=[e["date"] for e in troughs], y=[e["value"] for e in troughs],
                mode="markers+text", name="Proj. trough",
                marker=dict(symbol="triangle-down", size=11, color="#ff5252"),
                text=[e["date"].strftime("%b %d") for e in troughs],
                textposition="bottom center",
                textfont=dict(color="#ff8a80", size=9),
            ), row=curr_row, col=1)
        if peaks:
            fig.add_trace(go.Scatter(
                x=[e["date"] for e in peaks], y=[e["value"] for e in peaks],
                mode="markers", name="Proj. peak",
                marker=dict(symbol="triangle-up", size=9, color="#69f0ae"),
            ), row=curr_row, col=1)

    if show_st:
        trend, st_line = calculate_supertrend(high, low, close, st_period, st_mult)
        st_up = np.where(trend == 1, st_line, np.nan)
        st_dn = np.where(trend == -1, st_line, np.nan)
        fig.add_trace(go.Scatter(x=close.index, y=st_up, name="SuperTrend (Up)", line=dict(color="rgba(0, 230, 118, 0.8)", width=2.5)), row=curr_row, col=1)
        fig.add_trace(go.Scatter(x=close.index, y=st_dn, name="SuperTrend (Dn)", line=dict(color="rgba(255, 82, 82, 0.8)", width=2.5)), row=curr_row, col=1)

    fig.add_trace(go.Scatter(x=close.index, y=close.values, name=symbol, line=dict(color="#4fc3f7", width=1.8)), row=curr_row, col=1)

    if show_td:
        setup_arr, countdown_arr, signals = td_sequential(close, high, low, cd_tgt, cd_ext)

        if show_td_nums:
            bx,by,bt, sx,sy,st, bcx,bcy,bct, scx,scy,sct = [],[],[],[],[],[],[],[],[],[],[],[]
            off = (float(high.max()-low.min()) or 1.0) * 0.012
            for i in range(len(close)):
                d,pv,s,cd = close.index[i],float(close.iloc[i]),setup_arr[i],countdown_arr[i]
                if s < 0:    bx.append(d);  by.append(pv-off);      bt.append(str(-s))
                elif s > 0:  sx.append(d);  sy.append(pv+off);      st.append(str(s))
                if cd < 0:   bcx.append(d); bcy.append(pv-off*2.5); bct.append(str(-cd))
                elif cd > 0: scx.append(d); scy.append(pv+off*2.5); sct.append(str(cd))
            for xs,ys,ts,col,nm in [(bx,by,bt,"#00e676","Setup Buy#"), (sx,sy,st,"#ff5252","Setup Sell#"), (bcx,bcy,bct,"#00b0ff","CD Buy#"), (scx,scy,sct,"#ff9100","CD Sell#")]:
                if xs: fig.add_trace(go.Scatter(x=xs, y=ys, mode="text", text=ts, name=nm, textfont=dict(color=col, size=9), hoverinfo="skip"), row=curr_row, col=1)

        for kind in ['buy9','sell9','buy13','sell13','buy15','sell15']:
            if kind.endswith('9') and not show_td_setup: continue
            if not kind.endswith('9') and not show_td_cd: continue
            sigs = [s for s in signals if s.kind == kind]
            if not sigs: continue
            is_buy = kind.startswith('buy')
            color, mk, mksz = _TD_COLORS[kind], _TD_MARKER[kind], _TD_SIZE[kind]
            off = (float(high.max()-low.min()) or 1.0) * 0.04
            xs = [s.date for s in sigs]; ys = [s.price + (-off if is_buy else off) for s in sigs]
            txts = [kind.upper() + ('✓' if s.perfect and kind.endswith('9') else '') for s in sigs]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", marker=dict(symbol=mk, size=mksz, color=color, line=dict(color="white", width=0.5)), text=txts, textposition="bottom center" if is_buy else "top center", textfont=dict(color=color, size=9, family="Arial Black"), name=kind.upper()), row=curr_row, col=1)

    # --- ROW 2: SQZ 动量 (If Enabled) ---
    if show_sqz:
        curr_row += 1
        val, bar_colors, dot_colors, sqzOn, sqzOff, _ = squeeze_momentum(close, high, low, sqz_bb_len, sqz_bb_mult, sqz_kc_len, sqz_kc_mult)
        fig.add_trace(go.Bar(x=close.index, y=val, marker_color=bar_colors, name="SQZ Momentum"), row=curr_row, col=1)
        fig.add_trace(go.Scatter(x=close.index, y=np.zeros(len(close)), mode="markers", marker=dict(symbol="circle", size=4, color=dot_colors), name="SQZ State"), row=curr_row, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.15)", line_width=1, row=curr_row, col=1)

    # --- ROW 3: CMF 资金流 ---
    curr_row += 1
    fig.add_trace(go.Scatter(x=cmf_val.index, y=cmf_val.values, name="CMF (Money Flow)", line=dict(color="#f3a53a", width=1.5), fill='tozeroy', fillcolor='rgba(243, 165, 58, 0.1)'), row=curr_row, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.4)", line_width=1, row=curr_row, col=1)

    # --- Layout ---
    gs = dict(gridcolor="#2a2e3e", gridwidth=1, griddash="dot"); ts = dict(tickfont=dict(color="#787b86", size=10)); ls = dict(linecolor="#363c4e", showline=True)
    rng_sel = dict(buttons=[dict(count=1, label="1M", step="month", stepmode="backward"), dict(count=3, label="3M", step="month", stepmode="backward"), dict(count=6, label="6M", step="month", stepmode="backward"), dict(step="all", label="ALL")], bgcolor="#1c2030", activecolor="#4361ee", bordercolor="#363c4e", font=dict(color="#d1d4dc", size=10))

    layout_update = dict(
        title=dict(text=title_html, font=dict(size=14, color="#d1d4dc")),
        paper_bgcolor="#131722", plot_bgcolor="#131722", font=dict(color="#d1d4dc"),
        hovermode="x unified", legend=dict(x=0.01, y=0.99, bgcolor="rgba(19,23,34,0.85)", font=dict(size=9)),
        margin=dict(l=10, r=60, t=50, b=10), dragmode="pan",
        xaxis =dict(type="date", **gs, **ts, **ls, showticklabels=False),
    )

    # 动态配置 Y 轴和 X 轴滑块
    if rows == 3:
        layout_update.update({
            "xaxis2": dict(type="date", **gs, **ts, **ls, showticklabels=False),
            "xaxis3": dict(type="date", **gs, **ts, **ls, rangeslider=dict(visible=True, bgcolor="#1c2030", bordercolor="#2a2e3e", thickness=0.08), rangeselector=dict(**rng_sel, x=0.0, y=1.2)),
            "yaxis":  dict(side="right", **gs, **ts, **ls),
            "yaxis2": dict(side="right", **gs, **ts, **ls, title=dict(text="SQZ", font=dict(size=9, color="#787b86"))),
            "yaxis3": dict(side="right", **gs, **ts, **ls, title=dict(text="CMF", font=dict(size=9, color="#787b86")))
        })
    elif rows == 2:
         layout_update.update({
            "xaxis2": dict(type="date", **gs, **ts, **ls, rangeslider=dict(visible=True, bgcolor="#1c2030", bordercolor="#2a2e3e", thickness=0.08), rangeselector=dict(**rng_sel, x=0.0, y=1.2)),
            "yaxis":  dict(side="right", **gs, **ts, **ls),
            "yaxis2": dict(side="right", **gs, **ts, **ls, title=dict(text="CMF", font=dict(size=9, color="#787b86"))),
        })

    fig.update_layout(**layout_update)
    return fig

