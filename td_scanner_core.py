"""
Multi-indicator scanner engine for US / Canada equities.

Uses local quant_engine indicators (TD Sequential, CMF, SuperTrend, Squeeze Momentum,
cycle projection) and applies a hybrid hard-filter + scored ranking pipeline.
"""
from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

from quant_engine import (
    DEFAULT_FUTURE_DAYS,
    calculate_supertrend,
    chaikin_money_flow,
    metrics_for_window,
    resolve_cycle_config,
    squeeze_momentum,
    td_sequential,
)

MAX_RETRIES = 3
LOOKBACK_STAGE1 = 500
LOOKBACK_STAGE2 = 900
MIN_BARS_STAGE1 = 60

TD_KIND_TO_LABEL = {
    "buy9": "TD9",
    "buy13": "TD13",
    "buy15": "TD15",
}
TD_LABEL_SCORE = {"TD9": 12.0, "TD13": 18.0, "TD15": 25.0}


@dataclass
class ScanStats:
    failed_download: list[str] = field(default_factory=list)
    failed_analyse: list[str] = field(default_factory=list)
    liquidity_skipped: int = 0
    no_td_skipped: int = 0
    cycle_attempted: int = 0
    cycle_failed: int = 0


@dataclass
class ScoreWeights:
    td: float = 25.0
    cmf: float = 20.0
    sqz: float = 20.0
    supertrend: float = 15.0
    cycle: float = 20.0

    def normalized(self) -> "ScoreWeights":
        total = self.td + self.cmf + self.sqz + self.supertrend + self.cycle
        if total <= 0:
            return ScoreWeights()
        scale = 100.0 / total
        return ScoreWeights(
            td=self.td * scale,
            cmf=self.cmf * scale,
            sqz=self.sqz * scale,
            supertrend=self.supertrend * scale,
            cycle=self.cycle * scale,
        )


@dataclass
class CandidateFeatures:
    ticker: str
    source: str = ""
    signal: str = ""
    sig_date: str = ""
    days_ago: int = 0
    bars_ago: int = 0
    price: float = 0.0
    avg_vol: float = 0.0
    cmf: float = 0.0
    cmf_turn_pos: bool = False
    sqz: float = 0.0
    sqz_turn_pos: bool = False
    supertrend_bull: bool = False
    st_line: float = 0.0
    cycle_up: bool = False
    cycle_slope: float = 0.0
    cycle_r2: float = 0.0
    cycle_strength: float = 0.0
    cycle_error: str = ""
    score_td: float = 0.0
    score_cmf: float = 0.0
    score_sqz: float = 0.0
    score_st: float = 0.0
    score_cycle: float = 0.0
    score: float = 0.0
    reasons: str = ""
    missing: str = ""

    def to_row(self) -> dict:
        row = asdict(self)
        row["price"] = f"{self.price:.2f}"
        row["cmf"] = f"{self.cmf:.3f}"
        row["sqz"] = f"{self.sqz:.4f}"
        row["st_line"] = f"{self.st_line:.2f}"
        row["cycle_slope"] = f"{self.cycle_slope:.4f}"
        row["cycle_r2"] = f"{self.cycle_r2:.3f}"
        row["cycle_strength"] = f"{self.cycle_strength:.2f}"
        row["score"] = round(self.score, 1)
        row["score_td"] = round(self.score_td, 1)
        row["score_cmf"] = round(self.score_cmf, 1)
        row["score_sqz"] = round(self.score_sqz, 1)
        row["score_st"] = round(self.score_st, 1)
        row["score_cycle"] = round(self.score_cycle, 1)
        row["avg_vol"] = int(self.avg_vol)
        return row


def _sq_col(series) -> pd.Series:
    s = series
    return (s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s).dropna()


def _is_multi_ticker_df(df: pd.DataFrame) -> bool:
    return isinstance(df.columns, pd.MultiIndex)


def _parse_symbol_from_df(
    df: pd.DataFrame, sym: str,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series] | None:
    """Extract and align High/Low/Close/Volume for one symbol."""
    try:
        if _is_multi_ticker_df(df):
            high = df[sym]["High"]
            low = df[sym]["Low"]
            close = df[sym]["Close"]
            vol = df[sym]["Volume"]
        else:
            high = df["High"]
            low = df["Low"]
            close = df["Close"]
            vol = df["Volume"]
        close = _sq_col(close)
        if close.empty:
            return None
        idx = close.index
        high = _sq_col(high).reindex(idx)
        low = _sq_col(low).reindex(idx)
        vol = _sq_col(vol).reindex(idx)
        aligned = pd.DataFrame({"h": high, "l": low, "c": close, "v": vol}).dropna()
        if aligned.empty:
            return None
        return aligned["h"], aligned["l"], aligned["c"], aligned["v"]
    except (KeyError, TypeError, ValueError):
        return None


def _download_batch_once(
    tickers: list[str], lookback: int,
) -> dict[str, tuple[pd.Series, pd.Series, pd.Series, pd.Series]]:
    end = pd.Timestamp.today().normalize()
    start = (end - pd.offsets.BDay(lookback + 30)).date().isoformat()
    end_str = (end + pd.offsets.BDay(1)).date().isoformat()
    result: dict[str, tuple[pd.Series, pd.Series, pd.Series, pd.Series]] = {}

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
                high, low, close, vol = parsed
                if len(close) >= MIN_BARS_STAGE1:
                    result[sym] = (
                        high.tail(lookback),
                        low.tail(lookback),
                        close.tail(lookback),
                        vol.tail(lookback),
                    )
            return result
        except Exception:
            time.sleep(2 ** attempt + random.uniform(0.5, 1.5))
    return result


def batch_download(
    tickers: list[str], lookback: int,
) -> dict[str, tuple[pd.Series, pd.Series, pd.Series, pd.Series]]:
    """Download OHLCV for tickers with missing-symbol retries."""
    if not tickers:
        return {}

    result = _download_batch_once(tickers, lookback)
    missing = [t for t in tickers if t not in result]
    if not missing:
        return result

    for i in range(0, len(missing), 2):
        chunk = missing[i : i + 2]
        result.update(_download_batch_once(chunk, lookback))

    still_missing = [t for t in tickers if t not in result]
    for sym in still_missing:
        result.update(_download_batch_once([sym], lookback))

    return result


def _avg_vol_at(vol: pd.Series, end_idx: int) -> float:
    start = max(0, end_idx - 19)
    window = vol.iloc[start : end_idx + 1]
    if window.empty:
        return 0.0
    return float(window.mean())


def crossed_neg_to_pos(values: np.ndarray | pd.Series, lookback: int) -> tuple[bool, float]:
    """
    True if the latest value is > 0 and a non-positive value existed in the prior
    lookback-1 bars (inclusive window of `lookback` bars ending at latest).
    Returns (turned_positive, current_value).
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return False, float("nan")
    window = arr[-lookback:] if lookback > 0 else arr
    current = float(window[-1])
    if not np.isfinite(current) or current <= 0:
        return False, current
    prior = window[:-1]
    prior = prior[np.isfinite(prior)]
    if len(prior) == 0:
        return False, current
    return bool(np.any(prior <= 0)), current


def recent_td_buy_signals(
    close: pd.Series, high: pd.Series, low: pd.Series, signal_days: int,
) -> list[dict]:
    """Buy-side TD9/13/15 within the last `signal_days` bars using quant_engine.td_sequential."""
    _, _, signals = td_sequential(close, high, low)
    n = len(close)
    cutoff_idx = max(0, n - signal_days)
    out: list[dict] = []
    for s in signals:
        if s.kind not in TD_KIND_TO_LABEL:
            continue
        if s.idx < cutoff_idx:
            continue
        sig_date = s.date.date() if hasattr(s.date, "date") else s.date
        out.append({
            "type": TD_KIND_TO_LABEL[s.kind],
            "date": sig_date,
            "idx": int(s.idx),
            "perfect": bool(s.perfect),
            "price": float(s.price),
        })
    return out


def pick_primary_td(sigs: list[dict]) -> dict | None:
    """Prefer most recent signal; on same bar prefer TD15 > TD13 > TD9."""
    if not sigs:
        return None
    rank = {"TD15": 3, "TD13": 2, "TD9": 1}
    return sorted(sigs, key=lambda s: (s["idx"], rank.get(s["type"], 0)), reverse=True)[0]


def compute_stage1_features(
    ticker: str,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    vol: pd.Series,
    *,
    signal_days: int,
    min_price: float,
    min_vol: float,
    apply_liquidity_filter: bool,
    turn_lookback: int,
    source: str = "",
) -> tuple[CandidateFeatures | None, str]:
    """
    Hard-filter on liquidity + recent TD buy; compute CMF / SQZ / SuperTrend features.
    Returns (candidate_or_None, skip_reason) where skip_reason is
    'liquidity' | 'no_td' | '' .
    """
    if len(close) < MIN_BARS_STAGE1:
        return None, "no_td"

    price = float(close.iloc[-1])
    avg_vol = _avg_vol_at(vol, len(vol) - 1)
    if apply_liquidity_filter and (price < min_price or avg_vol < min_vol):
        return None, "liquidity"

    sigs = recent_td_buy_signals(close, high, low, signal_days)
    primary = pick_primary_td(sigs)
    if primary is None:
        return None, "no_td"

    cmf_series = chaikin_money_flow(close, high, low, vol)
    cmf_turn, cmf_val = crossed_neg_to_pos(cmf_series.values, turn_lookback)

    sqz_val, *_ = squeeze_momentum(close, high, low)
    sqz_turn, sqz_now = crossed_neg_to_pos(sqz_val, turn_lookback)

    trend, st_line = calculate_supertrend(high, low, close)
    st_bull = bool(trend[-1] == 1 and price > float(st_line[-1]))

    today = date.today()
    sig_date = primary["date"]
    bars_ago = len(close) - 1 - primary["idx"]
    days_ago = (today - sig_date).days if hasattr(sig_date, "year") else bars_ago

    feat = CandidateFeatures(
        ticker=ticker,
        source=source,
        signal=primary["type"],
        sig_date=str(sig_date),
        days_ago=int(days_ago),
        bars_ago=int(bars_ago),
        price=price,
        avg_vol=avg_vol,
        cmf=float(cmf_val) if np.isfinite(cmf_val) else 0.0,
        cmf_turn_pos=cmf_turn,
        sqz=float(sqz_now) if np.isfinite(sqz_now) else 0.0,
        sqz_turn_pos=sqz_turn,
        supertrend_bull=st_bull,
        st_line=float(st_line[-1]) if np.isfinite(st_line[-1]) else 0.0,
    )
    return feat, ""


def enrich_cycle_features(
    feat: CandidateFeatures,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    vol: pd.Series,
    *,
    future_days: int = DEFAULT_FUTURE_DAYS,
) -> CandidateFeatures:
    """Attach cycle projection metrics; never raises — sets cycle_error on failure."""
    try:
        cfg = resolve_cycle_config(
            close, high, low, vol,
            auto_dfe=True,
            future_days=future_days,
            auto_lookback=False,
        )
        m = metrics_for_window(cfg["close"], cfg["periods"], future_days)
        overlay: pd.Series = m["overlay"]
        hist_n = len(cfg["close"])
        proj = overlay.iloc[hist_n:]
        if len(proj) >= 2:
            y = proj.values.astype(float)
            x = np.arange(len(y), dtype=float)
            slope = float(np.polyfit(x, y, 1)[0])
            # Normalize slope by last close so scores are comparable across price levels
            last_px = float(cfg["close"].iloc[-1]) or 1.0
            norm_slope = slope / last_px
        else:
            norm_slope = 0.0
        feat.cycle_slope = norm_slope
        feat.cycle_up = norm_slope > 0
        feat.cycle_r2 = float(m["combined_r2"])
        feat.cycle_strength = float(m["top_strength"])
        feat.cycle_error = ""
    except Exception as exc:
        feat.cycle_error = str(exc)[:120]
        feat.cycle_up = False
        feat.cycle_slope = 0.0
        feat.cycle_r2 = 0.0
        feat.cycle_strength = 0.0
    return feat


def score_candidate(
    feat: CandidateFeatures,
    weights: ScoreWeights,
    signal_days: int,
) -> CandidateFeatures:
    """Explainable 0–100 score from weighted sub-scores."""
    w = weights.normalized()

    # TD: level base + freshness within signal window
    level = TD_LABEL_SCORE.get(feat.signal, 10.0)
    freshness = max(0.0, 1.0 - (feat.bars_ago / max(signal_days, 1)))
    td_raw = level * (0.55 + 0.45 * freshness)  # ~6.6–25
    feat.score_td = min(w.td, td_raw * (w.td / 25.0))

    # CMF: turn + currently positive magnitude
    if feat.cmf_turn_pos and feat.cmf > 0:
        mag = min(1.0, feat.cmf / 0.15)
        feat.score_cmf = w.cmf * (0.7 + 0.3 * mag)
    elif feat.cmf > 0:
        feat.score_cmf = w.cmf * 0.35
    else:
        feat.score_cmf = 0.0

    # SQZ (crowding proxy): turn negative→positive
    if feat.sqz_turn_pos and feat.sqz > 0:
        feat.score_sqz = w.sqz
    elif feat.sqz > 0:
        feat.score_sqz = w.sqz * 0.35
    else:
        feat.score_sqz = 0.0

    # SuperTrend bullish
    feat.score_st = w.supertrend if feat.supertrend_bull else 0.0

    # Cycle future upward (ranking hint only)
    if feat.cycle_error:
        feat.score_cycle = 0.0
    elif feat.cycle_up:
        slope_comp = min(1.0, max(0.0, feat.cycle_slope / 0.002))
        r2_comp = min(1.0, max(0.0, feat.cycle_r2))
        strength_comp = min(1.0, max(0.0, feat.cycle_strength / 2.0))
        feat.score_cycle = w.cycle * (0.45 * slope_comp + 0.30 * r2_comp + 0.25 * strength_comp)
    else:
        feat.score_cycle = 0.0

    feat.score = (
        feat.score_td + feat.score_cmf + feat.score_sqz + feat.score_st + feat.score_cycle
    )

    reasons: list[str] = []
    missing: list[str] = []
    reasons.append(f"{feat.signal} {feat.bars_ago}d ago")
    if feat.cmf_turn_pos:
        reasons.append("CMF−→+")
    elif feat.cmf > 0:
        reasons.append("CMF+")
    else:
        missing.append("CMF turn")
    if feat.sqz_turn_pos:
        reasons.append("SQZ−→+")
    elif feat.sqz > 0:
        reasons.append("SQZ+")
    else:
        missing.append("SQZ turn")
    if feat.supertrend_bull:
        reasons.append("ST↑")
    else:
        missing.append("SuperTrend")
    if feat.cycle_error:
        missing.append(f"cycle:{feat.cycle_error}")
    elif feat.cycle_up:
        reasons.append("Cycle↑")
    else:
        missing.append("Cycle up")

    feat.reasons = "; ".join(reasons)
    feat.missing = "; ".join(missing)
    return feat


# Backward-compatible thin wrappers used by older call sites / tests
def find_td_buy_signals(close: pd.Series, low: pd.Series, signal_days: int) -> list[dict]:
    """Legacy API: needs High — synthesizes High=Close when only Low is available."""
    high = close.copy()
    return recent_td_buy_signals(close, high, low, signal_days)


def analyse(
    ticker: str,
    close: pd.Series,
    low: pd.Series,
    vol: pd.Series,
    signal_days: int,
    min_price: float,
    min_vol: float,
    apply_liquidity_filter: bool = True,
    high: pd.Series | None = None,
    turn_lookback: int = 10,
    source: str = "",
    weights: ScoreWeights | None = None,
) -> tuple[list[dict], int]:
    """
    Legacy-compatible analyse used for simple TD scans.
    When high is provided, runs the multi-indicator stage-1 path and returns scored rows.
    """
    if high is None:
        # Old Close/Low/Volume path — TD + liquidity only
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

    feat, skip = compute_stage1_features(
        ticker, high, low, close, vol,
        signal_days=signal_days,
        min_price=min_price,
        min_vol=min_vol,
        apply_liquidity_filter=apply_liquidity_filter,
        turn_lookback=turn_lookback,
        source=source,
    )
    if skip == "liquidity":
        return [], 1
    if feat is None:
        return [], 0
    feat = score_candidate(feat, weights or ScoreWeights(), signal_days)
    return [feat.to_row()], 0
