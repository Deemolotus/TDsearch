"""Two-stage multi-indicator scan loop for Streamlit."""
from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from td_scanner_core import (
    LOOKBACK_STAGE1,
    LOOKBACK_STAGE2,
    CandidateFeatures,
    ScanStats,
    ScoreWeights,
    batch_download,
    compute_stage1_features,
    enrich_cycle_features,
    score_candidate,
)


@dataclass
class ScanUpdate:
    done: int
    total: int
    found_count: int
    elapsed_sec: float
    phase: str = "stage1"  # stage1 | stage2 | done


@dataclass
class ScanResult:
    results: list[dict]
    stats: ScanStats
    elapsed_sec: float


@dataclass
class ScanParams:
    signal_days: int = 10
    min_price: float = 2.0
    min_avg_vol: float = 300_000
    batch_size: int = 20
    batch_delay: float = 0.4
    apply_liquidity_filter: bool = True
    turn_lookback: int = 10
    cycle_shortlist: int = 50
    enable_cycle: bool = True
    future_days: int = 65
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    source_by_ticker: dict[str, str] = field(default_factory=dict)


def run_scan(
    tickers: list[str],
    params: ScanParams,
    stop_flag: Callable[[], bool],
) -> Iterator[ScanUpdate | ScanResult]:
    """
    Yield ScanUpdate progress during scan, then a final ScanResult.

    Stage 1: download ~500 bars, hard-filter liquidity + recent TD, score CMF/SQZ/ST.
    Stage 2: re-download shortlist with ~900 bars, enrich cycle projection, re-score.
    """
    if not tickers:
        yield ScanResult(results=[], stats=ScanStats(), elapsed_sec=0.0)
        return

    batches = [tickers[i : i + params.batch_size] for i in range(0, len(tickers), params.batch_size)]
    total = len(tickers)
    done = 0
    candidates: list[CandidateFeatures] = []
    stats = ScanStats()
    start = time.time()

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    for batch in batches:
        if stop_flag():
            break
        data = batch_download(batch, LOOKBACK_STAGE1)
        for sym in batch:
            if stop_flag():
                break
            done += 1
            d = data.get(sym)
            if d is None:
                stats.failed_download.append(sym)
            else:
                high, low, close, vol = d
                try:
                    feat, skip = compute_stage1_features(
                        sym, high, low, close, vol,
                        signal_days=params.signal_days,
                        min_price=params.min_price,
                        min_vol=params.min_avg_vol,
                        apply_liquidity_filter=params.apply_liquidity_filter,
                        turn_lookback=params.turn_lookback,
                        source=params.source_by_ticker.get(sym, ""),
                    )
                    if skip == "liquidity":
                        stats.liquidity_skipped += 1
                    elif skip == "no_td":
                        stats.no_td_skipped += 1
                    elif feat is not None:
                        feat = score_candidate(feat, params.weights, params.signal_days)
                        candidates.append(feat)
                except Exception as exc:
                    stats.failed_analyse.append(sym)
                    print(f"[WARN] analyse {sym}: {exc}")

            yield ScanUpdate(
                done=done,
                total=total,
                found_count=len(candidates),
                elapsed_sec=time.time() - start,
                phase="stage1",
            )

        if stop_flag():
            break
        time.sleep(params.batch_delay + random.uniform(0, 0.3))

    # Preliminary rank (without cycle or with zero cycle score)
    candidates.sort(key=lambda c: c.score, reverse=True)

    # ── Stage 2: cycle enrichment on shortlist ───────────────────────────────
    if params.enable_cycle and candidates and not stop_flag():
        shortlist = candidates[: max(1, params.cycle_shortlist)]
        stats.cycle_attempted = len(shortlist)
        short_tickers = [c.ticker for c in shortlist]
        stage2_batches = [
            short_tickers[i : i + params.batch_size]
            for i in range(0, len(short_tickers), params.batch_size)
        ]
        ohlcv: dict = {}
        for batch in stage2_batches:
            if stop_flag():
                break
            ohlcv.update(batch_download(batch, LOOKBACK_STAGE2))
            time.sleep(params.batch_delay + random.uniform(0, 0.2))

        by_ticker = {c.ticker: c for c in candidates}
        stage2_done = 0
        for sym in short_tickers:
            if stop_flag():
                break
            stage2_done += 1
            d = ohlcv.get(sym)
            feat = by_ticker[sym]
            if d is None:
                feat.cycle_error = "download failed"
                stats.cycle_failed += 1
            else:
                high, low, close, vol = d
                try:
                    enrich_cycle_features(
                        feat, high, low, close, vol,
                        future_days=params.future_days,
                    )
                    if feat.cycle_error:
                        stats.cycle_failed += 1
                except Exception as exc:
                    feat.cycle_error = str(exc)[:120]
                    stats.cycle_failed += 1
            feat = score_candidate(feat, params.weights, params.signal_days)
            by_ticker[sym] = feat
            yield ScanUpdate(
                done=stage2_done,
                total=len(short_tickers),
                found_count=len(candidates),
                elapsed_sec=time.time() - start,
                phase="stage2",
            )
        candidates = list(by_ticker.values())

    candidates.sort(key=lambda c: (-c.score, c.bars_ago, c.ticker))
    rows = [c.to_row() for c in candidates]
    yield ScanResult(
        results=rows,
        stats=stats,
        elapsed_sec=time.time() - start,
    )
