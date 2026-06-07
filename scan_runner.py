"""Headless TD buy scan loop for Streamlit (mirrors tkinter scanner _scan_thread)."""
from __future__ import annotations

import random
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PARENT_DIR = APP_DIR.parent
# Streamlit Cloud deploys TDdetector/ as repo root; core lives alongside scan_runner.
# Local dev may still use ../td_scanner_core.py only — check parent as fallback.
for path in (PARENT_DIR, APP_DIR):
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)

from td_scanner_core import ScanStats, analyse, batch_download

LOOKBACK = 500


@dataclass
class ScanUpdate:
    done: int
    total: int
    found_count: int
    elapsed_sec: float


@dataclass
class ScanResult:
    results: list[dict]
    stats: ScanStats
    elapsed_sec: float


@dataclass
class ScanParams:
    signal_days: int
    min_price: float
    min_avg_vol: float
    batch_size: int
    batch_delay: float
    apply_liquidity_filter: bool


def run_scan(
    tickers: list[str],
    params: ScanParams,
    stop_flag: Callable[[], bool],
) -> Iterator[ScanUpdate | ScanResult]:
    """
    Yield ScanUpdate progress during scan, then a final ScanResult.
    stop_flag() should return True when the user requests stop.
    """
    if not tickers:
        yield ScanResult(results=[], stats=ScanStats(), elapsed_sec=0.0)
        return

    batches = [tickers[i : i + params.batch_size] for i in range(0, len(tickers), params.batch_size)]
    total = len(tickers)
    done = 0
    found_all: list[dict] = []
    stats = ScanStats()
    start = time.time()

    for batch in batches:
        if stop_flag():
            break
        data = batch_download(batch, LOOKBACK)
        for sym in batch:
            if stop_flag():
                break
            done += 1
            d = data.get(sym)
            if d is None:
                stats.failed_download.append(sym)
            else:
                close, low, vol = d
                try:
                    rows, skipped = analyse(
                        sym,
                        close,
                        low,
                        vol,
                        params.signal_days,
                        params.min_price,
                        params.min_avg_vol,
                        apply_liquidity_filter=params.apply_liquidity_filter,
                    )
                    stats.liquidity_skipped += skipped
                    found_all.extend(rows)
                except Exception as exc:
                    stats.failed_analyse.append(sym)
                    print(f"[WARN] analyse {sym}: {exc}")

            yield ScanUpdate(
                done=done,
                total=total,
                found_count=len(found_all),
                elapsed_sec=time.time() - start,
            )

        if stop_flag():
            break
        time.sleep(params.batch_delay + random.uniform(0, 0.3))

    sorted_results = sorted(found_all, key=lambda x: x["days_ago"])
    yield ScanResult(
        results=sorted_results,
        stats=stats,
        elapsed_sec=time.time() - start,
    )
