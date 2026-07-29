"""Run the full US universe scan and persist ranked results."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from market_lists import resolve_tickers
from scan_runner import ScanParams, ScanResult, ScanUpdate, run_scan
from td_scanner_core import ScoreWeights


OUTPUT_DIR = Path(__file__).resolve().parent / "scan_results"
CSV_PATH = OUTPUT_DIR / "full_us_2026-07-17.csv"
SUMMARY_PATH = OUTPUT_DIR / "full_us_2026-07-17-summary.json"


def main() -> None:
    cache: dict[str, list[str]] = {}
    tickers, sources = resolve_tickers(
        "us",
        "us_all",
        "",
        cache,
        status_callback=lambda message: print(message, flush=True),
    )
    print(f"UNIVERSE_READY count={len(tickers)}", flush=True)

    params = ScanParams(
        signal_days=10,
        min_price=2.0,
        min_avg_vol=300_000,
        batch_size=25,
        batch_delay=0.5,
        apply_liquidity_filter=True,
        turn_lookback=10,
        cycle_shortlist=100,
        enable_cycle=True,
        future_days=65,
        weights=ScoreWeights(),
        source_by_ticker=sources,
    )

    final: ScanResult | None = None
    for update in run_scan(tickers, params, lambda: False):
        if isinstance(update, ScanUpdate):
            if update.done % 100 == 0 or update.phase == "stage2":
                print(
                    f"PROGRESS phase={update.phase} "
                    f"done={update.done}/{update.total} "
                    f"candidates={update.found_count} "
                    f"elapsed={update.elapsed_sec:.1f}",
                    flush=True,
                )
        else:
            final = update

    if final is None:
        raise RuntimeError("Scan ended without a final result")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(final.results[0].keys()) if final.results else []
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final.results)

    summary = {
        "universe_count": len(tickers),
        "result_count": len(final.results),
        "stats": asdict(final.stats),
        "elapsed_sec": final.elapsed_sec,
        "top_results": final.results[:30],
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"SCAN_COMPLETE results={len(final.results)}", flush=True)
    print(f"CSV_PATH {CSV_PATH}", flush=True)
    print(f"SUMMARY_PATH {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
