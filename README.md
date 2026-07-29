# Multi-Indicator Equity Scanner (Streamlit)

Standalone hybrid screener for **US** and **Canadian** equities. Hard-filters on
liquidity and recent TD Sequential buy signals, then ranks candidates with CMF,
Squeeze Momentum (拥挤值 proxy), SuperTrend, and a cycle projection model.

This repo is independent of the single-ticker CycleAnalysis app. Indicator logic
ships in the bundled [`quant_engine.py`](quant_engine.py).

## Strategy (hybrid)

| Stage | What | Role |
|-------|------|------|
| Hard filter | Latest price / 20d avg volume; TD9 / TD13 / TD15 buy in last N trading days (default **10 ≈ 2 weeks**) | Must pass |
| Soft score | CMF negative→positive; SQZ `val` negative→positive; SuperTrend bullish; cycle projection slope up | Rank 0–100 |

Scores are explainable sub-weights (TD / CMF / SQZ / SuperTrend / Cycle), normalized to 100.
Cycle projection is a **ranking hint only** — not a probability or target price.

**CMF disclaimer:** Chaikin Money Flow is a price-range × volume proxy. It is **not**
true institutional net inflow.

## Scan scopes (US)

- Curated watchlist (fast)
- All NASDAQ (common stocks, ETF filtered)
- All NYSE (strict `Exchange == N` from NASDAQ Trader otherlisted)
- S&P 500 constituents
- NYSE + NASDAQ
- NYSE + NASDAQ + S&P 500 (deduped; source tags preserved, e.g. `NYSE+SP500`)
- Custom tickers

Canada: curated / TSX / TSXV / both / custom (unchanged).

## Requirements

- Python 3.11+
- Dependencies in [`requirements.txt`](requirements.txt)

## Local setup

```bash
cd TDsearch-main
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the URL shown in the terminal (default `http://localhost:8501`).

**Tip:** Start with **Curated** or a small **Custom** list. Full NYSE+NASDAQ+SP500
scans take a long time and may hit Yahoo Finance rate limits.

## How the pipeline works

1. **Stage 1** — batch-download ~500 bars of High/Low/Close/Volume; require liquidity
   and a TD buy in the signal window; compute CMF / SQZ / SuperTrend features and a
   preliminary score.
2. **Stage 2** — re-download the top *Cycle shortlist* (~900 bars); run DFE cycle fit
   + future projection slope / R² / strength; re-score and sort descending by total.

Failed cycle enrichment keeps the candidate with `cycle_error` set (score cycle = 0).

## Tests

```bash
pip install pytest
pytest tests/ -q
```

Tests use synthetic OHLCV only (no live Yahoo calls).

## Project layout

```
TDsearch-main/
├── streamlit_app.py     # UI
├── market_lists.py      # Symbol pools (NYSE / NASDAQ / SP500 / CA)
├── scan_runner.py       # Two-stage scan loop
├── td_scanner_core.py   # Filters, scoring, Yahoo download
├── quant_engine.py      # Bundled indicators + cycle model
├── run_full_market_scan.py
├── tests/
├── requirements.txt
└── README.md
```

## Data sources

- Price/volume: Yahoo Finance via `yfinance` (`auto_adjust=False`)
- US symbols: NASDAQ Trader symbol directories
- S&P 500: public constituents CSV (GitHub datasets) with Wikipedia fallback
- Canadian symbols: TMX company directory API

## Known limits

- Yahoo can throttle or drop symbols; failures appear in scan stats.
- Cycle R² is in-sample and optimistic; do not treat as forecast skill.
- Historical cycle overlay has look-ahead in fitting — the scanner only uses the
  *forward* projection slope for ranking.
- TD implementation is a practical simplification vs full DeMark rules.

## Disclaimer

Research / education only. Not financial advice.

## License

MIT License — see [LICENSE](LICENSE).
