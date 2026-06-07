# TD Buy Signal Scanner (Streamlit)

Web app for scanning **TD Sequential buy signals** (TD9, TD13, TD15) on US and Canadian equities. Built with [Streamlit](https://streamlit.io/) and shares the scan engine with the desktop tkinter scanners in the parent folder.

## Features

- **US Scanner** — NASDAQ, NYSE, curated watchlist, or custom tickers (USD)
- **Canada Scanner** — TSX (`.TO`), TSXV (`.V`), curated list, or custom tickers (CAD)
- Configurable signal window, min price, min average volume, batch size, and liquidity filter
- Live progress and ETA during scans
- Color-coded results table and CSV download

## Requirements

- Python 3.11+
- Dependencies in [`requirements.txt`](requirements.txt)

## Local setup

**Option A — run from this folder** (matches Streamlit Cloud / GitHub repo layout):

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
streamlit run streamlit_app.py
```

**Option B — run from parent `TD9ETC`** (full toolkit checkout):

```bash
pip install -r TDdetector/requirements.txt
streamlit run TDdetector/streamlit_app.py
```

Open the URL shown in the terminal (default: `http://localhost:8501`).

## Streamlit Cloud

1. Push this folder to GitHub as the repo root (all files in `TDdetector/`, including `td_scanner_core.py`).
2. Create a new app on [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Set **Main file path** to `streamlit_app.py`.
4. Point dependencies to `requirements.txt`.

**Tip:** Start with **Curated (fast)** mode. Full-exchange scans can take 30+ minutes and may hit hosted runtime limits.

## Project layout

```
├── streamlit_app.py    # Streamlit UI entry point
├── market_lists.py     # Exchange symbol lists and curated tickers
├── scan_runner.py      # Headless scan loop
├── td_scanner_core.py  # TD scan engine (bundled for deployment)
├── requirements.txt
├── README.md
└── LICENSE
```

## Related desktop apps

| Script | Market |
|--------|--------|
| [`td_buy_scanner_v32.py`](../td_buy_scanner_v32.py) | US (tkinter GUI) |
| [`canada_td_scanner.py`](../canada_td_scanner.py) | Canada (tkinter GUI) |

## Data sources

- Price/volume: [Yahoo Finance](https://finance.yahoo.com/) via `yfinance`
- US symbols: NASDAQ Trader symbol directory
- Canadian symbols: [TMX](https://www.tsx.com/) company directory API

## Disclaimer

This tool is for research and education only. It is not financial advice.

## License

MIT License — see [LICENSE](LICENSE).
