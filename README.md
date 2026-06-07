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
- Parent module [`td_scanner_core.py`](../td_scanner_core.py) (must stay in the repo root next to `TDdetector/`)

## Local setup

From the `TD9ETC` directory:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r TDdetector/requirements.txt
streamlit run TDdetector/streamlit_app.py
```

Open the URL shown in the terminal (default: `http://localhost:8501`).

## Streamlit Cloud

1. Push this repo to GitHub (include `TDdetector/` and `td_scanner_core.py`).
2. Create a new app on [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Set **Main file path** to `TDdetector/streamlit_app.py`.
4. Point dependencies to `TDdetector/requirements.txt` (or copy those lines into a root `requirements.txt`).

**Tip:** Start with **Curated (fast)** mode. Full-exchange scans can take 30+ minutes and may hit hosted runtime limits.

## Project layout

```
TDdetector/
├── streamlit_app.py    # Streamlit UI entry point
├── market_lists.py     # Exchange symbol lists and curated tickers
├── scan_runner.py      # Headless scan loop
├── requirements.txt
├── README.md
└── LICENSE

../td_scanner_core.py   # Shared TD engine (required)
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
