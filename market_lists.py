"""Ticker list fetchers and curated watchlists for US and Canada scanners."""
from __future__ import annotations

import io
import string
import time
from typing import Callable, Literal

import pandas as pd
import requests

Market = Literal["us", "ca"]

US_MIN_PRICE = 2.0
US_MIN_AVG_VOL = 300_000
CA_MIN_PRICE = 1.0
CA_MIN_AVG_VOL = 100_000

# NASDAQ Trader otherlisted Exchange codes
NYSE_EXCHANGE_CODE = "N"  # New York Stock Exchange only (not Arca / American)

US_CURATED = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AVGO", "AMD", "INTC",
    "QCOM", "ORCL", "CRM", "NOW", "ADBE", "SHOP", "PLTR", "SNOW", "UBER", "LYFT",
    "TSM", "ASML", "MU", "AMAT", "LRCX", "KLAC", "MRVL", "ARM", "SMCI", "DELL",
    "JPM", "BAC", "GS", "MS", "V", "MA", "BRK-B", "AXP", "BX", "KKR",
    "XOM", "CVX", "COP", "EOG", "SLB", "HAL", "OXY", "MPC", "VLO", "PSX",
    "UNH", "LLY", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BSX",
    "WMT", "COST", "TGT", "HD", "LOW", "MCD", "SBUX", "NKE", "PG", "KO",
    "BABA", "PDD", "JD", "BIDU", "NIO", "XPEV", "LI", "TME", "VIPS", "FUTU",
    "SPY", "QQQ", "GLD", "SLV", "GDX", "GDXJ", "XLE", "XLF", "XLK", "TLT",
    "IWM", "IWN", "IWO", "AAL", "DAL", "UAL", "LUV", "BA",
]

CA_CURATED = [
    "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO",
    "MFC.TO", "SLF.TO", "GWO.TO", "IFC.TO", "FFH.TO", "IAG.TO",
    "CNQ.TO", "SU.TO", "CVE.TO", "ENB.TO", "TRP.TO", "PPL.TO",
    "ARC.TO", "MEG.TO", "WCP.TO", "ARX.TO", "BTE.TO", "ERF.TO",
    "ABX.TO", "AEM.TO", "FNV.TO", "WPM.TO", "K.TO", "G.TO",
    "EDV.TO", "SSL.TO", "OGC.TO",
    "FM.TO", "TECK-B.TO", "CS.TO", "LUN.TO", "HBM.TO",
    "SHOP.TO", "CSU.TO", "OTEX.TO", "BB.TO", "LSPD.TO",
    "DCBO.TO", "DSG.TO",
    "BCE.TO", "T.TO", "RCI-B.TO", "QBR-B.TO",
    "CP.TO", "CNR.TO", "TFI.TO", "MTL.TO",
    "L.TO", "WN.TO", "MRU.TO", "DOL.TO", "ATD.TO", "EMP-A.TO",
    "REI-UN.TO", "HR-UN.TO", "CAR-UN.TO", "CRT-UN.TO",
    "SRU-UN.TO", "AP-UN.TO",
    "FTS.TO", "EMA.TO", "NPI.TO", "ALA.TO", "BLX.TO",
    "MG.TO", "TRI.TO", "BAM.TO", "BN.TO", "WSP.TO", "STN.TO",
    "XIU.TO", "XIC.TO", "ZEB.TO", "ZEN.TO", "XGD.TO",
    "HGU.TO", "HGD.TO", "ZUB.TO",
]

US_MODE_OPTIONS = {
    "curated": "Curated (fast)",
    "nasdaq": "All NASDAQ",
    "nyse": "All NYSE",
    "sp500": "S&P 500",
    "us_all": "NYSE + NASDAQ + S&P 500",
    "both": "NYSE + NASDAQ",
    "custom": "Custom",
}

CA_MODE_OPTIONS = {
    "curated": "Curated (fast)",
    "tsx": "All TSX",
    "tsxv": "All TSXV",
    "both": "TSX + TSXV",
    "custom": "Custom",
}


def _normalize_us_symbol(raw: str) -> str | None:
    t = str(raw).strip().upper()
    if not t or len(t) > 6:
        return None
    yf_t = t.replace(".", "-")
    if yf_t.replace("-", "").isalnum():
        return yf_t
    return None


def _is_common_equity_name(raw: str) -> bool:
    """Exclude warrants, units, rights and debt-like listings from stock scans."""
    name = str(raw).strip().lower()
    excluded = (
        "warrant",
        " unit",
        "units",
        " right",
        "rights",
        "preferred",
        "preference",
        "depositary share",
        "depositary shares",
        "note due",
        "notes due",
        "senior note",
        "debenture",
        "bond",
    )
    return bool(name) and not any(term in name for term in excluded)


def fetch_nasdaq_listed() -> list[str]:
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), sep="|")
        df = df[(df["Test Issue"] == "N") & (df["ETF"] == "N")]
        df = df[df["Security Name"].map(_is_common_equity_name)]
        out: list[str] = []
        for s in df["Symbol"].dropna().tolist():
            yf_t = _normalize_us_symbol(s)
            if yf_t:
                out.append(yf_t)
        return out
    except Exception as e:
        print(f"[WARN] NASDAQ: {e}")
        return []


def fetch_nyse_listed() -> list[str]:
    """Strict NYSE common stocks from NASDAQ Trader otherlisted (Exchange == N)."""
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), sep="|")
        etf_col = "ETF/Investment Fund" if "ETF/Investment Fund" in df.columns else "ETF"
        df = df[
            (df["Test Issue"] == "N")
            & (df[etf_col] == "N")
            & (df["Exchange"] == NYSE_EXCHANGE_CODE)
        ]
        df = df[df["Security Name"].map(_is_common_equity_name)]
        out: list[str] = []
        for s in df["ACT Symbol"].dropna().tolist():
            yf_t = _normalize_us_symbol(s)
            if yf_t:
                out.append(yf_t)
        return out
    except Exception as e:
        print(f"[WARN] NYSE: {e}")
        return []


def fetch_sp500_listed() -> list[str]:
    """S&P 500 constituents (Yahoo-friendly symbols). Prefer GitHub dataset, fall back to Wikipedia."""
    sources = [
        (
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
            "Symbol",
        ),
        (
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            "Symbol",
        ),
    ]
    for url, col in sources:
        try:
            if url.endswith(".csv"):
                r = requests.get(url, timeout=20)
                r.raise_for_status()
                df = pd.read_csv(io.StringIO(r.text))
            else:
                tables = pd.read_html(url)
                df = tables[0]
            if col not in df.columns:
                # Wikipedia sometimes uses different casing
                match = next((c for c in df.columns if str(c).lower() == "symbol"), None)
                if match is None:
                    continue
                col = match
            out: list[str] = []
            for s in df[col].dropna().tolist():
                yf_t = _normalize_us_symbol(str(s).replace(".", "-"))
                if yf_t:
                    out.append(yf_t)
            if out:
                return list(dict.fromkeys(out))
        except Exception as e:
            print(f"[WARN] S&P 500 ({url}): {e}")
    return []


def _tmx_fetch(market: str, suffix: str) -> list[str]:
    all_syms: list[str] = []
    for letter in string.ascii_lowercase:
        url = (
            f"https://www.tsx.com/json/company-directory"
            f"/search/{market}/{letter}"
        )
        try:
            r = requests.get(
                url,
                timeout=15,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
                    )
                },
            )
            r.raise_for_status()
            data = r.json()
            for item in data.get("results", []):
                sym = str(item.get("symbol", "")).strip().upper()
                if not sym:
                    continue
                yf_sym = sym.replace(".", "-") + suffix
                all_syms.append(yf_sym)
        except Exception as e:
            print(f"[WARN] TMX {market.upper()} '{letter}': {e}")
        time.sleep(0.06)
    return list(dict.fromkeys(all_syms))


def fetch_tsx_listed() -> list[str]:
    return _tmx_fetch("tsx", ".TO")


def fetch_tsxv_listed() -> list[str]:
    return _tmx_fetch("tsxv", ".V")


def _parse_custom_tickers(custom_text: str) -> list[str]:
    raw = custom_text.strip()
    return [t.strip().upper() for t in raw.replace(";", ",").split(",") if t.strip()]


def merge_ticker_sources(source_map: dict[str, list[str]]) -> tuple[list[str], dict[str, str]]:
    """
    Deduplicate tickers across pools while preserving source labels.
    Returns (tickers, sources) where sources[ticker] is e.g. 'NYSE+SP500'.
    """
    tag_map: dict[str, list[str]] = {}
    for label, tickers in source_map.items():
        for t in tickers:
            tag_map.setdefault(t, [])
            if label not in tag_map[t]:
                tag_map[t].append(label)
    tickers = list(tag_map.keys())
    sources = {t: "+".join(tags) for t, tags in tag_map.items()}
    return tickers, sources


def resolve_tickers(
    market: Market,
    mode: str,
    custom_text: str,
    cache: dict[str, list[str]],
    status_callback: Callable[[str], None] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """
    Resolve ticker list for a scan mode.
    Returns (tickers, source_by_ticker).
    """
    if mode == "custom":
        tickers = _parse_custom_tickers(custom_text)
        return tickers, {t: "custom" for t in tickers}

    if market == "ca":
        if mode == "curated":
            tickers = list(CA_CURATED)
            return tickers, {t: "curated" for t in tickers}
        exchanges = {"tsx": ["tsx"], "tsxv": ["tsxv"], "both": ["tsx", "tsxv"]}[mode]
        fetchers = {"tsx": fetch_tsx_listed, "tsxv": fetch_tsxv_listed}
        labels = {"tsx": "TSX", "tsxv": "TSXV"}
        source_map: dict[str, list[str]] = {}
        for ex in exchanges:
            if ex not in cache:
                if status_callback:
                    status_callback(f"Fetching {ex.upper()} symbol list…")
                cache[ex] = fetchers[ex]()
                if status_callback:
                    status_callback(f"Loaded {len(cache[ex])} tickers from {ex.upper()}")
            source_map[labels[ex]] = cache[ex]
        return merge_ticker_sources(source_map)

    # US
    if mode == "curated":
        tickers = list(US_CURATED)
        return tickers, {t: "curated" for t in tickers}

    mode_exchanges = {
        "nasdaq": ["nasdaq"],
        "nyse": ["nyse"],
        "sp500": ["sp500"],
        "both": ["nasdaq", "nyse"],
        "us_all": ["nasdaq", "nyse", "sp500"],
    }
    exchanges = mode_exchanges[mode]
    fetchers = {
        "nasdaq": fetch_nasdaq_listed,
        "nyse": fetch_nyse_listed,
        "sp500": fetch_sp500_listed,
    }
    labels = {"nasdaq": "NASDAQ", "nyse": "NYSE", "sp500": "SP500"}
    source_map = {}
    for ex in exchanges:
        if ex not in cache:
            if status_callback:
                status_callback(f"Fetching {ex.upper()} symbol list…")
            cache[ex] = fetchers[ex]()
            if status_callback:
                status_callback(f"Loaded {len(cache[ex])} tickers from {ex.upper()}")
        source_map[labels[ex]] = cache[ex]
    return merge_ticker_sources(source_map)


def market_defaults(market: Market) -> tuple[float, int, str]:
    """Return (min_price, min_avg_vol, price_label) for a market."""
    if market == "us":
        return US_MIN_PRICE, US_MIN_AVG_VOL, "USD"
    return CA_MIN_PRICE, CA_MIN_AVG_VOL, "CAD"
