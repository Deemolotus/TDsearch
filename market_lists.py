"""Ticker list fetchers and curated watchlists for US and Canada TD scanners."""
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


def fetch_nasdaq_listed() -> list[str]:
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    try:
        r = requests.get(url, timeout=15)
        df = pd.read_csv(io.StringIO(r.text), sep="|")
        df = df[(df["Test Issue"] == "N") & (df["ETF"] == "N")]
        syms = df["Symbol"].dropna().tolist()
        out = []
        for s in syms:
            t = str(s).strip().upper()
            if not t or len(t) > 6:
                continue
            yf_t = t.replace(".", "-")
            if yf_t.replace("-", "").isalnum():
                out.append(yf_t)
        return out
    except Exception as e:
        print(f"[WARN] NASDAQ: {e}")
        return []


def fetch_nyse_listed() -> list[str]:
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
    try:
        r = requests.get(url, timeout=15)
        df = pd.read_csv(io.StringIO(r.text), sep="|")
        df = df[(df["Test Issue"] == "N") & (df["ETF/Investment Fund"] == "N")]
        syms = df["ACT Symbol"].dropna().tolist()
        out = []
        for s in syms:
            t = str(s).strip().upper()
            if not t or len(t) > 6:
                continue
            yf_t = t.replace(".", "-")
            if yf_t.replace("-", "").isalnum():
                out.append(yf_t)
        return out
    except Exception as e:
        print(f"[WARN] NYSE: {e}")
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


def resolve_tickers(
    market: Market,
    mode: str,
    custom_text: str,
    cache: dict[str, list[str]],
    status_callback: Callable[[str], None] | None = None,
) -> list[str]:
    """Resolve ticker list for a scan mode, using cache for exchange-wide lists."""
    if mode == "custom":
        return _parse_custom_tickers(custom_text)

    if market == "us":
        if mode == "curated":
            return list(US_CURATED)
        exchanges = {"nasdaq": ["nasdaq"], "nyse": ["nyse"], "both": ["nasdaq", "nyse"]}[mode]
        fetchers = {"nasdaq": fetch_nasdaq_listed, "nyse": fetch_nyse_listed}
    else:
        if mode == "curated":
            return list(CA_CURATED)
        exchanges = {"tsx": ["tsx"], "tsxv": ["tsxv"], "both": ["tsx", "tsxv"]}[mode]
        fetchers = {"tsx": fetch_tsx_listed, "tsxv": fetch_tsxv_listed}

    all_t: list[str] = []
    for ex in exchanges:
        if ex not in cache:
            if status_callback:
                status_callback(f"Fetching {ex.upper()} symbol list…")
            cache[ex] = fetchers[ex]()
            if status_callback:
                status_callback(f"Loaded {len(cache[ex])} tickers from {ex.upper()}")
        all_t += cache[ex]
    return list(dict.fromkeys(all_t))


def market_defaults(market: Market) -> tuple[float, int, str]:
    """Return (min_price, min_avg_vol, price_label) for a market."""
    if market == "us":
        return US_MIN_PRICE, US_MIN_AVG_VOL, "USD"
    return CA_MIN_PRICE, CA_MIN_AVG_VOL, "CAD"
