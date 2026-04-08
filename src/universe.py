from __future__ import annotations

from io import StringIO

import pandas as pd
import requests


SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
}


def _normalize_yahoo_ticker(symbol: str) -> str:
    return symbol.strip().replace(".", "-")


def load_universe(universe_name: str) -> list[str]:
    if universe_name != "sp500":
        raise ValueError(f"Unsupported universe: {universe_name}")

    response = requests.get(SP500_WIKI_URL, headers=DEFAULT_HEADERS, timeout=30)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    if not tables:
        raise RuntimeError("Failed to load S&P 500 constituent table.")

    sp500_table = tables[0]
    symbol_col = "Symbol" if "Symbol" in sp500_table.columns else sp500_table.columns[0]
    tickers = [
        _normalize_yahoo_ticker(symbol)
        for symbol in sp500_table[symbol_col].dropna().astype(str).tolist()
    ]
    return sorted(set(tickers))
