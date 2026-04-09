"""
CBOE volatility index data: download, cache, and map to equity tickers.

Available indices (all free via yfinance):
  ^VIX  — CBOE Volatility Index (S&P 500)
  ^VXN  — CBOE Nasdaq-100 Volatility Index
  ^RVX  — CBOE Russell 2000 Volatility Index (placeholder, no tickers mapped yet)
  ^OVX  — CBOE Crude Oil Volatility Index (energy sector)
  ^GVZ  — CBOE Gold Volatility Index (placeholder)

Mapping rationale:
  VXN (Nasdaq/tech): companies in the Nasdaq-100 / tech-heavy index
    AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, ADBE, NFLX, CRM, INTC, CSCO, ORCL

  VIX (S&P 500 general): remaining tickers — financials, healthcare,
    consumer staples, energy, industrials
    JPM, V, WMT, JNJ, PG, KO, PEP, HD, MA, BAC, DIS, PFE, MRK, ABT, NKE, CVX, XOM

  NOTE: This is a static, sector-level mapping. All tickers in the same
  group share the same IV time series. This is a known approximation —
  coarser than per-ticker IV, but free and market-consensus-based.
  If the LPI signal survives with shared IV, per-ticker IV (from options
  chains) would likely strengthen it further.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static mapping: equity ticker → CBOE index symbol
# ---------------------------------------------------------------------------
_TICKER_TO_CBOE = {
    # Nasdaq / tech  → VXN
    "AAPL":  "^VXN",
    "MSFT":  "^VXN",
    "GOOGL": "^VXN",
    "AMZN":  "^VXN",
    "META":  "^VXN",
    "NVDA":  "^VXN",
    "TSLA":  "^VXN",
    "ADBE":  "^VXN",
    "NFLX":  "^VXN",
    "CRM":   "^VXN",
    "INTC":  "^VXN",
    "CSCO":  "^VXN",
    "ORCL":  "^VXN",
    # S&P 500 general → VIX
    "JPM":   "^VIX",
    "V":     "^VIX",
    "WMT":   "^VIX",
    "JNJ":   "^VIX",
    "PG":    "^VIX",
    "KO":    "^VIX",
    "PEP":   "^VIX",
    "HD":    "^VIX",
    "MA":    "^VIX",
    "BAC":   "^VIX",
    "DIS":   "^VIX",
    "PFE":   "^VIX",
    "MRK":   "^VIX",
    "ABT":   "^VIX",
    "NKE":   "^VIX",
    "CVX":   "^VIX",
    "XOM":   "^VIX",
}

# All unique CBOE symbols we need to download
CBOE_SYMBOLS = sorted(set(_TICKER_TO_CBOE.values()))


def map_ticker_to_cboe_index(ticker: str) -> str:
    """
    Return the CBOE index symbol for a given equity ticker.

    Raises KeyError if the ticker has no mapping.
    """
    if ticker not in _TICKER_TO_CBOE:
        raise KeyError(
            f"No CBOE mapping for ticker '{ticker}'. "
            f"Add it to _TICKER_TO_CBOE in cboe_data.py."
        )
    return _TICKER_TO_CBOE[ticker]


def download_cboe_index(
    symbol: str,
    start: str,
    end: str,
    cache_dir: Path,
) -> pd.Series:
    """
    Download (or load from cache) a CBOE index as a daily closing price series.

    The index values represent annualized implied volatility in percentage
    points (e.g., VIX=20 means ~20% annualized IV). We store the raw values;
    callers divide by 100 to convert to decimal.

    Parameters
    ----------
    symbol : str, e.g. '^VIX'
    start  : str, e.g. '2015-01-01'
    end    : str, e.g. '2024-12-31'
    cache_dir : Path to data/raw/cboe/

    Returns
    -------
    pd.Series indexed by date, values = closing level (percentage points)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    safe_name = symbol.replace("^", "")
    cache_path = cache_dir / f"{safe_name}.parquet"

    # Try cache first
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        df.index = pd.to_datetime(df.index)
        end_dt = pd.Timestamp(end)
        if df.index.max() >= end_dt - pd.Timedelta(days=5):
            mask = (df.index >= start) & (df.index <= end)
            sliced = df["close"].loc[mask]
            if not sliced.empty:
                logger.info("Loaded %s from cache (%d rows)", symbol, len(sliced))
                return sliced

    logger.info("Downloading CBOE index %s …", symbol)
    raw = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"No data returned for CBOE index {symbol}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0].lower() for col in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    raw.index = pd.to_datetime(raw.index)
    raw.index.name = "date"

    # Cache the full download
    raw[["close"]].to_parquet(cache_path)

    mask = (raw.index >= start) & (raw.index <= end)
    return raw["close"].loc[mask]


def build_cboe_iv_panel(
    universe: list,
    start: str,
    end: str,
    cache_dir: Path,
) -> pd.DataFrame:
    """
    Build a DataFrame mapping each (date, ticker) pair to its CBOE IV value.

    iv_cboe is in decimal form: VIX=20 → iv_cboe=0.20.

    Parameters
    ----------
    universe : list of equity tickers
    start, end : date range strings
    cache_dir : Path to data/raw/cboe/

    Returns
    -------
    DataFrame with MultiIndex (date, ticker) and column 'iv_cboe'
    """
    # Download each unique CBOE index once
    cboe_series = {}
    for symbol in CBOE_SYMBOLS:
        cboe_series[symbol] = download_cboe_index(symbol, start, end, cache_dir)
        logger.info(
            "%s: %d days, range [%.1f, %.1f]",
            symbol,
            len(cboe_series[symbol]),
            cboe_series[symbol].min(),
            cboe_series[symbol].max(),
        )

    # Build panel: for each ticker, attach the corresponding CBOE series
    records = []
    for ticker in universe:
        cboe_sym = map_ticker_to_cboe_index(ticker)
        series = cboe_series[cboe_sym]
        df = series.to_frame(name="iv_cboe_raw")
        df["ticker"] = ticker
        df["iv_cboe"] = df["iv_cboe_raw"] / 100.0  # convert % → decimal
        df = df.drop(columns=["iv_cboe_raw"])
        df.index.name = "date"
        records.append(df.reset_index().set_index(["date", "ticker"]))

    panel = pd.concat(records).sort_index()
    logger.info(
        "CBOE IV panel: %d rows, %d tickers, date range %s → %s",
        len(panel),
        panel.index.get_level_values("ticker").nunique(),
        panel.index.get_level_values("date").min().date(),
        panel.index.get_level_values("date").max().date(),
    )
    return panel
