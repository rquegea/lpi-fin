"""
Features v2: CBOE-based IV replaces the rv_30 × 1.04 proxy.

This module produces the same 4-feature set as the parsimonious run,
but with iv_level sourced from CBOE indices (VIX/VXN) rather than
a mechanical rolling-vol proxy.

Features:
  1. iv_rv_spread_cboe  = iv_level_cboe(t) - rv_20(t)
  2. log_range          = log((high - low) / close)  [unchanged]
  3. vol_of_vol         = rolling std of 5d-vol       [unchanged]
  4. log_dvol           = log(dvol_5d / dvol_60d)    [unchanged]

Dropped vs 7-feature baseline:
  iv_level (old proxy rv_30 × 1.04)  — replaced by iv_level_cboe
  rv_20                               — dropped (negative ablation delta)
  momentum                            — dropped (negative ablation delta)

The iv_rv_spread_cboe feature now uses real market consensus IV
(from options-implied data embedded in VIX/VXN) rather than a
backward-looking vol estimate. This is a fundamentally different
signal: the old spread measured "how much does recent vol differ from
slightly-smoothed recent vol"; the new spread measures "how much does
the market EXPECT future vol to differ from recent realized vol" —
which is the actual risk premium we are trying to predict.
"""

import numpy as np
import pandas as pd

_SQRT252 = np.sqrt(252)

FEATURE_NAMES_V2 = [
    "iv_rv_spread_cboe",
    "log_range",
    "vol_of_vol",
    "log_dvol",
]


def _log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def compute_iv_rv_spread_cboe(
    df: pd.DataFrame,
    iv_cboe: pd.Series,
    rv20_window: int,
) -> pd.Series:
    """
    IV/RV spread using CBOE index as the IV reference.

    Esta feature usa:
      - iv_cboe(t): CBOE index value on day t (external, market-consensus IV)
      - rv_20(t): std of log_returns in [t-19, t], annualized

    iv_cboe(t) is already in decimal form (e.g., 0.20 for VIX=20).
    rv_20(t) is also annualized (same units).

    A positive spread means the market prices future vol higher than
    recent realized vol — i.e., options look "expensive" relative to
    trailing vol. A negative spread means realized vol has been running
    above implied — unusual regime.

    Parameters
    ----------
    df      : DataFrame with column 'close'
    iv_cboe : Series of CBOE IV values aligned to df's index (decimal)
    rv20_window : int (default 20)

    Returns
    -------
    pd.Series named 'iv_rv_spread_cboe'
    """
    lr = _log_returns(df["close"])
    rv_20 = lr.rolling(rv20_window).std() * _SQRT252
    iv_cboe_aligned = iv_cboe.reindex(df.index)
    return (iv_cboe_aligned - rv_20).rename("iv_rv_spread_cboe")


def compute_log_range(df: pd.DataFrame) -> pd.Series:
    """
    Intraday log range: log((high - low) / close).

    Esta feature usa: high, low, close en t (solo el día t).
    Unchanged from features.py.
    """
    return np.log((df["high"] - df["low"]) / df["close"]).rename("log_range")


def compute_vol_of_vol(
    df: pd.DataFrame,
    inner_window: int,
    outer_window: int,
) -> pd.Series:
    """
    Volatility of volatility: rolling std of short-term vol series.

    Esta feature usa: log_returns en [t - (inner + outer - 2), t].
    Unchanged from features.py.
    """
    lr = _log_returns(df["close"])
    vol_5d = lr.rolling(inner_window).std() * _SQRT252
    return vol_5d.rolling(outer_window).std().rename("vol_of_vol")


def compute_log_dvol(
    df: pd.DataFrame,
    short_window: int,
    long_window: int,
) -> pd.Series:
    """
    Log ratio of short-term to long-term dollar volume.

    Esta feature usa: close * volume en [t - {long_window-1}, t].
    Unchanged from features.py.
    """
    dvol = df["close"] * df["volume"]
    return np.log(
        dvol.rolling(short_window).mean() / dvol.rolling(long_window).mean()
    ).rename("log_dvol")


def build_features_v2(
    df: pd.DataFrame,
    iv_cboe: pd.Series,
    cfg: dict,
) -> pd.DataFrame:
    """
    Construct all 4 CBOE-based features for a single ticker.

    Parameters
    ----------
    df      : DataFrame with columns: open, high, low, close, volume
              Indexed by date, ascending.
    iv_cboe : Series of CBOE IV (decimal) aligned by date.
    cfg     : dict with feature window parameters.

    Returns
    -------
    DataFrame of shape (n, 4) with columns in FEATURE_NAMES_V2.
    """
    iv_rv_spread_cboe = compute_iv_rv_spread_cboe(df, iv_cboe, cfg["rv20_window"])
    log_range         = compute_log_range(df)
    vol_of_vol        = compute_vol_of_vol(df, cfg["vol_of_vol_inner"], cfg["vol_of_vol_outer"])
    log_dvol          = compute_log_dvol(df, cfg["dvol_short"], cfg["dvol_long"])

    return pd.concat(
        [iv_rv_spread_cboe, log_range, vol_of_vol, log_dvol],
        axis=1,
    )
