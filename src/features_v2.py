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


# ---------------------------------------------------------------------------
# V3 features: 3 additional OHLCV-based features
# EXPERIMENTO DESCARTADO (2026-04-09)
# Resultado: v3 single AUC 0.6015, v3 ensemble AUC 0.5785
# Peor que baseline CBOE 4-features (AUC 0.6253).
# Razón probable: maldición de dimensionalidad en GMM al pasar de
# 4 a 7 features. Se mantiene código como registro del intento.
# Modelo final del proyecto: CBOE 4-features. Ver scripts 09-13.
# ---------------------------------------------------------------------------

FEATURE_NAMES_V3 = [
    "iv_rv_spread_cboe",
    "log_range",
    "vol_of_vol",
    "log_dvol",
    "skew_60d",
    "kurt_60d",
    "vol_autocorr_5d",
]


def compute_skew_60d(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """
    Rolling 60-day skewness of log returns.

    Esta feature usa log_returns en [t-59, t], no look-ahead.
    pandas rolling().skew() usa la fórmula de Fisher-Pearson ajustada (bias-corrected).
    NaN para las primeras (window-1) filas.
    """
    return _log_returns(df["close"]).rolling(window).skew().rename("skew_60d")


def compute_kurt_60d(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """
    Rolling 60-day excess kurtosis of log returns.

    Esta feature usa log_returns en [t-59, t], no look-ahead.
    pandas rolling().kurt() devuelve kurtosis en exceso (normal = 0).
    NaN para las primeras (window-1) filas.
    """
    return _log_returns(df["close"]).rolling(window).kurt().rename("kurt_60d")


def compute_vol_autocorr_5d(
    df: pd.DataFrame,
    vol_window: int = 5,
    autocorr_window: int = 20,
) -> pd.Series:
    """
    Autocorrelación de lag-1 de la serie de volatilidad rolling a 5 días.

    Esta feature usa vol_5d en [t-23, t] (porque vol_5d[t] requiere returns en [t-4, t]).

    Paso 1: vol_5d(t) = std de log_returns en [t-4, t], anualizada.
    Paso 2: autocorr(t) = corr de Pearson de vol_5d[t-19:t] y vol_5d[t-20:t-1]
            (equivalente a rolling(20).corr(lag=1)).

    NaN para las primeras (vol_window + autocorr_window - 2) filas.
    """
    lr = _log_returns(df["close"])
    vol_5d = lr.rolling(vol_window).std() * _SQRT252
    return vol_5d.rolling(autocorr_window).corr(vol_5d.shift(1)).rename("vol_autocorr_5d")


def build_features_v3(
    df: pd.DataFrame,
    iv_cboe: pd.Series,
    cfg: dict,
) -> pd.DataFrame:
    """
    Construct all 7 features (CBOE 4 + OHLCV 3) for a single ticker.

    Parameters
    ----------
    df      : DataFrame with columns: open, high, low, close, volume
              Indexed by date, ascending.
    iv_cboe : Series of CBOE IV (decimal) aligned by date.
    cfg     : dict with feature window parameters. Required keys:
              rv20_window, vol_of_vol_inner, vol_of_vol_outer,
              dvol_short, dvol_long, skew_window, kurt_window,
              vol_autocorr_inner, vol_autocorr_outer.

    Returns
    -------
    DataFrame of shape (n, 7) with columns in FEATURE_NAMES_V3.
    """
    return pd.concat([
        compute_iv_rv_spread_cboe(df, iv_cboe, cfg["rv20_window"]),
        compute_log_range(df),
        compute_vol_of_vol(df, cfg["vol_of_vol_inner"], cfg["vol_of_vol_outer"]),
        compute_log_dvol(df, cfg["dvol_short"], cfg["dvol_long"]),
        compute_skew_60d(df, cfg["skew_window"]),
        compute_kurt_60d(df, cfg["kurt_window"]),
        compute_vol_autocorr_5d(df, cfg["vol_autocorr_inner"], cfg["vol_autocorr_outer"]),
    ], axis=1)
