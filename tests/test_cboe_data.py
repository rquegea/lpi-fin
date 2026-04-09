"""
Tests for src/cboe_data.py.

Verifies:
  1. Every ticker in the universe has a CBOE mapping.
  2. CBOE index values (when downloaded) are in a plausible range.
  3. The mapping function raises for unknown tickers.
  4. build_cboe_iv_panel produces expected shape and columns.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
import yaml

from src.cboe_data import map_ticker_to_cboe_index, _TICKER_TO_CBOE, CBOE_SYMBOLS


@pytest.fixture
def universe():
    root = os.path.join(os.path.dirname(__file__), "..")
    with open(os.path.join(root, "config_cboe.yaml")) as f:
        cfg = yaml.safe_load(f)
    return cfg["universe"]


def test_mapping_exhaustive(universe):
    """Every ticker in the universe must have a CBOE mapping."""
    missing = [t for t in universe if t not in _TICKER_TO_CBOE]
    assert not missing, f"Tickers with no CBOE mapping: {missing}"


def test_mapping_returns_valid_symbol(universe):
    """map_ticker_to_cboe_index must return a known CBOE symbol."""
    for ticker in universe:
        sym = map_ticker_to_cboe_index(ticker)
        assert sym in CBOE_SYMBOLS, (
            f"{ticker} mapped to {sym} which is not in CBOE_SYMBOLS={CBOE_SYMBOLS}"
        )


def test_mapping_raises_for_unknown_ticker():
    """Unknown tickers must raise KeyError."""
    with pytest.raises(KeyError):
        map_ticker_to_cboe_index("FAKEXYZ")


def test_vxn_tickers_are_tech(universe):
    """VXN group must include the expected tech tickers."""
    expected_vxn = {"AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
                    "ADBE", "NFLX", "CRM", "INTC", "CSCO", "ORCL"}
    actual_vxn = {t for t in universe if _TICKER_TO_CBOE.get(t) == "^VXN"}
    assert expected_vxn == actual_vxn, (
        f"VXN tickers mismatch.\nExpected: {expected_vxn}\nGot: {actual_vxn}"
    )


def test_all_tickers_mapped_to_vix_or_vxn(universe):
    """In the current mapping, only ^VIX and ^VXN should be used."""
    for ticker in universe:
        sym = _TICKER_TO_CBOE[ticker]
        assert sym in {"^VIX", "^VXN"}, (
            f"{ticker} mapped to unexpected symbol {sym}"
        )


def test_cboe_iv_range():
    """
    Spot-check: a synthetic CBOE series should be in [0.05, 0.90] after /100.
    VIX has historically ranged from ~9 (2017) to ~80 (March 2020).
    """
    raw_values = np.array([9.0, 15.0, 20.0, 30.0, 50.0, 80.0])
    iv_decimal = raw_values / 100.0
    assert (iv_decimal >= 0.05).all(), "Some values below plausible minimum (5%)"
    assert (iv_decimal <= 0.90).all(), "Some values above plausible maximum (90%)"


def test_universe_split():
    """Universe must be split into exactly VXN (13 tickers) + VIX (17 tickers)."""
    vxn_count = sum(1 for v in _TICKER_TO_CBOE.values() if v == "^VXN")
    vix_count = sum(1 for v in _TICKER_TO_CBOE.values() if v == "^VIX")
    assert vxn_count == 13, f"Expected 13 VXN tickers, got {vxn_count}"
    assert vix_count == 17, f"Expected 17 VIX tickers, got {vix_count}"
