"""
Tests for src/backtest.py — StraddleBacktest.

Test 1 — Signal quality discrimination
    Perfect signal (selected tickers always move a lot) → Sharpe > 2.
    Random signal (selected tickers move same as others) → Sharpe < 1.

Test 2 — Equity consistency
    final_capital == initial_capital + sum(all trade pnl).

Test 3 — No look-ahead bias
    For each closed trade:
      (a) close_date > open_date
      (b) return_5d == log(prices[close_date, ticker] / prices[open_date, ticker])
      (c) close_date is exactly hold_days trading days after open_date
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.backtest import StraddleBacktest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BT_CFG = {
    "initial_capital": 100_000.0,
    "position_size_pct": 0.01,
    "max_concurrent_positions": 10,
    "hold_days": 5,
    "top_quintile_pct": 0.20,
    "bid_ask_cost_pct": 0.08,
    "risk_free_rate": 0.0,
}

# cost_iv = 0.20 * sqrt(5/252) ≈ 0.02817
# total_cost = 0.02817 * 1.08 ≈ 0.03042
_TOTAL_COST_APPROX = 0.20 * np.sqrt(5 / 252) * 1.08


def _make_backtest_data(
    n_days: int = 300,
    n_tickers: int = 20,
    perfect_signal: bool = False,
    seed: int = 42,
) -> tuple:
    """
    Build minimal synthetic backtest inputs.

    perfect_signal=True:
        The first floor(n_tickers * top_quintile_pct) tickers (the "top group")
        always make a ±10 log-return move over every hold_days window.
        They are always assigned score=1.0; the rest get score=0.0.
        Every trade on the top group wins by a large margin.

    perfect_signal=False:
        All tickers execute a tiny random walk (σ=0.3% per day).
        Expected |return_5d| ≈ 0.67% << total_cost ≈ 3.0%.
        Scores are random, so we are essentially paying premium for nothing.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]

    n_top = max(1, int(n_tickers * _BT_CFG["top_quintile_pct"]))
    hold = _BT_CFG["hold_days"]

    # ----- prices -----------------------------------------------------------
    if perfect_signal:
        # Top n_top tickers: oscillate +10% / -10% in log-space every hold days
        prices_arr = np.ones((n_days, n_tickers)) * 100.0
        for ti in range(n_top):
            for t in range(1, n_days):
                # direction flips every hold_days
                direction = 1 if (t // hold) % 2 == 0 else -1
                # constant drift so each consecutive close is ±(10%/hold) per day
                log_ret_per_day = direction * 0.10 / hold
                prices_arr[t, ti] = 100.0 * np.exp(log_ret_per_day * t)
    else:
        log_rets = rng.normal(0.0, 0.003, (n_days, n_tickers))
        prices_arr = 100.0 * np.exp(np.cumsum(log_rets, axis=0))

    prices = pd.DataFrame(prices_arr, index=dates, columns=tickers)

    # ----- iv (flat 20% annualised for all) ---------------------------------
    iv = pd.DataFrame(
        np.full((n_days, n_tickers), 0.20), index=dates, columns=tickers
    )

    # ----- scores -----------------------------------------------------------
    if perfect_signal:
        score_vals = np.zeros((n_days, n_tickers))
        score_vals[:, :n_top] = 1.0
    else:
        score_vals = rng.uniform(0.0, 1.0, (n_days, n_tickers))

    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    scores_df = pd.DataFrame(
        {"lpi_score": score_vals.ravel()}, index=idx
    )

    return scores_df, prices, iv


# ---------------------------------------------------------------------------
# Test 1 — Signal quality
# ---------------------------------------------------------------------------

class TestSignalQuality:
    def test_perfect_signal_sharpe_above_2(self):
        """
        With a perfect signal (top tickers always make a big ±10% move),
        every straddle pays off handsomely.  Sharpe must exceed 2.0.
        """
        scores_df, prices, iv = _make_backtest_data(
            n_days=400, perfect_signal=True
        )
        bt = StraddleBacktest(scores_df, prices, iv, _BT_CFG)
        bt.run()
        m = bt.get_metrics()
        assert "error" not in m, f"Backtest error: {m}"
        assert m["sharpe"] > 2.0, (
            f"Perfect signal should yield Sharpe > 2.0, got {m['sharpe']:.3f}"
        )

    def test_random_signal_sharpe_below_1(self):
        """
        With a random signal on tiny-move data (σ=0.3%/day),
        expected |return_5d| ≈ 0.67% is well below total_cost ≈ 3.0%.
        Strategies consistently lose; Sharpe must be below 1.0.
        """
        scores_df, prices, iv = _make_backtest_data(
            n_days=400, perfect_signal=False, seed=99
        )
        bt = StraddleBacktest(scores_df, prices, iv, _BT_CFG)
        bt.run()
        m = bt.get_metrics()
        assert "error" not in m, f"Backtest error: {m}"
        assert m["sharpe"] < 1.0, (
            f"Random signal on low-vol data should yield Sharpe < 1.0, "
            f"got {m['sharpe']:.3f}"
        )


# ---------------------------------------------------------------------------
# Test 2 — Equity consistency
# ---------------------------------------------------------------------------

class TestEquityConsistency:
    def test_final_capital_equals_initial_plus_sum_pnl(self):
        """
        capital_final must equal capital_initial + sum(trade.pnl) within
        floating-point precision.  Capital only changes at position close,
        so this is an exact arithmetic identity.
        """
        scores_df, prices, iv = _make_backtest_data(
            n_days=300, perfect_signal=True
        )
        bt = StraddleBacktest(scores_df, prices, iv, _BT_CFG)
        bt.run()

        equity = bt.get_equity_curve()
        trades = bt.get_trades()

        computed_final = _BT_CFG["initial_capital"] + trades["pnl"].sum()
        # Use relative tolerance: floating-point accumulation over many additions
        # can shift the last bit; the identity must hold to ~10 significant digits.
        rel_err = abs(equity.iloc[-1] - computed_final) / max(abs(computed_final), 1.0)
        assert rel_err < 1e-10, (
            f"Equity mismatch (relative error {rel_err:.2e}): "
            f"curve ends at {equity.iloc[-1]:.6f}, "
            f"initial + sum(pnl) = {computed_final:.6f}"
        )

    def test_equity_curve_is_monotone_with_perfect_signal(self):
        """
        With a perfect signal every trade wins, so the equity curve
        must be non-decreasing (equity never falls).
        """
        scores_df, prices, iv = _make_backtest_data(
            n_days=400, perfect_signal=True
        )
        bt = StraddleBacktest(scores_df, prices, iv, _BT_CFG)
        bt.run()
        equity = bt.get_equity_curve()
        diffs = equity.diff().dropna()
        assert (diffs >= -1e-9).all(), (
            "Equity curve should be non-decreasing with a perfect signal."
        )

    def test_no_trades_when_no_scores(self):
        """
        If scores_df is empty, no trades should execute and capital stays flat.
        """
        scores_df, prices, iv = _make_backtest_data(n_days=50)
        empty_scores = scores_df.iloc[:0]  # empty DataFrame, correct columns
        bt = StraddleBacktest(empty_scores, prices, iv, _BT_CFG)
        bt.run()
        trades = bt.get_trades()
        assert len(trades) == 0


# ---------------------------------------------------------------------------
# Test 3 — No look-ahead bias
# ---------------------------------------------------------------------------

class TestNoLookAheadBias:
    @pytest.fixture(scope="class")
    def backtest_result(self):
        scores_df, prices, iv = _make_backtest_data(
            n_days=200, perfect_signal=True
        )
        bt = StraddleBacktest(scores_df, prices, iv, _BT_CFG)
        bt.run()
        return bt, prices

    def test_close_date_strictly_after_open_date(self, backtest_result):
        """Every trade must close strictly after it opens."""
        bt, _ = backtest_result
        trades = bt.get_trades()
        assert len(trades) > 0, "No trades were executed — test is vacuous."
        assert (trades["close_date"] > trades["open_date"]).all(), (
            "Found trades where close_date <= open_date (look-ahead bias)."
        )

    def test_return_uses_close_date_price(self, backtest_result):
        """
        return_5d stored in each trade must equal
        log(prices[close_date, ticker] / prices[open_date, ticker]).
        Any deviation means the backtest used the wrong price.
        """
        bt, prices = backtest_result
        trades = bt.get_trades()
        assert len(trades) > 0, "No trades were executed — test is vacuous."

        for _, row in trades.iterrows():
            expected = float(
                np.log(
                    prices.loc[row["close_date"], row["ticker"]]
                    / prices.loc[row["open_date"], row["ticker"]]
                )
            )
            assert abs(row["return_5d"] - expected) < 1e-9, (
                f"return_5d mismatch for {row['ticker']} "
                f"({row['open_date'].date()} → {row['close_date'].date()}): "
                f"stored={row['return_5d']:.8f}, expected={expected:.8f}"
            )

    def test_hold_period_is_exactly_hold_days(self, backtest_result):
        """
        close_date must be exactly hold_days trading days after open_date,
        as counted by the prices DataFrame index.
        """
        bt, prices = backtest_result
        trades = bt.get_trades()
        hold_days = _BT_CFG["hold_days"]
        all_dates = list(prices.index)
        date_to_idx = {d: i for i, d in enumerate(all_dates)}

        for _, row in trades.iterrows():
            open_idx = date_to_idx[row["open_date"]]
            close_idx = date_to_idx[row["close_date"]]
            assert close_idx - open_idx == hold_days, (
                f"Hold period mismatch for {row['ticker']}: "
                f"open_idx={open_idx}, close_idx={close_idx}, "
                f"expected gap={hold_days}"
            )
