"""
Script 21: Cost sensitivity sweep — stress test del backtest.

Corre el backtest con todas las combinaciones de:
  bid_ask_cost_pct ∈ {0.05, 0.10, 0.15, 0.20}
  iv_markup        ∈ {1.0, 1.2, 1.5, 2.0}

Muestra tabla de Sharpe, CAGR y Win rate para cada combinación,
marcando el punto de break-even (Sharpe cruzando cero).

Usage:
    python3.11 scripts/21_cost_sensitivity.py

Reads:
    data/processed/panel_cboe.parquet
    data/processed/scores_oos_cboe.npy
    data/raw/cboe/VIX.parquet, VXN.parquet
    data/raw/{ticker}.parquet
    config_cboe.yaml
    config_backtest.yaml   (para initial_capital y otros params fijos)

Outputs:
    results/tables/cost_sensitivity.csv
"""

import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest import StraddleBacktest
from src.cboe_data import _TICKER_TO_CBOE

ROOT = Path(__file__).parent.parent

BID_ASK_VALUES = [0.05, 0.10, 0.15, 0.20]
IV_MARKUP_VALUES = [1.0, 1.2, 1.5, 2.0]
POSITION_SIZE = 0.05   # fijo en 5% para todos los runs

# ---------------------------------------------------------------------------
# Data loaders (idénticos a scripts 05/19)
# ---------------------------------------------------------------------------

def _load_prices(root):
    frames = {}
    for path in (root / "data" / "raw").glob("*.parquet"):
        ticker = path.stem
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        frames[ticker] = df["close"].rename(ticker)
    return pd.concat(frames.values(), axis=1).sort_index()


def _load_cboe_iv(root, tickers):
    vix = pd.read_parquet(root / "data" / "raw" / "cboe" / "VIX.parquet")["close"] / 100.0
    vxn = pd.read_parquet(root / "data" / "raw" / "cboe" / "VXN.parquet")["close"] / 100.0
    vix.index = pd.to_datetime(vix.index)
    vxn.index = pd.to_datetime(vxn.index)
    index_map = {"^VIX": vix, "^VXN": vxn}
    frames = {}
    for ticker in tickers:
        sym = _TICKER_TO_CBOE.get(ticker)
        if sym:
            frames[ticker] = index_map[sym].rename(ticker)
    return pd.concat(frames.values(), axis=1).sort_index()


def _reconstruct_oos_index(panel, cboe_cfg):
    n = len(panel)
    fold_size = n // (cboe_cfg["n_folds"] + 1)
    test_indices = []
    for i in range(1, cboe_cfg["n_folds"] + 1):
        start = fold_size * i + cboe_cfg["embargo_days"]
        end = min(fold_size * (i + 1), n)
        if start < end:
            test_indices.extend(range(start, end))
    return panel.index[test_indices]


def _build_scores_df(panel, scores_oos, cboe_cfg):
    oos_idx = _reconstruct_oos_index(panel, cboe_cfg)
    return pd.DataFrame({"lpi_score": scores_oos}, index=oos_idx)


def _run_one(scores_df, prices, iv_base, bt_cfg_base, bid_ask, iv_markup):
    """Run a single backtest configuration and return metrics dict."""
    iv = iv_base * iv_markup if iv_markup != 1.0 else iv_base.copy()

    cfg = dict(bt_cfg_base)
    cfg["bid_ask_cost_pct"]  = bid_ask
    cfg["position_size_pct"] = POSITION_SIZE

    bt = StraddleBacktest(
        scores_df=scores_df,
        prices=prices,
        iv_per_ticker=iv,
        config=cfg,
    )
    bt.run()
    m = bt.get_metrics()
    if "error" in m:
        return {"sharpe": float("nan"), "cagr": float("nan"),
                "win_rate": float("nan"), "n_trades": 0,
                "max_drawdown": float("nan")}
    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(ROOT / "config_cboe.yaml") as f:
        cboe_cfg = yaml.safe_load(f)
    with open(ROOT / "config_backtest.yaml") as f:
        bt_cfg_base = yaml.safe_load(f)["backtest"]

    panel = pd.read_parquet(
        ROOT / "data" / "processed" / "panel_cboe.parquet"
    ).sort_index(level="date")
    scores_oos = np.load(ROOT / "data" / "processed" / "scores_oos_cboe.npy")
    scores_df  = _build_scores_df(panel, scores_oos, cboe_cfg)

    tickers = list(panel.index.get_level_values("ticker").unique())
    prices   = _load_prices(ROOT)
    # Keep only tickers in universe
    prices   = prices[[c for c in prices.columns if c in tickers]]
    iv_base  = _load_cboe_iv(ROOT, tickers)

    # -----------------------------------------------------------------------
    # Run all combinations
    # -----------------------------------------------------------------------
    n_total = len(BID_ASK_VALUES) * len(IV_MARKUP_VALUES)
    print(f"\nCorriendo {n_total} combinaciones de costes …\n")

    rows = []
    for i, (ba, mu) in enumerate(product(BID_ASK_VALUES, IV_MARKUP_VALUES), 1):
        m = _run_one(scores_df, prices, iv_base, bt_cfg_base, ba, mu)
        rows.append({
            "bid_ask_pct": ba,
            "iv_markup":   mu,
            "sharpe":      m["sharpe"],
            "cagr_pct":    m["cagr"] * 100,
            "win_rate_pct": m["win_rate"] * 100,
            "max_dd_pct":  m["max_drawdown"] * 100,
            "n_trades":    m["n_trades"],
        })
        print(
            f"  [{i:2d}/{n_total}] bid_ask={ba*100:.0f}%  iv_markup={mu:.1f}×  "
            f"→  Sharpe={m['sharpe']:+.3f}  CAGR={m['cagr']*100:+.2f}%  "
            f"WR={m['win_rate']*100:.1f}%"
        )

    df = pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Print pivot tables
    # -----------------------------------------------------------------------
    def _pivot(col, fmt):
        piv = df.pivot(index="iv_markup", columns="bid_ask_pct", values=col)
        piv.columns = [f"bid_ask={c*100:.0f}%" for c in piv.columns]
        piv.index   = [f"iv_markup={r:.1f}×"   for r in piv.index]
        return piv.map(lambda v: fmt.format(v) if not pd.isna(v) else "?")

    print(f"\n{'='*72}")
    print("TABLA SHARPE  (position_size=5% fijo)")
    print(f"{'='*72}")
    sharpe_piv = df.pivot(index="iv_markup", columns="bid_ask_pct", values="sharpe")
    sharpe_piv.columns = [f"bid_ask {c*100:.0f}%" for c in sharpe_piv.columns]
    sharpe_piv.index   = [f"iv_markup {r:.1f}×"   for r in sharpe_piv.index]
    # Format with + and mark negatives/positives
    def _fmt_sharpe(v):
        if pd.isna(v):
            return "     ?"
        s = f"{v:+.3f}"
        if v > 0:
            return f"  {s} ✓"
        elif v > -1:
            return f"  {s}  "
        else:
            return f"  {s} ✗"
    print(sharpe_piv.map(_fmt_sharpe).to_string())

    print(f"\n{'='*72}")
    print("TABLA CAGR (%)")
    print(f"{'='*72}")
    cagr_piv = df.pivot(index="iv_markup", columns="bid_ask_pct", values="cagr_pct")
    cagr_piv.columns = [f"bid_ask {c*100:.0f}%" for c in cagr_piv.columns]
    cagr_piv.index   = [f"iv_markup {r:.1f}×"   for r in cagr_piv.index]
    print(cagr_piv.map(lambda v: f"{v:+6.2f}%" if not pd.isna(v) else "?").to_string())

    print(f"\n{'='*72}")
    print("TABLA WIN RATE (%)")
    print(f"{'='*72}")
    wr_piv = df.pivot(index="iv_markup", columns="bid_ask_pct", values="win_rate_pct")
    wr_piv.columns = [f"bid_ask {c*100:.0f}%" for c in wr_piv.columns]
    wr_piv.index   = [f"iv_markup {r:.1f}×"   for r in wr_piv.index]
    print(wr_piv.map(lambda v: f"{v:5.1f}%" if not pd.isna(v) else "?").to_string())

    # -----------------------------------------------------------------------
    # Break-even analysis
    # -----------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("ANÁLISIS DE BREAK-EVEN")
    print(f"{'='*72}")
    positive = df[df["sharpe"] > 0].sort_values("sharpe", ascending=False)
    negative = df[df["sharpe"] < 0].sort_values("sharpe", ascending=False)

    if len(positive) > 0:
        best = positive.iloc[0]
        print(f"\nCombinaciones con Sharpe > 0: {len(positive)}/{n_total}")
        print(f"Mejor combinación: bid_ask={best['bid_ask_pct']*100:.0f}%  "
              f"iv_markup={best['iv_markup']:.1f}×  →  Sharpe={best['sharpe']:+.3f}")
        print(f"\nTodas las combinaciones viables (Sharpe > 0):")
        for _, row in positive.iterrows():
            print(f"  bid_ask={row['bid_ask_pct']*100:.0f}%  iv_markup={row['iv_markup']:.1f}×  "
                  f"→  Sharpe={row['sharpe']:+.3f}  CAGR={row['cagr_pct']:+.2f}%  "
                  f"WR={row['win_rate_pct']:.1f}%")
    else:
        print("\nNinguna combinación produce Sharpe > 0.")
        print("La señal estadística (AUC 0.6253) no cubre costes ni en el escenario más optimista.")
        print(f"\nCombinación menos negativa:")
        best_neg = negative.iloc[0]
        print(f"  bid_ask={best_neg['bid_ask_pct']*100:.0f}%  iv_markup={best_neg['iv_markup']:.1f}×  "
              f"→  Sharpe={best_neg['sharpe']:+.3f}  CAGR={best_neg['cagr_pct']:+.2f}%")

    # Original backtest point (1% pos size, but rescaled is not comparable;
    # show baseline with position_size=5% and original costs for reference)
    base_row = df[(df["bid_ask_pct"] == 0.05) & (df["iv_markup"] == 1.0)]
    if len(base_row) > 0:
        br = base_row.iloc[0]
        print(f"\nReferencia — costes mínimos (bid_ask=5%, markup=1.0×, pos=5%):")
        print(f"  Sharpe={br['sharpe']:+.3f}  CAGR={br['cagr_pct']:+.2f}%  WR={br['win_rate_pct']:.1f}%")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    out_path = ROOT / "results" / "tables" / "cost_sensitivity.csv"
    df.to_csv(out_path, index=False)
    print(f"\nGuardado en: {out_path}")


if __name__ == "__main__":
    main()
