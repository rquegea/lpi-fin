"""
Script 19: Backtest económico HONESTO con costes realistas.

Diferencias vs script 05 (backtest original):
  - position_size_pct: 1% → 5%
  - bid_ask_cost_pct:  8% → 15%
  - iv_markup: 1.0 → 1.5 (VIX/VXN subestima IV real de acciones individuales)

El iv_markup se aplica multiplicando iv_df * iv_markup antes de pasar
al StraddleBacktest. No requiere modificar la clase StraddleBacktest.

Usage:
    python3.11 scripts/19_backtest_honest.py

Reads:
    data/processed/panel_cboe.parquet
    data/processed/scores_oos_cboe.npy
    data/raw/cboe/VIX.parquet, VXN.parquet
    data/raw/{ticker}.parquet
    config_cboe.yaml
    config_backtest_honest.yaml

Outputs:
    results/tables/backtest_metrics_honest.csv
    results/tables/backtest_trades_honest.csv
    results/figures/equity_curve_honest.png
    results/figures/drawdown_honest.png
    results/figures/win_loss_distribution_honest.png

Stop conditions:
    Sharpe < -0.5 or > 2.0 → exit(2)
    Win rate outside [35%, 65%] → exit(2)
    Max drawdown > 40% → exit(2)
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest import StraddleBacktest
from src.cboe_data import _TICKER_TO_CBOE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Data loaders (same as script 05)
# ---------------------------------------------------------------------------

def _load_prices(root: Path, tickers: list) -> pd.DataFrame:
    frames = {}
    for ticker in tickers:
        path = root / "data" / "raw" / f"{ticker}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        frames[ticker] = df["close"].rename(ticker)
    if not frames:
        raise FileNotFoundError("No ticker price files found in data/raw/.")
    return pd.concat(frames.values(), axis=1).sort_index()


def _load_cboe_iv(root: Path, tickers: list) -> pd.DataFrame:
    vix = pd.read_parquet(root / "data" / "raw" / "cboe" / "VIX.parquet")["close"] / 100.0
    vxn = pd.read_parquet(root / "data" / "raw" / "cboe" / "VXN.parquet")["close"] / 100.0
    vix.index = pd.to_datetime(vix.index)
    vxn.index = pd.to_datetime(vxn.index)
    index_map = {"^VIX": vix, "^VXN": vxn}
    frames = {}
    for ticker in tickers:
        cboe_sym = _TICKER_TO_CBOE.get(ticker)
        if cboe_sym:
            frames[ticker] = index_map[cboe_sym].rename(ticker)
    return pd.concat(frames.values(), axis=1).sort_index()


def _reconstruct_oos_index(panel: pd.DataFrame, cboe_cfg: dict) -> pd.Index:
    n = len(panel)
    n_folds = cboe_cfg["n_folds"]
    embargo = cboe_cfg["embargo_days"]
    fold_size = n // (n_folds + 1)
    test_indices = []
    for i in range(1, n_folds + 1):
        start = fold_size * i + embargo
        end = min(fold_size * (i + 1), n)
        if start < end:
            test_indices.extend(range(start, end))
    return panel.index[test_indices]


def _build_scores_df(panel: pd.DataFrame, scores_oos: np.ndarray, cboe_cfg: dict) -> pd.DataFrame:
    oos_idx = _reconstruct_oos_index(panel, cboe_cfg)
    if len(oos_idx) != len(scores_oos):
        raise ValueError(
            f"OOS index length mismatch: {len(oos_idx)} vs {len(scores_oos)}"
        )
    return pd.DataFrame({"lpi_score": scores_oos}, index=oos_idx)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save_figures(equity: pd.Series, trades: pd.DataFrame, output_dir: Path) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity.index, equity.values, lw=1.5, color="darkorange")
    ax.axhline(equity.iloc[0], color="grey", lw=0.8, linestyle="--", label="Capital inicial")
    ax.set_title("Equity Curve — Backtest Honesto (5% pos, 15% spread, 1.5× IV)")
    ax.set_ylabel("Capital (USD)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "equity_curve_honest.png", dpi=120)
    plt.close()

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max * 100
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(drawdown.index, drawdown.values, 0, color="firebrick", alpha=0.5)
    ax.set_title("Drawdown (%) — Backtest Honesto")
    ax.set_ylabel("Drawdown (%)")
    plt.tight_layout()
    plt.savefig(fig_dir / "drawdown_honest.png", dpi=120)
    plt.close()

    if len(trades) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        wins = trades.loc[trades["won"], "pnl"]
        losses = trades.loc[~trades["won"], "pnl"]
        bins = np.linspace(trades["pnl"].min(), trades["pnl"].max(), 50)
        ax.hist(losses, bins=bins, color="firebrick", alpha=0.7, label=f"Loss ({len(losses)})")
        ax.hist(wins, bins=bins, color="steelblue", alpha=0.7, label=f"Win ({len(wins)})")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title("Trade P&L Distribution — Backtest Honesto")
        ax.set_xlabel("P&L (USD)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "win_loss_distribution_honest.png", dpi=120)
        plt.close()


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def _sharpe_verdict(sharpe: float) -> tuple:
    if sharpe < 0:
        return "NO VIABLE", "La señal no cubre costes realistas. Edge insuficiente para operar."
    elif sharpe < 0.5:
        return "MARGINAL", "Cubre costes apenas. No justifica capital. Publicable como resultado académico."
    elif sharpe < 1.0:
        return "POSITIVO DÉBIL", "Edge real pero por debajo del umbral institucional (SR≥1). Próximo paso: IV individual (OptionMetrics/Polygon)."
    elif sharpe < 1.5:
        return "BUENO", "Nivel profesional. Justifica conversación con fondos y backtest con datos reales de opciones."
    else:
        return "EXCELENTE — REVISAR", "Sharpe > 1.5 con costes conservadores. Verificar no hay overfitting residual."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    with open(ROOT / "config_cboe.yaml") as f:
        cboe_cfg = yaml.safe_load(f)
    with open(ROOT / "config_backtest_honest.yaml") as f:
        bt_raw = yaml.safe_load(f)
    bt_cfg = bt_raw["backtest"]

    iv_markup = float(bt_cfg.get("iv_markup", 1.0))
    logger.info(
        "Backtest HONESTO: position_size=%.0f%%, bid_ask=%.0f%%, iv_markup=%.1f×",
        bt_cfg["position_size_pct"] * 100,
        bt_cfg["bid_ask_cost_pct"] * 100,
        iv_markup,
    )

    panel_path = ROOT / "data" / "processed" / "panel_cboe.parquet"
    if not panel_path.exists():
        print("Error: panel_cboe.parquet not found.")
        sys.exit(1)

    scores_path = ROOT / "data" / "processed" / "scores_oos_cboe.npy"
    if not scores_path.exists():
        print("Error: scores_oos_cboe.npy not found. Run 11_run_lpi_cboe.py first.")
        sys.exit(1)

    panel = pd.read_parquet(panel_path).sort_index(level="date")
    scores_oos = np.load(scores_path)

    try:
        scores_df = _build_scores_df(panel, scores_oos, cboe_cfg)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    tickers = list(panel.index.get_level_values("ticker").unique())
    prices = _load_prices(ROOT, tickers)
    iv_df  = _load_cboe_iv(ROOT, tickers)

    # Apply iv_markup: multiply all IV values by the markup factor
    if iv_markup != 1.0:
        logger.info("Applying iv_markup=%.1f× to all IV values", iv_markup)
        iv_df = iv_df * iv_markup

    bt = StraddleBacktest(
        scores_df=scores_df,
        prices=prices,
        iv_per_ticker=iv_df,
        config=bt_cfg,
    )
    bt.run()

    metrics = bt.get_metrics()
    if "error" in metrics:
        print(f"Backtest error: {metrics['error']}")
        sys.exit(1)

    # Add config metadata to metrics
    metrics["position_size_pct"] = bt_cfg["position_size_pct"]
    metrics["bid_ask_cost_pct"]  = bt_cfg["bid_ask_cost_pct"]
    metrics["iv_markup"]         = iv_markup

    # Stop conditions
    stop = []
    if metrics["sharpe"] < -0.5:
        stop.append(f"Sharpe = {metrics['sharpe']:.3f} < -0.5 — sospechoso, revisar código")
    if metrics["sharpe"] > 2.0:
        stop.append(f"Sharpe = {metrics['sharpe']:.3f} > 2.0 — demasiado alto incluso con costes realistas")
    if not (0.35 <= metrics["win_rate"] <= 0.65):
        stop.append(f"Win rate = {metrics['win_rate']*100:.1f}% fuera de [35%, 65%] — posible bug")
    if metrics["max_drawdown"] < -0.40:
        stop.append(f"Max drawdown = {metrics['max_drawdown']*100:.1f}% > 40% — revisar si es real")

    equity = bt.get_equity_curve()
    trades = bt.get_trades()

    tab_dir = ROOT / "results" / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(tab_dir / "backtest_metrics_honest.csv", index=False)
    if len(trades) > 0:
        trades.to_csv(tab_dir / "backtest_trades_honest.csv", index=False)

    _save_figures(equity, trades, ROOT / "results")

    verdict_label, verdict_rec = _sharpe_verdict(metrics["sharpe"])
    avg_edge = metrics.get("avg_win_usd", 0) * metrics["win_rate"] + metrics.get("avg_loss_usd", 0) * (1 - metrics["win_rate"])

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print("====== BACKTEST HONESTO — LPI STRADDLE (CBOE IV) ======")
    print(f"{'='*60}")
    print(f"Generado: {timestamp}")
    print(f"\nParámetros:")
    print(f"  Position size:    {bt_cfg['position_size_pct']*100:.0f}%")
    print(f"  Bid-ask cost:     {bt_cfg['bid_ask_cost_pct']*100:.0f}%")
    print(f"  IV markup:        {iv_markup:.1f}×")
    print(f"\n--- Capital ---")
    print(f"  Capital inicial:  ${metrics['initial_capital']:>12,.2f}")
    print(f"  Capital final:    ${metrics['final_capital']:>12,.2f}")
    print(f"  Retorno total:    {metrics['total_return_pct']:>8.2f}%")
    print(f"  CAGR anualizado:  {metrics['cagr']*100:>8.2f}%")
    print(f"\n--- Riesgo / Retorno ---")
    print(f"  Sharpe ratio:     {metrics['sharpe']:>8.3f}")
    print(f"  Sortino ratio:    {metrics['sortino']:>8.3f}")
    print(f"  Max drawdown:     {metrics['max_drawdown']*100:>8.2f}%")
    print(f"\n--- Operaciones ---")
    print(f"  N operaciones:    {metrics['n_trades']:>8,}")
    print(f"  Win rate:         {metrics['win_rate']*100:>8.1f}%")
    print(f"  Avg win (USD):    ${metrics['avg_win_usd']:>10,.2f}")
    print(f"  Avg loss (USD):   ${metrics['avg_loss_usd']:>10,.2f}")
    print(f"  Edge por trade:   ${avg_edge:>10,.2f}")
    print(f"  Días simulados:   {metrics['n_trading_days']:>8,}")
    print(f"\n====== VEREDICTO ======")
    print(f"[{verdict_label}]")
    print(f"{verdict_rec}")

    if stop:
        print(f"\n====== ⚠  CONDICIONES DE PARADA ======")
        for s in stop:
            print(f"  ⚠  {s}")
        print(f"\nResultados en: {ROOT / 'results'}")
        sys.exit(2)

    log_path = ROOT / "results" / "run_log.txt"
    with open(log_path, "a") as f:
        f.write(
            f"\n{'='*60}\nBACKTEST HONESTO — {timestamp}\n"
            f"position_size={bt_cfg['position_size_pct']*100:.0f}% "
            f"bid_ask={bt_cfg['bid_ask_cost_pct']*100:.0f}% "
            f"iv_markup={iv_markup:.1f}×\n"
            f"Sharpe={metrics['sharpe']:.3f} CAGR={metrics['cagr']*100:.2f}% "
            f"MDD={metrics['max_drawdown']*100:.2f}% Trades={metrics['n_trades']}\n"
            f"Veredicto: [{verdict_label}]\n{'='*60}\n"
        )

    print(f"\nResultados en: {ROOT / 'results'}")


if __name__ == "__main__":
    main()
