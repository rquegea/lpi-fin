"""
Script 10: Build features_v2 + target_v2 panel using CBOE IV.

Usage:
    python3.11 scripts/10_build_panel_cboe.py

Reads:
  - data/raw/{ticker}.parquet  (OHLCV)
  - data/raw/cboe/{VIX,VXN}.parquet  (must run 09 first)

Outputs:
  - data/processed/panel_cboe.parquet  (features_v2 + target_v2)

Abort conditions:
  - Fewer than 50,000 usable observations after merge and NaN drop.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_universe
from src.cboe_data import build_cboe_iv_panel
from src.features_v2 import build_features_v2, FEATURE_NAMES_V2
from src.target_v2 import compute_target_v2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

MIN_OBS = 50_000


def main():
    root = Path(__file__).parent.parent
    with open(root / "config_cboe.yaml") as f:
        cfg = yaml.safe_load(f)

    raw_dir       = root / "data" / "raw"
    cboe_dir      = root / "data" / "raw" / "cboe"
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(exist_ok=True)

    # Load OHLCV (from cache)
    logger.info("Loading OHLCV data …")
    try:
        universe_data = load_universe(
            universe=cfg["universe"],
            start=cfg["fecha_inicio"],
            end=cfg["fecha_fin"],
            raw_dir=raw_dir,
            min_days=cfg["min_days_per_ticker"],
            min_tickers_pct=cfg["min_tickers_pct"],
        )
    except RuntimeError as e:
        logger.error("ABORT: %s", e)
        sys.exit(1)

    # Build CBOE IV panel
    logger.info("Building CBOE IV panel …")
    try:
        cboe_panel = build_cboe_iv_panel(
            universe=cfg["universe"],
            start=cfg["fecha_inicio"],
            end=cfg["fecha_fin"],
            cache_dir=cboe_dir,
        )
    except Exception as e:
        logger.error("ABORT: CBOE panel failed: %s", e)
        print(
            f"\n⚠  CONDICIÓN DE PARADA: Error al construir el panel CBOE.\n"
            f"Ejecuta primero: python3.11 scripts/09_download_cboe.py\nError: {e}"
        )
        sys.exit(1)

    # Build per-ticker features_v2 + target_v2
    panels = []
    for ticker, df in universe_data.items():
        logger.info("Building features_v2 for %s …", ticker)

        # Extract the CBOE IV series for this ticker
        try:
            iv_cboe = cboe_panel.xs(ticker, level="ticker")["iv_cboe"]
        except KeyError:
            logger.warning("No CBOE data for %s, skipping", ticker)
            continue

        feats  = build_features_v2(df, iv_cboe, cfg)
        target = compute_target_v2(df, iv_cboe, cfg)

        ticker_panel = feats.copy()
        ticker_panel["target"] = target
        ticker_panel["ticker"] = ticker
        ticker_panel["date"]   = df.index
        ticker_panel = ticker_panel.set_index(["date", "ticker"])
        panels.append(ticker_panel)

    panel = pd.concat(panels).sort_index()

    n_before = len(panel)
    panel = panel.dropna()
    n_after = len(panel)
    logger.info("Dropped %d rows with NaN (%d → %d)", n_before - n_after, n_before, n_after)

    if n_after < MIN_OBS:
        msg = (
            f"ABORT: Solo {n_after:,} observaciones utilizables (< {MIN_OBS:,}).\n"
            "Revisa los datos CBOE antes de continuar."
        )
        logger.error(msg)
        print(f"\n⚠  CONDICIÓN DE PARADA\n{msg}")
        sys.exit(1)

    out_path = processed_dir / "panel_cboe.parquet"
    panel.to_parquet(out_path)
    logger.info("Panel CBOE guardado en %s", out_path)

    base_rate = panel["target"].mean()
    print(f"\n✓ Panel CBOE construido:")
    print(f"  Tickers:               {panel.index.get_level_values('ticker').nunique()}")
    print(f"  Observaciones totales: {len(panel):,}")
    print(f"  Base rate (y=1):       {base_rate*100:.2f}%")
    print(
        f"  Rango fechas: "
        f"{panel.index.get_level_values('date').min().date()} → "
        f"{panel.index.get_level_values('date').max().date()}"
    )
    print(f"  Features: {FEATURE_NAMES_V2}")
    print(f"  Guardado en: {out_path}")


if __name__ == "__main__":
    main()
