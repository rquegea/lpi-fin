"""
Script 02: Build features + target panel from cached OHLCV data.

Usage:
    python scripts/02_build_panel.py

Reads data/raw/{ticker}.parquet for each ticker in the universe.
Computes all 7 features and the binary target.
Saves the combined panel to data/processed/panel.parquet.

Abort conditions:
  - Panel has fewer than 50,000 usable (non-NaN) observations.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_universe
from src.features import build_features
from src.target import compute_target

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

MIN_OBS = 50_000


def main():
    root = Path(__file__).parent.parent
    with open(root / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(exist_ok=True)

    logger.info("Loading cached OHLCV data …")
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

    panels = []
    for ticker, df in universe_data.items():
        logger.info("Building features for %s …", ticker)
        feats = build_features(df, cfg)
        tgt = compute_target(df, cfg)
        ticker_panel = feats.copy()
        ticker_panel["target"] = tgt
        ticker_panel["ticker"] = ticker
        ticker_panel["date"] = df.index
        ticker_panel = ticker_panel.set_index(["date", "ticker"])
        panels.append(ticker_panel)

    panel = pd.concat(panels).sort_index()

    # Drop rows with any NaN in features or target
    n_before = len(panel)
    panel = panel.dropna()
    n_after = len(panel)
    logger.info("Dropped %d rows with NaN (%d → %d)", n_before - n_after, n_before, n_after)

    if n_after < MIN_OBS:
        msg = (
            f"ABORT: Solo {n_after:,} observaciones utilizables (< {MIN_OBS:,} requeridas).\n"
            "Por favor revisa los datos antes de continuar."
        )
        logger.error(msg)
        print(f"\n⚠  CONDICIÓN DE PARADA ACTIVADA\n{msg}")
        sys.exit(1)

    out_path = processed_dir / "panel.parquet"
    panel.to_parquet(out_path)
    logger.info("Panel guardado en %s", out_path)

    # Summary stats
    base_rate = panel["target"].mean()
    print(f"\n✓ Panel construido:")
    print(f"  Tickers: {panel.index.get_level_values('ticker').nunique()}")
    print(f"  Observaciones totales: {len(panel):,}")
    print(f"  Base rate (y=1): {base_rate*100:.2f}%")
    print(f"  Rango fechas: {panel.index.get_level_values('date').min().date()} → "
          f"{panel.index.get_level_values('date').max().date()}")
    print(f"  Guardado en: {out_path}")


if __name__ == "__main__":
    main()
