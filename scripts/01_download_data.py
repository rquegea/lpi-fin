"""
Script 01: Download OHLCV data for all tickers in the universe.

Usage:
    python scripts/01_download_data.py

Downloads data from Yahoo Finance and caches it as Parquet files in data/raw/.
If a ticker's cache is already up to date, it is not re-downloaded.

Abort conditions (requires user decision):
  - Fewer than 25 of 30 tickers download successfully.
"""

import logging
import sys
from pathlib import Path

import yaml

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    root = Path(__file__).parent.parent
    cfg_path = root / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    raw_dir = root / "data" / "raw"
    universe = cfg["universe"]

    logger.info("Starting download for %d tickers: %s to %s",
                len(universe), cfg["fecha_inicio"], cfg["fecha_fin"])

    try:
        data = load_universe(
            universe=universe,
            start=cfg["fecha_inicio"],
            end=cfg["fecha_fin"],
            raw_dir=raw_dir,
            min_days=cfg["min_days_per_ticker"],
            min_tickers_pct=cfg["min_tickers_pct"],
        )
    except RuntimeError as e:
        logger.error("ABORT: %s", e)
        print(
            "\n⚠  CONDICIÓN DE PARADA ACTIVADA\n"
            "Demasiados tickers fallaron. Por favor revisa los logs y decide\n"
            "si continuar con el universo reducido o ajustar la lista.\n"
            f"Error: {e}"
        )
        sys.exit(1)

    print(f"\n✓ Descarga completada: {len(data)}/{len(universe)} tickers OK")
    for ticker, df in sorted(data.items()):
        print(f"  {ticker}: {len(df):,} días  [{df.index.min().date()} → {df.index.max().date()}]")


if __name__ == "__main__":
    main()
