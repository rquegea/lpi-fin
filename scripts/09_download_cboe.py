"""
Script 09: Download CBOE volatility indices (^VIX, ^VXN) and cache them.

Usage:
    python3.11 scripts/09_download_cboe.py

Downloads: ^VIX and ^VXN from Yahoo Finance.
Cache location: data/raw/cboe/{VIX,VXN}.parquet

Abort conditions (per spec):
  - Any index fails to download correctly.
"""

import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cboe_data import download_cboe_index, CBOE_SYMBOLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    root = Path(__file__).parent.parent
    with open(root / "config_cboe.yaml") as f:
        cfg = yaml.safe_load(f)

    cache_dir = root / "data" / "raw" / "cboe"
    start = cfg["fecha_inicio"]
    end   = cfg["fecha_fin"]

    failed = []
    for symbol in CBOE_SYMBOLS:
        try:
            series = download_cboe_index(symbol, start, end, cache_dir)
            if series is None or series.empty:
                raise ValueError("Empty series returned")
            # Sanity check: values in percentage-point range
            if series.min() < 1.0 or series.max() > 200.0:
                logger.warning(
                    "%s has suspicious range: [%.2f, %.2f]",
                    symbol, series.min(), series.max(),
                )
            print(
                f"  {symbol}: {len(series):,} days  "
                f"[{series.index.min().date()} → {series.index.max().date()}]  "
                f"range [{series.min():.1f}, {series.max():.1f}]"
            )
        except Exception as exc:
            logger.error("FAILED %s: %s", symbol, exc)
            failed.append(symbol)

    if failed:
        print(
            f"\n⚠  CONDICIÓN DE PARADA: Los siguientes índices CBOE no se "
            f"descargaron correctamente: {failed}\n"
            "Revisa la conexión o el símbolo antes de continuar."
        )
        sys.exit(1)

    print(f"\n✓ Descarga CBOE completada: {len(CBOE_SYMBOLS)} índices OK")
    print(f"  Caché en: {cache_dir}")


if __name__ == "__main__":
    main()
