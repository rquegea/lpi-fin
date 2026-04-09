# EXPERIMENTO DESCARTADO (2026-04-09)
# Resultado: v3 single AUC 0.6015, v3 ensemble AUC 0.5785
# Peor que baseline CBOE 4-features (AUC 0.6253).
# Razón probable: maldición de dimensionalidad en GMM al pasar de
# 4 a 7 features. Se mantiene código como registro del intento.
# Modelo final del proyecto: CBOE 4-features. Ver scripts 09-13.

"""
Script 14: Build features_v3 + target_v2 panel (7 features).

Usage:
    python3.11 scripts/14_build_panel_v3.py

Reads:
  - data/raw/{ticker}.parquet  (OHLCV)
  - data/raw/cboe/{VIX,VXN}.parquet  (must run 09 first)

Outputs:
  - data/processed/panel_v3.parquet  (features_v3 + target_v2)

Abort conditions:
  - Fewer than 60,000 usable observations after merge and NaN drop.
  - Correlation > 0.95 between any new feature and any original feature.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_universe
from src.cboe_data import build_cboe_iv_panel
from src.features_v2 import build_features_v3, FEATURE_NAMES_V3
from src.target_v2 import compute_target_v2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

MIN_OBS = 60_000
NEW_FEATURES = ["skew_60d", "kurt_60d", "vol_autocorr_5d"]
OLD_FEATURES = ["iv_rv_spread_cboe", "log_range", "vol_of_vol", "log_dvol"]
CORR_THRESHOLD = 0.95


def main():
    root = Path(__file__).parent.parent
    with open(root / "config_v3.yaml") as f:
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

    # Build per-ticker features_v3 + target_v2
    panels = []
    for ticker, df in universe_data.items():
        logger.info("Building features_v3 for %s …", ticker)

        try:
            iv_cboe = cboe_panel.xs(ticker, level="ticker")["iv_cboe"]
        except KeyError:
            logger.warning("No CBOE data for %s, skipping", ticker)
            continue

        feats  = build_features_v3(df, iv_cboe, cfg)
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
            "Revisa los datos antes de continuar."
        )
        logger.error(msg)
        print(f"\n⚠  CONDICIÓN DE PARADA\n{msg}")
        sys.exit(1)

    # Check correlations between new and old features
    corr = panel[FEATURE_NAMES_V3].corr()
    stop_corr = False
    for nf in NEW_FEATURES:
        for of in OLD_FEATURES:
            c = abs(corr.loc[nf, of])
            if c > CORR_THRESHOLD:
                print(
                    f"\n⚠  CONDICIÓN DE PARADA: correlación {c:.4f} entre '{nf}' y '{of}' "
                    f"> {CORR_THRESHOLD} — feature redundante. Investigar antes de continuar."
                )
                stop_corr = True
    if stop_corr:
        sys.exit(1)

    out_path = processed_dir / "panel_v3.parquet"
    panel.to_parquet(out_path)
    logger.info("Panel v3 guardado en %s", out_path)

    base_rate = panel["target"].mean()
    print(f"\n✓ Panel v3 construido:")
    print(f"  Tickers:               {panel.index.get_level_values('ticker').nunique()}")
    print(f"  Observaciones totales: {len(panel):,}")
    print(f"  Base rate (y=1):       {base_rate*100:.2f}%")
    print(
        f"  Rango fechas: "
        f"{panel.index.get_level_values('date').min().date()} → "
        f"{panel.index.get_level_values('date').max().date()}"
    )
    print(f"  Features: {FEATURE_NAMES_V3}")
    print(f"  Guardado en: {out_path}")

    # Print correlation matrix between new and old features
    print("\n  Correlaciones entre features nuevas y originales:")
    print(f"  {'':25s}", end="")
    for of in OLD_FEATURES:
        print(f"  {of:>20s}", end="")
    print()
    for nf in NEW_FEATURES:
        print(f"  {nf:25s}", end="")
        for of in OLD_FEATURES:
            print(f"  {corr.loc[nf, of]:>20.4f}", end="")
        print()


if __name__ == "__main__":
    main()
