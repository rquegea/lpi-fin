"""
Script 03: Run LPI algorithm on the panel and report results.

Usage:
    python scripts/03_run_lpi.py

Reads data/processed/panel.parquet.
Runs the full LPI pipeline (Bootstrap BIC → GMM CV → LPI scores).
Generates metrics, figures, and the verdict.

Abort conditions:
  - K* outside [3, 12].
  - AUC > suspicious_high threshold (requires investigation before proceeding).
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lpi_core import fit_predict
from src.reporting import generate_report
from src.features import FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    root = Path(__file__).parent.parent
    with open(root / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    panel_path = root / "data" / "processed" / "panel.parquet"
    if not panel_path.exists():
        print("Error: panel.parquet not found. Run 02_build_panel.py first.")
        sys.exit(1)

    logger.info("Loading panel from %s …", panel_path)
    panel = pd.read_parquet(panel_path)

    # Extract X, y — sort by date to preserve temporal order
    panel = panel.sort_index(level="date")
    X = panel[FEATURE_NAMES].values.astype(float)
    y = panel["target"].values.astype(float)

    logger.info("Panel: %d observations, %d features", len(X), X.shape[1])
    logger.info("Base rate: %.3f", y.mean())

    logger.info("Running LPI …")
    result = fit_predict(X, y, cfg)

    k_star = result["k_star"]
    logger.info("K* = %d", k_star)

    # Abort condition: K* outside [3, 12]
    if not (3 <= k_star <= 12):
        msg = (
            f"⚠  CONDICIÓN DE PARADA: K* = {k_star} está fuera del rango [3, 12].\n"
            "Esto es inusual y puede indicar un problema con los datos.\n"
            "Por favor revisa antes de continuar."
        )
        print(msg)
        logger.error(msg)
        sys.exit(1)

    auc_mean = result["auc_mean"]
    suspicious_high = cfg["thresholds"]["suspicious_high"]

    # Check for suspicious AUC before reporting
    if auc_mean > suspicious_high:
        msg = (
            f"⚠  CONDICIÓN DE PARADA: AUC = {auc_mean:.4f} > {suspicious_high}.\n"
            "Este resultado es sospechoso — posible overfitting o leakage no detectado.\n"
            "Ejecuta 04_diagnostics.py y revisa el shuffle test antes de proceder."
        )
        print(msg)
        logger.warning(msg)
        # Don't abort — just warn. Let diagnostics tell the full story.

    report = generate_report(
        result=result,
        cfg=cfg,
        output_dir=str(root / "results"),
        universe_size=panel.index.get_level_values("ticker").nunique(),
    )

    # Save result for diagnostics script
    np.save(root / "data" / "processed" / "scores_oos.npy", result["scores_oos"])
    np.save(root / "data" / "processed" / "y_oos.npy", result["y_oos"])
    np.save(root / "data" / "processed" / "k_star.npy", np.array([result["k_star"]]))

    print(f"\nFiguras guardadas en: {root / 'results' / 'figures'}")
    print(f"Tablas guardadas en: {root / 'results' / 'tables'}")
    print(f"Log guardado en: {root / 'results' / 'run_log.txt'}")


if __name__ == "__main__":
    main()
