"""
Script 04: Run diagnostics on the real dataset.

Usage:
    python scripts/04_diagnostics.py

Requires that 03_run_lpi.py has already been run (reads saved scores).

Runs:
  1. Shuffle test (12 permutations) — detect leakage
  2. Leave-one-feature-out ablation — identify critical features
  3. Multi-seed stability test — verify result consistency

Abort condition:
  - Shuffle test does not collapse to 0.50 ± 0.02 (prints prominent warning).
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diagnostics import shuffle_test, ablation_test, stability_test
from src.reporting import print_diagnostics_report
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
    scores_path = root / "data" / "processed" / "scores_oos.npy"

    if not panel_path.exists():
        print("Error: panel.parquet not found. Run 02_build_panel.py first.")
        sys.exit(1)

    logger.info("Loading panel …")
    panel = pd.read_parquet(panel_path)
    panel = panel.sort_index(level="date")
    X = panel[FEATURE_NAMES].values.astype(float)
    y = panel["target"].values.astype(float)

    # Load real AUC from saved run
    auc_real = None
    if scores_path.exists():
        scores = np.load(scores_path)
        y_oos = np.load(root / "data" / "processed" / "y_oos.npy")
        from sklearn.metrics import roc_auc_score
        auc_real = float(roc_auc_score(y_oos, scores))
        logger.info("Real AUC (from 03_run_lpi.py): %.4f", auc_real)

    # 1. Shuffle test
    logger.info("Running shuffle test (%d permutations) …", cfg["shuffle_n_permutations"])
    shuf = shuffle_test(X, y, cfg)

    # 2. Ablation
    logger.info("Running feature ablation (leave-one-out) …")
    ablation_df = ablation_test(X, y, cfg, feature_names=FEATURE_NAMES)

    # 3. Stability
    logger.info("Running multi-seed stability test …")
    stab = stability_test(X, y, cfg)

    # Report
    print_diagnostics_report(
        shuffle_result=shuf,
        ablation_df=ablation_df,
        stability_result=stab,
        auc_real=auc_real if auc_real is not None else float("nan"),
        output_dir=str(root / "results"),
    )

    # Save ablation table
    ablation_path = root / "results" / "tables" / "ablation.csv"
    ablation_df.to_csv(ablation_path, index=False)
    logger.info("Ablation table saved to %s", ablation_path)

    # Final warning if leakage detected
    if shuf["leakage_detected"]:
        print(
            "\n⚠  STOP: El shuffle test detectó posible leakage.\n"
            "No interpretes el AUC real hasta resolver el problema.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
