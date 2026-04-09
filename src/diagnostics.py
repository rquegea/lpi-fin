"""
Diagnostics: shuffle test, feature ablation, multi-seed stability.

These three tests collectively answer:
  1. Is the signal real or is it data leakage? (shuffle test)
  2. Which features drive the signal? (ablation)
  3. Is the result stable across random seeds? (stability test)

The shuffle test is the most important. If it does NOT collapse to
0.50 ± tolerance, the pipeline has leakage somewhere. Stop and
investigate before trusting any other result.
"""

import logging
import sys
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.lpi_core import fit_predict
from src.features import FEATURE_NAMES

logger = logging.getLogger(__name__)

_LEAKAGE_WARNING = """
╔══════════════════════════════════════════════════════════════════╗
║          ⚠  POSIBLE LEAKAGE DETECTADO  ⚠                        ║
║                                                                  ║
║  El shuffle test NO colapsó a 0.50 ± {tol:.2f}                    ║
║  AUC permutado: {mean:.4f} ± {std:.4f}                             ║
║                                                                  ║
║  Qué revisar:                                                    ║
║  1. ¿Las features usan datos después de t?                       ║
║     → Revisar los docstrings en features.py                      ║
║  2. ¿El target fue construido antes de split train/test?         ║
║     → El target DEBE construirse antes del CV, pero los          ║
║       valores forward (t+1..t+5) sólo deben usarse en test.     ║
║  3. ¿El embargo es suficiente?                                   ║
║     → embargo_days debe ser >= rv5_window (default 5+buffer)     ║
║  4. ¿Hay filas duplicadas por ticker-date en el panel?           ║
║     → Verificar que el panel no tiene duplicados                 ║
║  NO continúes analizando el AUC real hasta resolver esto.        ║
╚══════════════════════════════════════════════════════════════════╝
"""


def shuffle_test(
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    n_permutations: Optional[int] = None,
) -> dict:
    """
    Shuffle test: permute y n_permutations times and run LPI each time.

    Uses the same K* and same CV as the real run. This verifies that
    the model has no access to future information — if it does, it would
    still find signal even with shuffled labels.

    Parameters
    ----------
    X : ndarray of shape (n, p)
    y : ndarray of shape (n,)
    cfg : dict  (must include shuffle_n_permutations if n_permutations=None)
    n_permutations : override cfg value

    Returns
    -------
    dict with:
      auc_permuted : list[float]
      auc_mean : float
      auc_std  : float
      auc_min  : float
      auc_max  : float
      p_empirical : float  (fraction of permuted AUCs >= real AUC, if provided)
      leakage_detected : bool
    """
    n_perm = n_permutations or cfg["shuffle_n_permutations"]
    tol = cfg["shuffle_collapse_tolerance"]
    rng = np.random.default_rng(cfg["random_state"] + 999)

    auc_permuted = []
    for i in range(n_perm):
        y_shuffled = rng.permutation(y)
        result = fit_predict(X, y_shuffled, cfg)
        auc_permuted.append(result["auc_mean"])
        logger.info("Shuffle %d/%d: AUC = %.4f", i + 1, n_perm, result["auc_mean"])

    auc_mean = float(np.mean(auc_permuted))
    auc_std = float(np.std(auc_permuted))
    auc_min = float(np.min(auc_permuted))
    auc_max = float(np.max(auc_permuted))

    # Leakage: permuted AUC is too far from 0.50
    leakage_detected = abs(auc_mean - 0.50) > tol

    if leakage_detected:
        print(
            _LEAKAGE_WARNING.format(tol=tol, mean=auc_mean, std=auc_std),
            file=sys.stderr,
        )

    return {
        "auc_permuted": auc_permuted,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "auc_min": auc_min,
        "auc_max": auc_max,
        "leakage_detected": leakage_detected,
    }


def compute_p_empirical(auc_real: float, auc_permuted: list) -> float:
    """Fraction of permuted AUCs >= real AUC (empirical p-value)."""
    return float(np.mean([a >= auc_real for a in auc_permuted]))


def ablation_test(
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    feature_names: Optional[list] = None,
) -> pd.DataFrame:
    """
    Leave-one-feature-out ablation.

    For each feature i, refit LPI without that feature and record
    delta_AUC = AUC_full - AUC_without_i.

    A large positive delta means the feature is important.
    A near-zero or negative delta means the feature adds no value
    (or is noise that the model learned to avoid).

    Parameters
    ----------
    X : ndarray of shape (n, p)
    y : ndarray of shape (n,)
    cfg : dict
    feature_names : list of length p (uses FEATURE_NAMES if None)

    Returns
    -------
    DataFrame with columns: feature, auc_without, delta_auc
    Sorted by delta_auc descending (most important first).
    """
    names = feature_names or FEATURE_NAMES
    assert len(names) == X.shape[1], "feature_names length must match X.shape[1]"

    # Baseline AUC
    baseline = fit_predict(X, y, cfg)
    auc_full = baseline["auc_mean"]
    logger.info("Ablation baseline AUC: %.4f", auc_full)

    rows = []
    for i, name in enumerate(names):
        cols = [j for j in range(X.shape[1]) if j != i]
        X_reduced = X[:, cols]
        result = fit_predict(X_reduced, y, cfg)
        delta = auc_full - result["auc_mean"]
        rows.append(
            {
                "feature": name,
                "auc_without": result["auc_mean"],
                "delta_auc": delta,
            }
        )
        logger.info(
            "Ablation drop=%s: AUC=%.4f delta=%.4f",
            name,
            result["auc_mean"],
            delta,
        )

    df = pd.DataFrame(rows).sort_values("delta_auc", ascending=False).reset_index(drop=True)
    return df


def stability_test(
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    seeds: Optional[list] = None,
) -> dict:
    """
    Multi-seed stability: run LPI with different random seeds.

    A std < 0.02 across seeds suggests the result is not seed-sensitive.

    Parameters
    ----------
    X : ndarray
    y : ndarray
    cfg : dict
    seeds : list of int (uses cfg['stability_seeds'] if None)

    Returns
    -------
    dict with:
      auc_by_seed : dict {seed: auc}
      auc_mean : float
      auc_std  : float
      is_stable : bool (std < 0.02)
    """
    seeds = seeds or cfg["stability_seeds"]
    auc_by_seed = {}
    for seed in seeds:
        cfg_seed = dict(cfg, random_state=seed)
        result = fit_predict(X, y, cfg_seed)
        auc_by_seed[seed] = result["auc_mean"]
        logger.info("Stability seed=%d: AUC=%.4f", seed, result["auc_mean"])

    aucs = list(auc_by_seed.values())
    auc_mean = float(np.mean(aucs))
    auc_std = float(np.std(aucs))
    is_stable = auc_std < 0.02

    return {
        "auc_by_seed": auc_by_seed,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "is_stable": is_stable,
    }
