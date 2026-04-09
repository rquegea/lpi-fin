# EXPERIMENTO DESCARTADO (2026-04-09)
# Resultado: v3 single AUC 0.6015, v3 ensemble AUC 0.5785
# Peor que baseline CBOE 4-features (AUC 0.6253).
# Razón probable: maldición de dimensionalidad en GMM al pasar de
# 4 a 7 features. Se mantiene código como registro del intento.
# Modelo final del proyecto: CBOE 4-features. Ver scripts 09-13.

"""
LPI Ensemble: average OOS scores across multiple random seeds.

Motivation: Bootstrap BIC K* selection and GMM fitting both depend on
random_state. Averaging scores across seeds reduces variance without
touching any architectural choices or CV splits.

Algorithm:
  1. For each seed in seeds:
       a. Override cfg["random_state"] = seed.
       b. Run fit_predict(X, y, cfg_seed).
       c. Collect scores_oos, y_oos, k_star.
  2. Average the n_models score arrays elementwise (simple mean).
  3. Compute AUC on the aggregated scores against y_oos.

Determinism guarantee: PurgedTimeSeriesSplit depends only on len(X)
and (n_splits, embargo_days) — not on random_state. So all seeds
produce identical train/test splits, identical y_oos ordering, and
the score arrays can be safely averaged.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from src.lpi_core import fit_predict

ENSEMBLE_SEEDS = [42, 7, 123, 2024, 99, 314, 2718]


def run_lpi_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    n_models: int = 7,
    seeds: list = None,
) -> dict:
    """
    Train n_models LPI instances with different random seeds and
    return averaged OOS scores.

    Parameters
    ----------
    X        : ndarray of shape (n_samples, n_features).
    y        : ndarray of shape (n_samples,). Binary labels (0/1).
    cfg      : dict. Same format as lpi_core.fit_predict cfg.
               random_state is overridden per seed; all other keys unchanged.
               Caller may reduce cfg["bootstrap_b"] before calling if
               runtime is a concern (document the change in the log).
    n_models : int, default 7.
    seeds    : list of int, default ENSEMBLE_SEEDS[:n_models].

    Returns
    -------
    dict with keys:
      scores_oos      : ndarray — averaged LPI scores (shape n_oos,)
      y_oos           : ndarray — labels for OOS samples (identical across seeds)
      k_star_list     : list[int] — K* selected per seed
      auc_mean        : float — AUC on aggregated scores
      auc_std         : float — std of per-seed AUCs (spread metric)
      per_seed_aucs   : list[float] — AUC of each individual seed
      per_seed_results: list[dict] — raw fit_predict output per seed

    Raises
    ------
    ValueError
        If scores_oos lengths differ across seeds (should never happen
        with identical cfg and X).
    """
    if seeds is None:
        seeds = ENSEMBLE_SEEDS[:n_models]

    scores_list: list = []
    per_seed_aucs: list = []
    k_stars: list = []
    results: list = []

    for seed in seeds:
        r = fit_predict(X, y, dict(cfg, random_state=seed))
        scores_list.append(r["scores_oos"])
        per_seed_aucs.append(r["auc_mean"])
        k_stars.append(r["k_star"])
        results.append(r)

    # Validate identical lengths (mandatory for valid averaging)
    lengths = [len(s) for s in scores_list]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"scores_oos lengths differ across seeds: {dict(zip(seeds, lengths))}. "
            "This should never happen with identical cfg and X — check your config."
        )

    # Average scores elementwise
    scores_oos = np.vstack(scores_list).mean(axis=0)
    y_oos = results[0]["y_oos"]  # identical across all seeds

    if len(np.unique(y_oos)) > 1:
        auc_mean = float(roc_auc_score(y_oos, scores_oos))
    else:
        auc_mean = float("nan")

    return {
        "scores_oos": scores_oos,
        "y_oos": y_oos,
        "k_star_list": k_stars,
        "auc_mean": auc_mean,
        "auc_std": float(np.std(per_seed_aucs)),
        "per_seed_aucs": per_seed_aucs,
        "per_seed_results": results,
    }
