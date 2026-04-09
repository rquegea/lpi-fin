# EXPERIMENTO DESCARTADO (2026-04-09)
# Resultado: v3 single AUC 0.6015, v3 ensemble AUC 0.5785
# Peor que baseline CBOE 4-features (AUC 0.6253).
# Razón probable: maldición de dimensionalidad en GMM al pasar de
# 4 a 7 features. Se mantiene código como registro del intento.
# Modelo final del proyecto: CBOE 4-features. Ver scripts 09-13.

"""
Tests for lpi_ensemble.run_lpi_ensemble.

Four tests:
  1. Separable Gaussians: ensemble AUC > 0.95
  2. Random labels: ensemble AUC < 0.60
  3. Reproducibility: same seeds → bit-identical scores_oos
  4. Output size: scores_oos length == y_oos length, k_star_list length == n_models
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.lpi_ensemble import run_lpi_ensemble


@pytest.fixture
def ensemble_cfg():
    """Minimal cfg for fast ensemble tests (bootstrap_b=5 for speed)."""
    return {
        "n_folds": 3,
        "embargo_days": 5,
        "random_state": 42,
        "k_min": 2,
        "k_max": 6,
        "bootstrap_b": 5,
        "bootstrap_subsample": 300,
        "gmm_reg_covar": 1e-5,
        "gmm_max_iter": 300,
    }


def make_two_gaussians(n=2000, seed=42):
    """Two well-separated 2D Gaussians, shuffled for realistic CV conditions."""
    rng = np.random.default_rng(seed)
    half = n // 2
    X0 = rng.multivariate_normal([-3, -3], [[0.5, 0], [0, 0.5]], size=half)
    X1 = rng.multivariate_normal([3, 3], [[0.5, 0], [0, 0.5]], size=half)
    X = np.vstack([X0, X1])
    y = np.array([0.0] * half + [1.0] * half)
    idx = rng.permutation(n)
    return X[idx], y[idx]


def test_ensemble_auc_separable(ensemble_cfg):
    """Ensemble must achieve AUC > 0.95 on two well-separated Gaussians."""
    X, y = make_two_gaussians(n=2000)
    result = run_lpi_ensemble(X, y, ensemble_cfg, n_models=3, seeds=[42, 7, 123])
    assert result["auc_mean"] > 0.95, (
        f"Ensemble AUC on separable data should be > 0.95, got {result['auc_mean']:.4f}"
    )


def test_ensemble_random_labels(ensemble_cfg):
    """Ensemble AUC must be near 0.50 when labels are random (no signal)."""
    X, _ = make_two_gaussians(n=2000)
    rng = np.random.default_rng(999)
    y_random = rng.integers(0, 2, size=len(X)).astype(float)
    result = run_lpi_ensemble(X, y_random, ensemble_cfg, n_models=3, seeds=[42, 7, 123])
    assert result["auc_mean"] < 0.60, (
        f"Ensemble AUC on random labels should be < 0.60, got {result['auc_mean']:.4f}"
    )


def test_ensemble_reproducibility(ensemble_cfg):
    """Two calls with identical seeds must produce bit-identical scores_oos."""
    X, y = make_two_gaussians(n=1000)
    seeds = [42, 7, 123]
    r1 = run_lpi_ensemble(X, y, ensemble_cfg, n_models=3, seeds=seeds)
    r2 = run_lpi_ensemble(X, y, ensemble_cfg, n_models=3, seeds=seeds)
    np.testing.assert_array_equal(
        r1["scores_oos"], r2["scores_oos"],
        err_msg="Ensemble scores are not reproducible across identical runs",
    )


def test_ensemble_output_size(ensemble_cfg):
    """scores_oos and y_oos must match in length; k_star_list length == n_models."""
    X, y = make_two_gaussians(n=1000)
    n_models = 3
    result = run_lpi_ensemble(X, y, ensemble_cfg, n_models=n_models, seeds=[42, 7, 123])
    assert len(result["scores_oos"]) == len(result["y_oos"]), (
        "scores_oos and y_oos must have identical length"
    )
    assert len(result["scores_oos"]) > 0, "scores_oos must be non-empty"
    assert len(result["k_star_list"]) == n_models, (
        f"k_star_list must have {n_models} entries, got {len(result['k_star_list'])}"
    )
