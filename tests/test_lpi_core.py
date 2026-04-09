"""
Tests for lpi_core.py.

Verifies the LPI algorithm on a synthetic dataset with a known answer:
two well-separated 2D Gaussians, labels determined by cluster membership.
If the algorithm is correct it should find AUC > 0.95.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.lpi_core import fit_predict, _bootstrap_bic
from sklearn.preprocessing import RobustScaler


@pytest.fixture
def synthetic_cfg():
    return {
        "n_folds": 3,
        "embargo_days": 5,
        "random_state": 42,
        "k_min": 2,
        "k_max": 6,
        "bootstrap_b": 10,
        "bootstrap_subsample": 500,
        "gmm_reg_covar": 1e-5,
        "gmm_max_iter": 300,
    }


def make_two_gaussians(n=2000, seed=42):
    """
    Two well-separated 2D Gaussians.
    Cluster 0: center (-3, -3), y=0
    Cluster 1: center (+3, +3), y=1
    Returns X (n, 2), y (n,) in temporal order (sorted by a fake time index).
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    X0 = rng.multivariate_normal([-3, -3], [[0.5, 0], [0, 0.5]], size=half)
    X1 = rng.multivariate_normal([3, 3], [[0.5, 0], [0, 0.5]], size=half)
    X = np.vstack([X0, X1])
    y = np.array([0.0] * half + [1.0] * half)
    # Shuffle to simulate non-block structure, but keep temporal index intact
    idx = rng.permutation(n)
    return X[idx], y[idx]


def test_fit_predict_auc_on_separable_data(synthetic_cfg):
    """LPI must achieve AUC > 0.95 on two well-separated Gaussians."""
    X, y = make_two_gaussians(n=3000)
    result = fit_predict(X, y, synthetic_cfg)
    assert result["auc_mean"] > 0.95, (
        f"Expected AUC > 0.95 on separable synthetic data, got {result['auc_mean']:.4f}"
    )


def test_fit_predict_returns_required_keys(synthetic_cfg):
    """fit_predict must return all expected keys."""
    X, y = make_two_gaussians(n=1000)
    result = fit_predict(X, y, synthetic_cfg)
    required_keys = {"scores_oos", "y_oos", "k_star", "auc_folds", "auc_mean", "auc_std", "fold_sizes"}
    assert required_keys.issubset(result.keys())


def test_fit_predict_scores_length(synthetic_cfg):
    """OOS scores must cover all samples (minus embargo gaps)."""
    X, y = make_two_gaussians(n=1000)
    result = fit_predict(X, y, synthetic_cfg)
    # OOS scores may be slightly fewer than n due to embargo
    assert len(result["scores_oos"]) == len(result["y_oos"])
    assert len(result["scores_oos"]) > 0


def test_k_star_in_range(synthetic_cfg):
    """K* must be within [k_min, k_max]."""
    X, y = make_two_gaussians(n=1000)
    result = fit_predict(X, y, synthetic_cfg)
    assert synthetic_cfg["k_min"] <= result["k_star"] <= synthetic_cfg["k_max"]


def test_bootstrap_bic_selects_k2_for_two_gaussians(synthetic_cfg):
    """Bootstrap BIC should select K=2 for data generated from 2 Gaussians."""
    X, _ = make_two_gaussians(n=2000)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    rng = np.random.default_rng(42)
    k_star = _bootstrap_bic(X_scaled, synthetic_cfg, rng)
    # Allow some slack: 2 or 3 are both acceptable
    assert k_star in [2, 3], f"Expected K*=2 or 3 for two Gaussians, got {k_star}"


def test_shuffle_degrades_auc(synthetic_cfg):
    """Shuffling y must degrade AUC to near 0.50."""
    X, y = make_two_gaussians(n=2000)
    rng = np.random.default_rng(123)
    y_shuffled = rng.permutation(y)
    result = fit_predict(X, y_shuffled, synthetic_cfg)
    assert result["auc_mean"] < 0.60, (
        f"Shuffled AUC should be near 0.50, got {result['auc_mean']:.4f}"
    )


def test_scores_are_bounded(synthetic_cfg):
    """LPI scores must be in [0, 1] since they are weighted averages of rates."""
    X, y = make_two_gaussians(n=1000)
    result = fit_predict(X, y, synthetic_cfg)
    scores = result["scores_oos"]
    assert np.all(scores >= -1e-9), "Scores contain values below 0"
    assert np.all(scores <= 1 + 1e-9), "Scores contain values above 1"
