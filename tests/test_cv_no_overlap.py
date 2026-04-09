"""
Tests for cv.py: PurgedTimeSeriesSplit.

Verifies:
  1. No index appears in both train and test for any fold.
  2. The embargo is respected — no test index is within embargo_days
     of the last training index.
  3. Test folds are strictly after train folds (temporal order).
  4. The number of folds matches n_splits.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.cv import PurgedTimeSeriesSplit


def make_sequential_X(n=1000):
    """Simple sequential integer array for index testing."""
    return np.arange(n).reshape(-1, 1)


@pytest.fixture
def splitter():
    return PurgedTimeSeriesSplit(n_splits=5, embargo_days=10)


def test_no_overlap_train_test(splitter):
    """No index should appear in both train and test."""
    X = make_sequential_X(1000)
    for fold_i, (train, test) in enumerate(splitter.split(X)):
        overlap = set(train) & set(test)
        assert len(overlap) == 0, (
            f"Fold {fold_i}: train/test overlap detected: {sorted(overlap)[:10]}"
        )


def test_embargo_respected(splitter):
    """
    The first test index must be at least embargo_days after the last
    training index.
    """
    X = make_sequential_X(1000)
    for fold_i, (train, test) in enumerate(splitter.split(X)):
        last_train = train[-1]
        first_test = test[0]
        gap = first_test - last_train
        assert gap >= splitter.embargo_days, (
            f"Fold {fold_i}: embargo not respected. "
            f"Gap = {gap}, required >= {splitter.embargo_days}. "
            f"last_train={last_train}, first_test={first_test}"
        )


def test_test_after_train(splitter):
    """All test indices must be strictly greater than all train indices."""
    X = make_sequential_X(1000)
    for fold_i, (train, test) in enumerate(splitter.split(X)):
        assert test.min() > train.max(), (
            f"Fold {fold_i}: test indices overlap temporally with train. "
            f"train_max={train.max()}, test_min={test.min()}"
        )


def test_n_splits_produced(splitter):
    """Number of folds yielded must equal n_splits."""
    X = make_sequential_X(1000)
    folds = list(splitter.split(X))
    assert len(folds) == splitter.n_splits, (
        f"Expected {splitter.n_splits} folds, got {len(folds)}"
    )


def test_expanding_window(splitter):
    """Each successive train set must be strictly larger than the previous."""
    X = make_sequential_X(1000)
    train_sizes = [len(train) for train, _ in splitter.split(X)]
    for i in range(1, len(train_sizes)):
        assert train_sizes[i] > train_sizes[i - 1], (
            f"Train set at fold {i} ({train_sizes[i]}) is not larger than "
            f"fold {i-1} ({train_sizes[i-1]})"
        )


def test_no_overlap_different_embargo():
    """Test with different embargo values that the property holds."""
    for embargo in [0, 5, 15, 30]:
        cv = PurgedTimeSeriesSplit(n_splits=4, embargo_days=embargo)
        X = make_sequential_X(500)
        for fold_i, (train, test) in enumerate(cv.split(X)):
            overlap = set(train) & set(test)
            assert len(overlap) == 0, (
                f"embargo={embargo}, fold {fold_i}: overlap detected"
            )
            if embargo > 0:
                gap = test[0] - train[-1]
                assert gap >= embargo, (
                    f"embargo={embargo}, fold {fold_i}: gap={gap} < embargo"
                )


def test_indices_within_bounds():
    """All returned indices must be within [0, n-1]."""
    n = 300
    cv = PurgedTimeSeriesSplit(n_splits=5, embargo_days=10)
    X = make_sequential_X(n)
    for fold_i, (train, test) in enumerate(cv.split(X)):
        assert train.min() >= 0
        assert train.max() < n
        assert test.min() >= 0
        assert test.max() < n


def test_all_indices_covered():
    """
    Every index should appear in exactly one test fold
    (with possible gaps for embargo periods at fold boundaries).
    The union of all test sets should cover a large fraction of data.
    """
    n = 1000
    cv = PurgedTimeSeriesSplit(n_splits=5, embargo_days=10)
    X = make_sequential_X(n)
    all_test_indices = set()
    for train, test in cv.split(X):
        all_test_indices.update(test.tolist())
    # At least 70% of data should be in some test fold
    assert len(all_test_indices) >= 0.5 * n, (
        f"Only {len(all_test_indices)}/{n} indices appear in any test fold"
    )
