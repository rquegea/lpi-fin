"""
PurgedTimeSeriesSplit: expanding-window cross-validation with embargo.

The embargo prevents leakage when the target uses future data.
If y(t) uses information from [t+1, t+5], then the first 5 days of
the test set are contaminated by the last rows of the training set
unless we enforce a gap (embargo) of at least 5 days between them.
We use embargo_days=10 as a conservative buffer.
"""

import numpy as np


class PurgedTimeSeriesSplit:
    """
    Expanding-window time series split with purging and embargo.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    embargo_days : int
        Number of rows to drop from the start of each test fold.
        These rows are adjacent to the training set and may be
        contaminated because the target uses t+1..t+5.
    """

    def __init__(self, n_splits: int = 5, embargo_days: int = 10):
        self.n_splits = n_splits
        self.embargo_days = embargo_days

    def split(self, X, y=None, groups=None):
        """
        Yield (train_indices, test_indices) pairs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix. Only n_samples is used.
        y : ignored
        groups : ignored

        Yields
        ------
        train : ndarray of int
        test : ndarray of int
        """
        n = len(X)
        fold_size = n // (self.n_splits + 1)

        for i in range(1, self.n_splits + 1):
            train_end = fold_size * i
            test_start = train_end + self.embargo_days
            test_end = fold_size * (i + 1)

            if test_end > n:
                test_end = n
            if test_start >= test_end:
                continue

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
