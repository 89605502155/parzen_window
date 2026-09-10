"""Parzen window classifier with incremental updates, compaction and persistence."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array, check_is_fitted

from . import io as _io
from .compaction import select_points_to_remove
from .kernels import resolve_kernel
from .scheduler import CompactionScheduler

_NOT_FITTED_MSG = "This ParzenWindowClassifier instance is not fitted yet. Call fit() first."


class ParzenWindowClassifier(BaseEstimator, ClassifierMixin):
    """Non-parametric classifier based on the Parzen window method.

    Parameters
    ----------
    h : float, default=0.5
        Kernel bandwidth. Used as-is when ``adaptive_bandwidth=False``; used
        as a scale factor on the per-point bandwidth when it is ``True``.
    kernel : {"gaussian", "epanechnikov", "quartic", "triangular", "rectangular"}
        Kernel function shape.
    adaptive_bandwidth : bool, default=False
        When ``True``, each training point gets its own bandwidth equal to
        ``h`` times its distance to its ``bandwidth_neighbors``-th nearest
        neighbor, instead of a single global ``h``. This lets the kernel
        widen in sparse regions and narrow in dense ones (a variable-kernel
        / "balloon" estimator), which the fixed-bandwidth Parzen window
        cannot do.
    bandwidth_neighbors : int, default=5
        Number of neighbors used to derive the adaptive bandwidth. Ignored
        when ``adaptive_bandwidth=False``.
    """

    def __init__(
        self,
        h: float = 0.5,
        kernel: str = "gaussian",
        adaptive_bandwidth: bool = False,
        bandwidth_neighbors: int = 5,
    ):
        self.h = h
        self.kernel = kernel
        self.adaptive_bandwidth = adaptive_bandwidth
        self.bandwidth_neighbors = bandwidth_neighbors
        self._scheduler = CompactionScheduler()
        self._bandwidth_cache = None

    # -- persistence-related plumbing --------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_scheduler", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._scheduler = CompactionScheduler()

    @property
    def is_compacting(self) -> bool:
        return self._scheduler.is_compacting

    # -- fitting --------------------------------------------------------

    def fit(self, X, y):
        """(Re)fit the model from scratch, discarding any previous data."""
        X = check_array(X)
        y = np.asarray(y)
        return self._scheduler.call(self._fit_impl, X, y)

    def _fit_impl(self, X, y):
        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)
        self._bandwidth_cache = None
        return self

    def partial_fit(self, X, y):
        """Incrementally fit the model on a new batch, keeping prior data.

        While a :meth:`compact` call is in progress on another thread, this
        call does not fail -- it is queued and applied automatically right
        after compaction finishes.
        """
        check_is_fitted(self, "X_train_", msg=_NOT_FITTED_MSG)
        X = check_array(X)
        y = np.asarray(y)
        return self._scheduler.call(self._partial_fit_impl, X, y)

    def _partial_fit_impl(self, X, y):
        self.X_train_ = np.vstack([self.X_train_, X])
        self.y_train_ = np.concatenate([self.y_train_, y])
        self.classes_ = np.unique(self.y_train_)
        self._bandwidth_cache = None
        return self

    # -- inference --------------------------------------------------------

    def predict(self, X):
        """Predict class labels. Queued automatically if compaction is running."""
        check_is_fitted(self, "X_train_", msg=_NOT_FITTED_MSG)
        X = check_array(X)
        return self._scheduler.call(self._predict_impl, X)

    def _predict_impl(self, X):
        scores = self._class_scores(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        """Predict per-class weights normalized to sum to 1 for each sample."""
        check_is_fitted(self, "X_train_", msg=_NOT_FITTED_MSG)
        X = check_array(X)
        return self._scheduler.call(self._predict_proba_impl, X)

    def _predict_proba_impl(self, X):
        scores = self._class_scores(X)
        totals = scores.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0  # query point has ~zero kernel weight everywhere
        return scores / totals

    def _class_scores(self, X):
        kernel_fn = resolve_kernel(self.kernel)
        distances = cdist(X, self.X_train_)
        kernel_values = kernel_fn(distances / self._bandwidth())
        scores = np.zeros((len(X), len(self.classes_)))
        for i, label in enumerate(self.classes_):
            scores[:, i] = kernel_values[:, self.y_train_ == label].sum(axis=1)
        return scores

    def _bandwidth(self):
        if not self.adaptive_bandwidth:
            return self.h
        if self._bandwidth_cache is None:
            self._bandwidth_cache = self._compute_adaptive_bandwidth()
        return self._bandwidth_cache

    def _compute_adaptive_bandwidth(self):
        n_samples = len(self.X_train_)
        k = min(self.bandwidth_neighbors, n_samples - 1)
        if k < 1:
            return np.full(n_samples, self.h)
        neighbors = NearestNeighbors(n_neighbors=k + 1).fit(self.X_train_)
        distances, _ = neighbors.kneighbors(self.X_train_)
        bandwidth = self.h * distances[:, -1]
        bandwidth[bandwidth == 0] = self.h
        return bandwidth

    # -- memory management --------------------------------------------------------

    def compact(self, reduction_fraction: float, n_neighbors: int = 5, min_per_class: int = 2) -> int:
        """Drop training points that lie deep inside their own class's territory.

        Uses convex-neighborhood analysis (see :mod:`parzen_window.compaction`)
        to find points that are both surrounded entirely by same-class
        neighbors and far from the decision boundary, then removes up to
        ``reduction_fraction`` of the dataset. While this runs, other threads'
        ``predict``/``partial_fit`` calls are queued and executed automatically
        as soon as compaction completes, instead of failing.

        Returns the number of points actually removed.
        """
        check_is_fitted(self, "X_train_", msg=_NOT_FITTED_MSG)
        return self._scheduler.run_exclusively(
            self._compact_impl, reduction_fraction, n_neighbors, min_per_class
        )

    def _compact_impl(self, reduction_fraction, n_neighbors, min_per_class):
        to_remove = select_points_to_remove(
            self.X_train_, self.y_train_, reduction_fraction, n_neighbors, min_per_class
        )
        if len(to_remove) == 0:
            return 0
        mask = np.ones(len(self.X_train_), dtype=bool)
        mask[to_remove] = False
        self.X_train_ = self.X_train_[mask]
        self.y_train_ = self.y_train_[mask]
        self._bandwidth_cache = None
        return len(to_remove)

    # -- persistence --------------------------------------------------------

    def save(self, path) -> None:
        """Save model weights to a ``.npz`` file for backup or transfer."""
        check_is_fitted(self, "X_train_", msg=_NOT_FITTED_MSG)
        self._scheduler.call(
            _io.save_npz,
            path,
            h=self.h,
            kernel=self.kernel,
            adaptive_bandwidth=self.adaptive_bandwidth,
            bandwidth_neighbors=self.bandwidth_neighbors,
            X_train=self.X_train_,
            y_train=self.y_train_,
            classes=self.classes_,
        )

    @classmethod
    def load(cls, path) -> "ParzenWindowClassifier":
        """Load a model previously saved with :meth:`save`."""
        data = _io.load_npz(path)
        model = cls(
            h=data["h"],
            kernel=data["kernel"],
            adaptive_bandwidth=data["adaptive_bandwidth"],
            bandwidth_neighbors=data["bandwidth_neighbors"],
        )
        model.X_train_ = data["X_train"]
        model.y_train_ = data["y_train"]
        model.classes_ = data["classes"]
        return model
