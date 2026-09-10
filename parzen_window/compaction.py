"""Dataset compaction via convex-neighborhood analysis.

A training point contributes almost nothing to the decision boundary when
it is (a) surrounded entirely by same-class neighbors -- i.e. it sits deep
inside a convex, single-class neighborhood -- and (b) far from the nearest
point of any other class. Such points are safe to drop to free up memory,
since the kernel weight they contribute is dominated by points closer to
the actual boundary.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_removal_scores(X: np.ndarray, y: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Score each point by how safe it is to remove (higher = safer).

    A point scores 0 (never removed) unless all of its ``n_neighbors``
    nearest neighbors share its label; otherwise its score is the distance
    to the nearest point of a *different* class -- a proxy for how far it
    sits from the decision boundary.
    """
    n_samples = len(X)
    k = min(n_neighbors, n_samples - 1)
    if k < 1:
        return np.zeros(n_samples)

    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, indices = neighbors.kneighbors(X)
    indices = indices[:, 1:]  # drop the point itself (distance 0)
    homogeneous = (y[indices] == y[:, None]).all(axis=1)

    margin = np.zeros(n_samples)
    for label in np.unique(y):
        own, other = y == label, y != label
        if not other.any() or not own.any():
            continue
        boundary = NearestNeighbors(n_neighbors=1).fit(X[other])
        boundary_distance, _ = boundary.kneighbors(X[own])
        margin[own] = boundary_distance[:, 0]

    return np.where(homogeneous, margin, 0.0)


def select_points_to_remove(
    X: np.ndarray,
    y: np.ndarray,
    reduction_fraction: float,
    n_neighbors: int = 5,
    min_per_class: int = 2,
) -> np.ndarray:
    """Pick indices that can be dropped without touching the decision boundary.

    Points are removed greedily, safest (highest score) first, until either
    ``reduction_fraction`` of the dataset has been removed, points with a
    zero score are reached (mixed neighborhood or near the boundary), or a
    class would drop below ``min_per_class`` remaining points.
    """
    if not 0.0 < reduction_fraction < 1.0:
        raise ValueError("reduction_fraction must be in (0, 1)")

    n_samples = len(X)
    target_removals = int(n_samples * reduction_fraction)
    if target_removals <= 0:
        return np.array([], dtype=int)

    scores = compute_removal_scores(X, y, n_neighbors)
    order = np.argsort(scores)[::-1]
    remaining_per_class = {label: int(np.sum(y == label)) for label in np.unique(y)}

    to_remove = []
    for idx in order:
        if len(to_remove) >= target_removals or scores[idx] <= 0:
            break
        label = y[idx]
        if remaining_per_class[label] - 1 < min_per_class:
            continue
        remaining_per_class[label] -= 1
        to_remove.append(idx)

    return np.array(sorted(to_remove), dtype=int)
