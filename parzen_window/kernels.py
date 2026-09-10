"""Kernel functions used by :class:`parzen_window.core.ParzenWindowClassifier`.

Every kernel accepts a numpy array of normalized distances ``u = distance / h``
and returns the corresponding weight, vectorized over arrays of any shape.
"""

from __future__ import annotations

import numpy as np


def gaussian(u: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)


def epanechnikov(u: np.ndarray) -> np.ndarray:
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)


def quartic(u: np.ndarray) -> np.ndarray:
    return np.where(np.abs(u) <= 1, (15.0 / 16.0) * (1 - u**2) ** 2, 0.0)


def triangular(u: np.ndarray) -> np.ndarray:
    return np.where(np.abs(u) <= 1, 1 - np.abs(u), 0.0)


def rectangular(u: np.ndarray) -> np.ndarray:
    return np.where(np.abs(u) <= 1, 0.5, 0.0)


KERNELS = {
    "gaussian": gaussian,
    "epanechnikov": epanechnikov,
    "quartic": quartic,
    "triangular": triangular,
    "rectangular": rectangular,
}


def resolve_kernel(name: str):
    try:
        return KERNELS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown kernel {name!r}. Supported kernels: {sorted(KERNELS)}") from exc
