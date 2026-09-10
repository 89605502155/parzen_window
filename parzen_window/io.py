"""Persisting model weights to/from ``.npz`` files for backups and transfer."""

from __future__ import annotations

import numpy as np


def save_npz(
    path, *, h, kernel, adaptive_bandwidth, bandwidth_neighbors, X_train, y_train, classes
) -> None:
    np.savez(
        path,
        h=np.asarray(h),
        kernel=np.asarray(kernel),
        adaptive_bandwidth=np.asarray(adaptive_bandwidth),
        bandwidth_neighbors=np.asarray(bandwidth_neighbors),
        X_train=X_train,
        y_train=y_train,
        classes=classes,
    )


def load_npz(path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            "h": float(data["h"]),
            "kernel": str(data["kernel"]),
            "adaptive_bandwidth": bool(data["adaptive_bandwidth"]),
            "bandwidth_neighbors": int(data["bandwidth_neighbors"]),
            "X_train": data["X_train"],
            "y_train": data["y_train"],
            "classes": data["classes"],
        }
