"""Heuristic candidate-screening helpers for selection correction plugins."""

from __future__ import annotations

from typing import Optional

import numpy as np


def residualize(target: np.ndarray, design: Optional[np.ndarray]) -> np.ndarray:
    y = np.asarray(target, dtype=float).reshape(-1)
    if design is None:
        return y - np.mean(y)
    X = np.asarray(design, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] == 0:
        return y - np.mean(y)
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
    coef, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    return y - X_aug @ coef


def safe_abs_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= eps or y_std <= eps:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(abs(corr))


def partial_abs_corr(candidate: np.ndarray, target: np.ndarray, nuisance: Optional[np.ndarray]) -> float:
    rc = residualize(candidate, nuisance)
    rt = residualize(target, nuisance)
    return safe_abs_corr(rc, rt)
