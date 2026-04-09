"""Utility helpers for robust sample-weight processing."""

from __future__ import annotations

from typing import Dict

import numpy as np


def ensure_1d_float(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def effective_sample_size(weights: np.ndarray) -> float:
    w = ensure_1d_float(weights)
    denom = float(np.sum(w**2))
    if denom <= 0:
        return 0.0
    return float((np.sum(w) ** 2) / denom)


def weight_summary(weights: np.ndarray) -> Dict[str, float]:
    w = ensure_1d_float(weights)
    return {
        "min": float(np.min(w)),
        "max": float(np.max(w)),
        "mean": float(np.mean(w)),
        "std": float(np.std(w)),
        "ess": float(effective_sample_size(w)),
    }


def normalize_weights(weights: np.ndarray, target_mean: float = 1.0, eps: float = 1e-12) -> np.ndarray:
    """Normalize positive weights to a target mean (default 1)."""
    w = ensure_1d_float(weights)
    if np.any(~np.isfinite(w)):
        raise ValueError("Weight normalization failed: non-finite values found.")
    w = np.maximum(w, eps)
    current_mean = float(np.mean(w))
    if current_mean <= 0:
        raise ValueError("Weight normalization failed: non-positive mean.")
    return w * (target_mean / current_mean)


def finalize_weights(
    raw_weights: np.ndarray,
    clip_min: float = 0.05,
    clip_max: float = 20.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Common finite-check, clipping and normalization routine for plugin weights.
    """
    if clip_min <= 0:
        raise ValueError("clip_min must be > 0.")
    if clip_max <= clip_min:
        raise ValueError("clip_max must be > clip_min.")

    w = ensure_1d_float(raw_weights)
    if np.any(~np.isfinite(w)):
        raise ValueError("Raw weights contain NaN/Inf.")

    w = np.clip(w, clip_min, clip_max)
    if np.any(~np.isfinite(w)):
        raise ValueError("Clipped weights contain NaN/Inf.")

    if normalize:
        w = normalize_weights(w)
    return w
