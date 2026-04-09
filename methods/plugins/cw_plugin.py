"""Calibration-weighting plugin for OBS-target eta-model correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from methods.plugins.base import SelectionCorrectionPlugin
from utils.weight_utils import finalize_weights, weight_summary


def _build_balance_features(X: np.ndarray, degree: int = 1, include_interactions: bool = False) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if degree < 1:
        raise ValueError("degree must be >= 1")

    feats = [X]
    if degree >= 2:
        feats.append(X**2)
    if include_interactions:
        p = X.shape[1]
        inter_cols = []
        for i in range(p):
            for j in range(i + 1, p):
                inter_cols.append((X[:, i] * X[:, j]).reshape(-1, 1))
        if inter_cols:
            feats.append(np.hstack(inter_cols))
    return np.hstack(feats)


def _compute_calibration_weights(
    X_rct: np.ndarray,
    X_obs: np.ndarray,
    degree: int,
    include_interactions: bool,
    max_iter: int,
) -> Dict[str, object]:
    F_rct = _build_balance_features(X_rct, degree=degree, include_interactions=include_interactions)
    F_obs = _build_balance_features(X_obs, degree=degree, include_interactions=include_interactions)

    F_rct_aug = np.hstack([np.ones((F_rct.shape[0], 1)), F_rct])
    target = np.mean(np.hstack([np.ones((F_obs.shape[0], 1)), F_obs]), axis=0)

    def objective(lam: np.ndarray) -> float:
        linear = F_rct_aug @ lam
        return logsumexp(linear) - np.log(F_rct_aug.shape[0]) - target @ lam

    def grad(lam: np.ndarray) -> np.ndarray:
        linear = F_rct_aug @ lam
        scaled = np.exp(linear - logsumexp(linear)).reshape(-1, 1)
        return np.sum(F_rct_aug * scaled, axis=0) - target

    result = minimize(
        objective,
        x0=np.zeros(F_rct_aug.shape[1], dtype=float),
        jac=grad,
        method="BFGS",
        options={"maxiter": max_iter},
    )
    if not result.success:
        raise RuntimeError(f"Calibration weighting optimization failed: {result.message}")

    linear = F_rct_aug @ result.x
    raw_weights = np.exp(linear - logsumexp(linear)) * F_rct_aug.shape[0]
    weights = finalize_weights(raw_weights, clip_min=1e-6, clip_max=1e6)

    before_diff = np.mean(F_rct, axis=0) - np.mean(F_obs, axis=0)
    after_diff = np.average(F_rct, axis=0, weights=weights) - np.mean(F_obs, axis=0)

    return {
        "weights": weights,
        "balance_before_abs_mean_diff": float(np.mean(np.abs(before_diff))),
        "balance_after_abs_mean_diff": float(np.mean(np.abs(after_diff))),
        "balance_before_max_abs_diff": float(np.max(np.abs(before_diff))),
        "balance_after_max_abs_diff": float(np.max(np.abs(after_diff))),
    }


@dataclass
class CWPlugin(SelectionCorrectionPlugin):
    name: str = "cw"
    degree: int = 1
    include_interactions: bool = False
    max_iter: int = 2000

    def fit(self, df_rct: pd.DataFrame, df_obs: pd.DataFrame, x_cols, a_col, y_col, g_col):
        _ = a_col, y_col, g_col
        result = _compute_calibration_weights(
            X_rct=df_rct[list(x_cols)].to_numpy(dtype=float),
            X_obs=df_obs[list(x_cols)].to_numpy(dtype=float),
            degree=self.degree,
            include_interactions=self.include_interactions,
            max_iter=self.max_iter,
        )
        self.weights_ = result["weights"]
        self.fitted_ = True
        self.diagnostics_ = {
            "plugin": self.name,
            "degree": self.degree,
            "include_interactions": self.include_interactions,
            "weight_summary": weight_summary(self.weights_),
            "balance_before_abs_mean_diff": result["balance_before_abs_mean_diff"],
            "balance_after_abs_mean_diff": result["balance_after_abs_mean_diff"],
            "balance_before_max_abs_diff": result["balance_before_max_abs_diff"],
            "balance_after_max_abs_diff": result["balance_after_max_abs_diff"],
        }
        return self

    def get_rct_weights(self, df_rct: pd.DataFrame):
        _ = df_rct
        return self.weights_
