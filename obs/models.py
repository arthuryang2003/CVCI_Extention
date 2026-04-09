"""
Model components shared by the OBS-target estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.linear_model import LinearRegression


def _to_1d_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


@dataclass
class LinearTreatmentEffectModel:
    """
    Linear treatment effect model for w(x).

    Outcome model:
        E[Y | X, A] = b0 + bA * A + bX^T X + bAX^T (A * X)
    Then:
        w_hat(x) = bA + bAX^T x
    """

    fit_interactions: bool = True

    def __post_init__(self):
        self.model_ = LinearRegression()
        self.n_features_in_: Optional[int] = None
        self.coef_a_: Optional[float] = None
        self.coef_ax_: Optional[np.ndarray] = None
        self.coef_x_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None

    def _build_design(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        A = _to_1d_float_array(A).reshape(-1, 1)
        if self.fit_interactions:
            return np.hstack([A, X, A * X])
        return np.hstack([A, X])

    def fit(self, X: np.ndarray, A: np.ndarray, Y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        X = np.asarray(X, dtype=float)
        A = _to_1d_float_array(A)
        Y = _to_1d_float_array(Y)
        design = self._build_design(X, A)

        self.model_.fit(design, Y, sample_weight=sample_weight)
        coef = self.model_.coef_.reshape(-1)
        self.intercept_ = float(self.model_.intercept_)
        self.n_features_in_ = X.shape[1]
        self.coef_a_ = float(coef[0])
        self.coef_x_ = coef[1 : 1 + X.shape[1]]
        if self.fit_interactions:
            self.coef_ax_ = coef[1 + X.shape[1] :]
        else:
            self.coef_ax_ = np.zeros(X.shape[1], dtype=float)
        return self

    def predict_tau(self, X: np.ndarray) -> np.ndarray:
        if self.coef_a_ is None or self.coef_ax_ is None:
            raise RuntimeError("LinearTreatmentEffectModel is not fitted.")
        X = np.asarray(X, dtype=float)
        return self.coef_a_ + X @ self.coef_ax_

    def summary(self) -> Dict[str, object]:
        if self.coef_a_ is None:
            return {"fitted": False}
        return {
            "fitted": True,
            "intercept": self.intercept_,
            "coef_a": self.coef_a_,
            "coef_x": self.coef_x_.tolist() if self.coef_x_ is not None else None,
            "coef_ax": self.coef_ax_.tolist() if self.coef_ax_ is not None else None,
            "fit_interactions": self.fit_interactions,
        }


@dataclass
class LinearBiasModel:
    """
    Linear model for bias function eta(x).
    """

    def __post_init__(self):
        self.model_ = LinearRegression()
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None

    def fit(self, X: np.ndarray, target: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        X = np.asarray(X, dtype=float)
        target = _to_1d_float_array(target)
        self.model_.fit(X, target, sample_weight=sample_weight)
        self.coef_ = self.model_.coef_.reshape(-1)
        self.intercept_ = float(self.model_.intercept_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("LinearBiasModel is not fitted.")
        X = np.asarray(X, dtype=float)
        return self.model_.predict(X)

    def summary(self) -> Dict[str, object]:
        if self.coef_ is None:
            return {"fitted": False}
        return {
            "fitted": True,
            "intercept": self.intercept_,
            "coef": self.coef_.tolist(),
        }
