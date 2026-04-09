"""
Unified OBS-target estimator with pluggable selection correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from obs.models import LinearBiasModel, LinearTreatmentEffectModel
from obs.plugins import SelectionCorrectionPlugin


def _safe_clip_prob(p: float, lower: float = 0.05, upper: float = 0.95) -> float:
    return float(np.clip(p, lower, upper))


def _compute_rct_pseudo_effect(
    df_rct: pd.DataFrame,
    x_cols: Sequence[str],
    a_col: str,
    y_col: str,
) -> np.ndarray:
    """
    Construct experimental grounding signal using a linear AIPW-style pseudo effect.
    """
    X = df_rct[list(x_cols)].to_numpy(dtype=float)
    A = df_rct[a_col].to_numpy(dtype=float)
    Y = df_rct[y_col].to_numpy(dtype=float)

    treated_idx = A == 1
    control_idx = A == 0
    if treated_idx.sum() == 0 or control_idx.sum() == 0:
        raise ValueError("RCT data must contain both treated and control samples.")

    model_t = LinearRegression()
    model_c = LinearRegression()
    model_t.fit(X[treated_idx], Y[treated_idx])
    model_c.fit(X[control_idx], Y[control_idx])

    mu1 = model_t.predict(X)
    mu0 = model_c.predict(X)
    p = _safe_clip_prob(np.mean(A))
    pseudo = mu1 - mu0 + A * (Y - mu1) / p - (1.0 - A) * (Y - mu0) / (1.0 - p)
    return pseudo


@dataclass
class ObsTargetBaseEstimator:
    """
    Main estimator for obs-target CI:
      tau_hat(x) = w_hat(x) + eta_hat(x)
    """

    plugin: Optional[SelectionCorrectionPlugin] = None
    model_type: str = "linear"
    random_state: int = 2024

    def __post_init__(self):
        if self.model_type != "linear":
            raise ValueError("Currently only model_type='linear' is supported.")
        self.w_model_ = LinearTreatmentEffectModel(fit_interactions=True)
        self.eta_model_ = LinearBiasModel()
        if self.plugin is None:
            self.plugin = SelectionCorrectionPlugin(name="base")
        self.fitted_ = False
        self.summary_: Dict[str, object] = {}

    def fit(
        self,
        df_rct: pd.DataFrame,
        df_obs: pd.DataFrame,
        x_cols: Sequence[str],
        a_col: str = "A",
        y_col: str = "Y",
        g_col: str = "G",
    ):
        X_obs = df_obs[list(x_cols)].to_numpy(dtype=float)
        A_obs = df_obs[a_col].to_numpy(dtype=float)
        Y_obs = df_obs[y_col].to_numpy(dtype=float)
        self.w_model_.fit(X_obs, A_obs, Y_obs)

        X_rct = df_rct[list(x_cols)].to_numpy(dtype=float)
        w_hat_rct = self.w_model_.predict_tau(X_rct)
        pseudo_effect_rct = _compute_rct_pseudo_effect(df_rct, x_cols=x_cols, a_col=a_col, y_col=y_col)
        default_bias_target = pseudo_effect_rct - w_hat_rct

        self.plugin.fit(df_rct=df_rct, df_obs=df_obs, x_cols=x_cols, a_col=a_col, y_col=y_col, g_col=g_col)
        weights = self.plugin.get_rct_weights(df_rct)
        corrected_target = self.plugin.get_corrected_bias_target(df_rct, base_w_hat=w_hat_rct)
        if corrected_target is None:
            corrected_target = default_bias_target
        corrected_target = np.asarray(corrected_target, dtype=float).reshape(-1)
        if corrected_target.shape[0] != df_rct.shape[0]:
            raise ValueError("Plugin returned corrected bias target with invalid shape.")

        self.eta_model_.fit(X_rct, corrected_target, sample_weight=weights)
        self.fitted_ = True
        self.x_cols_ = list(x_cols)
        self.a_col_ = a_col
        self.y_col_ = y_col
        self.g_col_ = g_col
        self.rct_size_ = int(df_rct.shape[0])
        self.obs_size_ = int(df_obs.shape[0])
        self.eta_train_size_ = int(corrected_target.shape[0])
        self.bias_target_mean_ = float(np.mean(corrected_target))
        self.bias_target_std_ = float(np.std(corrected_target))

        self.summary_ = {
            "model_type": self.model_type,
            "plugin": self.plugin.name,
            "rct_size": self.rct_size_,
            "obs_size": self.obs_size_,
            "eta_train_size": self.eta_train_size_,
            "bias_target_mean": self.bias_target_mean_,
            "bias_target_std": self.bias_target_std_,
            "w_model": self.w_model_.summary(),
            "eta_model": self.eta_model_.summary(),
            "plugin_summary": self.plugin.summary(),
        }
        return self

    def predict_tau(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("ObsTargetBaseEstimator is not fitted.")
        X = np.asarray(X, dtype=float)
        return self.w_model_.predict_tau(X) + self.eta_model_.predict(X)

    def estimate_ate(self, X: np.ndarray) -> float:
        return float(np.mean(self.predict_tau(X)))

    def summary(self) -> Dict[str, object]:
        if not self.fitted_:
            return {"fitted": False, "plugin": self.plugin.name}
        return {"fitted": True, **self.summary_}

