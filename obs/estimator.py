"""
Unified OBS-target estimators with pluggable selection correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from obs.models import LinearBiasModel, LinearTreatmentEffectModel
from obs.plugins import SelectionCorrectionPlugin


_INTEGRATIVE_SHADOW_ERROR = (
    "For integrative and integrative_rlearner, use plugin='shadow_source_ep' because these methods require "
    "extended participation weights to estimate bG."
)


def _to_1d_float_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if np.any(~np.isfinite(arr)):
        raise ValueError("Expected finite 1D float array.")
    return arr


def _basis_with_intercept(X) -> np.ndarray:
    x_arr = np.asarray(X, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(1, -1)
    if x_arr.ndim != 2:
        raise ValueError(f"Expected 2D X, got shape={x_arr.shape}.")
    if np.any(~np.isfinite(x_arr)):
        raise ValueError("Expected finite X matrix.")
    return np.hstack([np.ones((x_arr.shape[0], 1), dtype=float), x_arr])


def _safe_clip_prob(p: float, lower: float = 0.05, upper: float = 0.95) -> float:
    return float(np.clip(p, lower, upper))


def _safe_clip_prob_array(p, lower: float = 0.05, upper: float = 0.95) -> np.ndarray:
    arr = np.asarray(p, dtype=float).reshape(-1)
    if np.any(~np.isfinite(arr)):
        raise ValueError("Probability array contains non-finite values.")
    return np.clip(arr, lower, upper)


def _fit_mu_model(Z: np.ndarray, Y: np.ndarray) -> LinearRegression:
    z_arr = np.asarray(Z, dtype=float)
    y_arr = _to_1d_float_array(Y)
    if z_arr.ndim != 2:
        raise ValueError(f"Expected 2D nuisance design Z, got shape={z_arr.shape}.")
    if z_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("mu model Z/Y row mismatch.")
    if np.any(~np.isfinite(z_arr)):
        raise ValueError("mu model Z contains non-finite values.")
    model = LinearRegression()
    model.fit(z_arr, y_arr)
    return model


def _fit_e_model(Z: np.ndarray, T: np.ndarray) -> LogisticRegression:
    z_arr = np.asarray(Z, dtype=float)
    t_arr = _to_1d_float_array(T)
    if z_arr.ndim != 2:
        raise ValueError(f"Expected 2D nuisance design Z, got shape={z_arr.shape}.")
    if z_arr.shape[0] != t_arr.shape[0]:
        raise ValueError("e model Z/T row mismatch.")
    if np.any(~np.isfinite(z_arr)):
        raise ValueError("e model Z contains non-finite values.")
    if np.unique(t_arr).size < 2:
        raise ValueError("Cannot fit e(X,G): treatment has fewer than 2 classes.")
    model = LogisticRegression(max_iter=2000)
    model.fit(z_arr, t_arr)
    return model


def _solve_linear_system(
    design,
    target,
    sample_weight: Optional[np.ndarray] = None,
    ridge_diag: Optional[np.ndarray] = None,
) -> np.ndarray:
    d_arr = np.asarray(design, dtype=float)
    y_arr = _to_1d_float_array(target)
    if d_arr.ndim != 2:
        raise ValueError(f"Expected 2D design, got shape={d_arr.shape}.")
    if d_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("design/target row mismatch.")
    if np.any(~np.isfinite(d_arr)):
        raise ValueError("design contains non-finite values.")

    if sample_weight is not None:
        w = _to_1d_float_array(sample_weight)
        if w.shape[0] != d_arr.shape[0]:
            raise ValueError("sample_weight length mismatch.")
        if np.any(w < 0):
            raise ValueError("sample_weight must be non-negative.")
        sqrt_w = np.sqrt(w)
        d_arr = d_arr * sqrt_w[:, None]
        y_arr = y_arr * sqrt_w

    if ridge_diag is None:
        coef = np.linalg.lstsq(d_arr, y_arr, rcond=None)[0]
    else:
        r_arr = _to_1d_float_array(ridge_diag)
        if r_arr.shape[0] != d_arr.shape[1]:
            raise ValueError("ridge_diag length mismatch.")
        normal = d_arr.T @ d_arr + np.diag(r_arr)
        rhs = d_arr.T @ y_arr
        try:
            coef = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(normal, rhs, rcond=None)[0]

    coef = _to_1d_float_array(coef)
    if coef.shape[0] != d_arr.shape[1]:
        raise ValueError("Solved coefficient shape mismatch.")
    return coef


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
    return _to_1d_float_array(pseudo)


def _resolve_iv_effect_x_cols(x_cols, plugin_summary):
    x_cols = list(x_cols)
    summary = dict(plugin_summary or {})
    selected_iv_cols = list(summary.get("selected_iv_cols") or [])
    xz_cols = list(summary.get("Xz_cols") or [])
    xc_cols = list(summary.get("Xc_cols") or [])

    plugin_name = str(summary.get("plugin", "")).lower()
    is_iv_plugin = plugin_name in {"selection_iv", "iv"} or (bool(selected_iv_cols) and bool(xz_cols))
    if not is_iv_plugin:
        effect_x_cols = list(x_cols)
        excluded_iv_cols = []
    elif xc_cols and set(xc_cols).issubset(set(x_cols)):
        effect_x_cols = [c for c in x_cols if c in set(xc_cols)]
        excluded_iv_cols = [c for c in x_cols if c not in set(effect_x_cols)]
    else:
        excluded_source = selected_iv_cols or xz_cols
        excluded_iv_cols = [c for c in x_cols if c in set(excluded_source)]
        effect_x_cols = [c for c in x_cols if c not in set(excluded_iv_cols)]

    if not effect_x_cols:
        raise ValueError("No effect covariates left after removing selection IV variables.")
    return effect_x_cols, excluded_iv_cols


@dataclass
class _SelectionAnchor:
    """
    Estimate raw RCT signal, transported OBS-target anchor, and RCT selection bias bG.
    """

    def __post_init__(self):
        self.tau_r_model_ = LinearBiasModel()
        self.tau0_anchor_treated_model_ = LinearRegression()
        self.tau0_anchor_control_model_ = LinearRegression()
        self.bG_model_ = LinearBiasModel()
        self.fitted_ = False
        self.summary_: Dict[str, object] = {}

    def fit(
        self,
        df_rct: pd.DataFrame,
        effect_x_cols: Sequence[str],
        plugin: SelectionCorrectionPlugin,
        a_col: str,
        y_col: str,
    ):
        effect_x_cols = list(effect_x_cols)
        X_rct = df_rct[effect_x_cols].to_numpy(dtype=float)
        pseudo_rct = _compute_rct_pseudo_effect(df_rct, x_cols=effect_x_cols, a_col=a_col, y_col=y_col)

        self.tau_r_model_.fit(X_rct, pseudo_rct)
        weights = plugin.get_rct_weights(df_rct)
        if weights is None:
            weights = np.ones(df_rct.shape[0], dtype=float)
        weights = _to_1d_float_array(weights)
        if weights.shape[0] != df_rct.shape[0]:
            raise ValueError("Plugin returned RCT weights with invalid shape.")

        T = df_rct[a_col].to_numpy(dtype=float)
        Y = df_rct[y_col].to_numpy(dtype=float)
        treated_mask = T == 1
        control_mask = T == 0
        if treated_mask.sum() == 0 or control_mask.sum() == 0:
            raise ValueError("RCT data must contain both treated and control samples.")

        self.tau0_anchor_treated_model_.fit(
            X_rct[treated_mask],
            Y[treated_mask],
            sample_weight=weights[treated_mask],
        )
        self.tau0_anchor_control_model_.fit(
            X_rct[control_mask],
            Y[control_mask],
            sample_weight=weights[control_mask],
        )

        bG_target = self.tau_r_model_.predict(X_rct) - self.predict_tau0_anchor(X_rct)
        self.bG_model_.fit(X_rct, bG_target)

        tau_r_hat = self.tau_r_model_.predict(X_rct)
        tau0_anchor_hat = self.predict_tau0_anchor(X_rct)
        bG_hat = self.bG_model_.predict(X_rct)
        self.fitted_ = True
        self.summary_ = {
            "anchor_estimator": "armwise_weighted_outcome_regression",
            "effect_x_cols": effect_x_cols,
            "rct_weight_min": float(np.min(weights)),
            "rct_weight_mean": float(np.mean(weights)),
            "rct_weight_max": float(np.max(weights)),
            "rct_weight_std": float(np.std(weights)),
            "tau_r_mean": float(np.mean(tau_r_hat)),
            "tau0_anchor_mean": float(np.mean(tau0_anchor_hat)),
            "bG_mean": float(np.mean(bG_hat)),
            "bG_std": float(np.std(bG_hat)),
            "tau_r_model": self.tau_r_model_.summary(),
            "tau0_anchor_treated_model": {
                "fitted": True,
                "intercept": float(self.tau0_anchor_treated_model_.intercept_),
                "coef": self.tau0_anchor_treated_model_.coef_.reshape(-1).tolist(),
            },
            "tau0_anchor_control_model": {
                "fitted": True,
                "intercept": float(self.tau0_anchor_control_model_.intercept_),
                "coef": self.tau0_anchor_control_model_.coef_.reshape(-1).tolist(),
            },
            "bG_model": self.bG_model_.summary(),
        }
        return self

    def predict_tau_r(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("_SelectionAnchor is not fitted.")
        return self.tau_r_model_.predict(X)

    def predict_tau0_anchor(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self.tau0_anchor_treated_model_, "coef_") or not hasattr(self.tau0_anchor_control_model_, "coef_"):
            raise RuntimeError("_SelectionAnchor is not fitted.")
        return self.tau0_anchor_treated_model_.predict(X) - self.tau0_anchor_control_model_.predict(X)

    def predict_bG(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("_SelectionAnchor is not fitted.")
        return self.bG_model_.predict(X)

    def summary(self) -> Dict[str, object]:
        if not self.fitted_:
            return {"fitted": False}
        return {"fitted": True, **self.summary_}


@dataclass
class RHCObsEstimator:
    """
    RHC estimator for obs-target CI:
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
        corrected_target = _to_1d_float_array(corrected_target)
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
            raise RuntimeError("RHCObsEstimator is not fitted.")
        X = np.asarray(X, dtype=float)
        return self.w_model_.predict_tau(X) + self.eta_model_.predict(X)

    def estimate_ate(self, X: np.ndarray) -> float:
        return float(np.mean(self.predict_tau(X)))

    def summary(self) -> Dict[str, object]:
        if not self.fitted_:
            return {"fitted": False, "plugin": self.plugin.name}
        return {"fitted": True, **self.summary_}


@dataclass
class IntegrativeObsEstimator:
    """
    Modified integrative OBS-target estimator.

    The final target effect is tau0(X). bG is estimated from extended participation
    weights and used only as a residual offset; bT is the OBS treatment-confounding bias.
    """

    plugin: Optional[SelectionCorrectionPlugin] = None
    model_type: str = "linear"
    random_state: int = 2024

    def __post_init__(self):
        if self.model_type != "linear":
            raise ValueError("Currently only model_type='linear' is supported.")
        if self.plugin is None:
            self.plugin = SelectionCorrectionPlugin(name="base")
        self.fitted_ = False
        self.summary_: Dict[str, object] = {}

    def _validate_plugin(self):
        plugin_name = str(getattr(self.plugin, "name", "")).lower()
        if plugin_name == "shadow" or self.plugin.__class__.__name__ == "ShadowPlugin":
            raise ValueError(_INTEGRATIVE_SHADOW_ERROR)

    def _ridge_diag(self, p: int) -> Optional[np.ndarray]:
        _ = p
        return None

    def _regularization_summary(self) -> Dict[str, object]:
        return {}

    def fit(
        self,
        df_rct: pd.DataFrame,
        df_obs: pd.DataFrame,
        x_cols: Sequence[str],
        a_col: str = "A",
        y_col: str = "Y",
        g_col: str = "G",
    ):
        self._validate_plugin()
        x_cols = list(x_cols)

        df_all = pd.concat([df_rct, df_obs], ignore_index=True)
        T_all = _to_1d_float_array(df_all[a_col])
        Y_all = _to_1d_float_array(df_all[y_col])
        if g_col in df_all.columns:
            G_all = _to_1d_float_array(df_all[g_col])
        else:
            G_all = np.concatenate(
                [np.ones(df_rct.shape[0], dtype=float), np.zeros(df_obs.shape[0], dtype=float)],
                axis=0,
            )

        self.plugin.fit(df_rct=df_rct, df_obs=df_obs, x_cols=x_cols, a_col=a_col, y_col=y_col, g_col=g_col)
        plugin_summary = self.plugin.summary()
        effect_x_cols, excluded_iv_cols = _resolve_iv_effect_x_cols(x_cols, plugin_summary)
        X_effect_all = df_all[effect_x_cols].to_numpy(dtype=float)
        if (
            X_effect_all.shape[0] != T_all.shape[0]
            or X_effect_all.shape[0] != Y_all.shape[0]
            or X_effect_all.shape[0] != G_all.shape[0]
        ):
            raise ValueError("Combined OBS/RCT arrays have inconsistent row counts.")

        self.x_cols_ = list(x_cols)
        self.effect_x_cols_ = list(effect_x_cols)
        self.excluded_iv_cols_ = list(excluded_iv_cols)
        self.effect_col_indices_ = [self.x_cols_.index(c) for c in self.effect_x_cols_]
        self.a_col_ = a_col
        self.y_col_ = y_col
        self.g_col_ = g_col

        self.anchor_ = _SelectionAnchor()
        self.anchor_.fit(df_rct=df_rct, effect_x_cols=effect_x_cols, plugin=self.plugin, a_col=a_col, y_col=y_col)

        Z_all = np.hstack([X_effect_all, G_all.reshape(-1, 1)])
        self.mu_model_ = _fit_mu_model(Z_all, Y_all)
        self.e_model_ = _fit_e_model(Z_all, T_all)
        mu_hat = _to_1d_float_array(self.mu_model_.predict(Z_all))
        e_hat = _safe_clip_prob_array(self.e_model_.predict_proba(Z_all)[:, 1], lower=0.05, upper=0.95)

        r_y = Y_all - mu_hat
        r_t = T_all - e_hat
        bG_hat = _to_1d_float_array(self.anchor_.predict_bG(X_effect_all))
        target = r_y - r_t * G_all * bG_hat

        Phi = _basis_with_intercept(X_effect_all)
        design_tau = r_t[:, None] * Phi
        design_bT = (r_t * (1.0 - G_all))[:, None] * Phi
        design = np.hstack([design_tau, design_bT])
        p = Phi.shape[1]

        coef = _solve_linear_system(design, target, ridge_diag=self._ridge_diag(p))
        self.tau_coef_ = coef[:p]
        self.bT_coef_ = coef[p:]
        self.fitted_ = True

        X_rct = df_rct[effect_x_cols].to_numpy(dtype=float)
        X_obs = df_obs[effect_x_cols].to_numpy(dtype=float)
        tau0_obs = self.predict_tau(X_obs)
        bG_obs = self.predict_bG(X_obs)
        bT_obs = self.predict_bT(X_obs)
        tau_r_rct = self.predict_tau_r(X_rct)
        tau0_anchor_rct = self.anchor_.predict_tau0_anchor(X_rct)
        obs_biased_tau_obs = self.predict_obs_biased_tau(X_obs)

        self.rct_size_ = int(df_rct.shape[0])
        self.obs_size_ = int(df_obs.shape[0])
        self.n_features_in_ = int(X_effect_all.shape[1])
        self.summary_ = {
            "model_type": self.model_type,
            "plugin": self.plugin.name,
            "rct_size": self.rct_size_,
            "obs_size": self.obs_size_,
            "original_x_cols": self.x_cols_,
            "effect_x_cols": self.effect_x_cols_,
            "excluded_iv_cols": self.excluded_iv_cols_,
            "tau0_mean": float(np.mean(tau0_obs)),
            "bG_mean": float(np.mean(bG_obs)),
            "bT_mean": float(np.mean(bT_obs)),
            "raw_rct_tau_mean": float(np.mean(tau_r_rct)),
            "tau0_anchor_mean": float(np.mean(tau0_anchor_rct)),
            "obs_biased_tau_mean": float(np.mean(obs_biased_tau_obs)),
            "nuisance": {
                "mu_intercept": float(self.mu_model_.intercept_),
                "mu_coef": self.mu_model_.coef_.reshape(-1).tolist(),
                "e_intercept": self.e_model_.intercept_.reshape(-1).tolist(),
                "e_coef": self.e_model_.coef_.reshape(-1).tolist(),
                "e_hat_min": float(np.min(e_hat)),
                "e_hat_mean": float(np.mean(e_hat)),
                "e_hat_max": float(np.max(e_hat)),
            },
            "anchor_summary": self.anchor_.summary(),
            "plugin_summary": plugin_summary,
            **self._regularization_summary(),
        }
        return self

    def _predict_linear(self, X: np.ndarray, coef: np.ndarray) -> np.ndarray:
        return _basis_with_intercept(X) @ _to_1d_float_array(coef)

    def _extract_effect_X(self, X) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("IntegrativeObsEstimator is not fitted.")
        if isinstance(X, pd.DataFrame):
            return X[self.effect_x_cols_].to_numpy(dtype=float)

        x_arr = np.asarray(X, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        if x_arr.ndim != 2:
            raise ValueError(f"Expected 2D X, got shape={x_arr.shape}.")
        if x_arr.shape[1] == len(self.effect_x_cols_):
            return x_arr
        if x_arr.shape[1] == len(self.x_cols_):
            return x_arr[:, self.effect_col_indices_]
        raise ValueError(
            "X has incompatible number of columns: "
            f"got {x_arr.shape[1]}, expected {len(self.effect_x_cols_)} effect columns "
            f"or {len(self.x_cols_)} original columns."
        )

    def predict_tau(self, X) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("IntegrativeObsEstimator is not fitted.")
        X_effect = self._extract_effect_X(X)
        return self._predict_linear(X_effect, self.tau_coef_)

    def predict_bT(self, X) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("IntegrativeObsEstimator is not fitted.")
        X_effect = self._extract_effect_X(X)
        return self._predict_linear(X_effect, self.bT_coef_)

    def predict_bG(self, X) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("IntegrativeObsEstimator is not fitted.")
        X_effect = self._extract_effect_X(X)
        return self.anchor_.predict_bG(X_effect)

    def predict_tau_r(self, X) -> np.ndarray:
        return self.predict_tau(X) + self.predict_bG(X)

    def predict_obs_biased_tau(self, X) -> np.ndarray:
        return self.predict_tau(X) + self.predict_bT(X)

    def estimate_ate(self, X) -> float:
        return float(np.mean(self.predict_tau(X)))

    def summary(self) -> Dict[str, object]:
        if not self.fitted_:
            return {"fitted": False, "plugin": self.plugin.name}
        return {"fitted": True, **self.summary_}


@dataclass
class IntegrativeRLearnerObsEstimator(IntegrativeObsEstimator):
    """
    Modified integrative R-learner with separate ridge penalties for tau0 and bT.
    """

    alpha_tau: float = 1e-4
    alpha_bT: float = 1e-2

    def _ridge_diag(self, p: int) -> Optional[np.ndarray]:
        ridge_diag = np.concatenate(
            [
                np.repeat(float(self.alpha_tau), p),
                np.repeat(float(self.alpha_bT), p),
            ],
            axis=0,
        )
        ridge_diag[0] = 0.0
        ridge_diag[p] = 0.0
        return ridge_diag

    def _regularization_summary(self) -> Dict[str, object]:
        return {
            "alpha_tau": float(self.alpha_tau),
            "alpha_bT": float(self.alpha_bT),
            "regularization": "separate_ridge",
        }


ObsTargetBaseEstimator = RHCObsEstimator
