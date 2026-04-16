"""Unified estimator entrypoint for RCT-target CVCI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from rct.cv import cross_validation
from rct.plugins import ObsPluginOutput, build_obs_plugin


@dataclass
class RCTTargetBaseEstimator:
    """Unified RCT-target estimator for base and plugin-adjusted CVCI."""

    plugin_method: str = "none"
    mode: str = "linear"
    exp_model: str = "response_func"
    lambda_bin: int = 5
    k_fold: int = 5
    stratified_kfold: bool = True
    random_state: int = 2024
    plugin_config: Optional[Dict[str, object]] = None

    def __post_init__(self):
        if self.mode not in {"mean", "linear"}:
            raise ValueError("mode must be either 'mean' or 'linear'.")
        self.plugin_method = str(self.plugin_method).lower()
        self.plugin_config = dict(self.plugin_config or {})
        self.fitted_ = False
        self.summary_: Dict[str, object] = {}

    def _default_lambda_grid(self) -> np.ndarray:
        if self.lambda_bin <= 1:
            return np.array([0.0], dtype=float)
        return np.linspace(0.0, 1.0, int(self.lambda_bin))

    def fit(
        self,
        exp_data: np.ndarray,
        obs_data: np.ndarray,
        d_exp: Optional[int] = None,
        d_obs: Optional[int] = None,
        lambda_vals: Optional[Sequence[float]] = None,
        plugin_output: Optional[ObsPluginOutput] = None,
        covariate_names: Optional[Sequence[str]] = None,
    ):
        exp_data = np.asarray(exp_data, dtype=float)
        obs_data = np.asarray(obs_data, dtype=float)

        if self.mode == "linear" and (d_exp is None or d_obs is None):
            raise ValueError("d_exp and d_obs are required in linear mode.")
        if self.mode == "mean":
            d_exp = d_exp
            d_obs = d_obs

        if lambda_vals is None:
            lambda_vals = self._default_lambda_grid()
        lambda_vals = np.asarray(lambda_vals, dtype=float)

        config = {
            "mode": self.mode,
            "k_fold": self.k_fold,
            "d_exp": d_exp,
            "d_obs": d_obs,
            "exp_model": self.exp_model,
            "stratified_kfold": self.stratified_kfold,
            "random_state": self.random_state,
            "covariate_names": list(covariate_names) if covariate_names is not None else None,
            **self.plugin_config,
        }

        if plugin_output is None:
            plugin_output = build_obs_plugin(obs_data, exp_data, self.plugin_method, config=config)

        q_values, lambda_opt, theta_opt = cross_validation(
            exp_data,
            obs_data,
            lambda_vals,
            mode=self.mode,
            k_fold=self.k_fold,
            d_exp=d_exp,
            d_obs=d_obs,
            exp_model=self.exp_model,
            stratified_kfold=self.stratified_kfold,
            random_state=self.random_state,
            obs_weights=plugin_output.sample_weights,
            obs_outcomes=plugin_output.pseudo_outcome,
        )

        self.fitted_ = True
        self.exp_data_ = exp_data
        self.obs_data_ = obs_data
        self.d_exp_ = d_exp
        self.d_obs_ = d_obs
        self.lambda_vals_ = lambda_vals
        self.q_values_ = q_values
        self.lambda_opt_ = float(lambda_opt)
        self.theta_opt_ = theta_opt
        self.plugin_output_ = plugin_output
        self.estimate_ = float(theta_opt.beta().item() if hasattr(theta_opt.beta(), "item") else theta_opt.beta())

        self.summary_ = {
            "mode": self.mode,
            "exp_model": self.exp_model,
            "plugin_method": plugin_output.method,
            "lambda_opt": self.lambda_opt_,
            "estimate": self.estimate_,
            "q_values": q_values.tolist(),
            "obs_weight_summary": {
                "min": float(np.min(plugin_output.sample_weights)),
                "mean": float(np.mean(plugin_output.sample_weights)),
                "max": float(np.max(plugin_output.sample_weights)),
                "std": float(np.std(plugin_output.sample_weights)),
            },
            "plugin_metadata": plugin_output.metadata,
        }
        return self

    def predict_tau(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("RCTTargetBaseEstimator is not fitted.")
        if self.mode != "linear":
            raise ValueError("predict_tau is only supported in linear mode.")
        return self.theta_opt_.predict_tau(X)

    def estimate_ate(self, X: np.ndarray) -> float:
        if not self.fitted_:
            raise RuntimeError("RCTTargetBaseEstimator is not fitted.")
        if self.mode == "mean":
            return float(self.theta_opt_.beta(self.lambda_opt_, self.exp_data_, self.obs_data_))
        return float(np.mean(self.predict_tau(X)))

    def summary(self) -> Dict[str, object]:
        if not self.fitted_:
            return {"fitted": False, "plugin_method": self.plugin_method}
        return {"fitted": True, **self.summary_}
