"""Model definitions for RCT-target CVCI."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold

random_seed = 2024
np.random.seed(random_seed)


def _compute_exp_minimizer_from_experimental_data(
    X: np.ndarray,
    mode: str = "linear",
    exp_model: str = "aipw",
    stratified_kfold: bool = False,
    d_exp: int | None = None,
    rng=None,
    pi_func=None,
):
    """Compute experimental-only treatment effect estimate used by the linear model fit."""
    if mode == "mean":
        return None
    if mode != "linear":
        raise ValueError(f"Unsupported mode in _compute_exp_minimizer_from_experimental_data: {mode}")

    if d_exp is None:
        raise ValueError("please specify d_exp in _compute_exp_minimizer_from_experimental_data")

    Z_exp_all = X[:, :d_exp]
    A_exp_all = X[:, d_exp]
    Y_exp_all = X[:, -1]

    if exp_model == "aipw" and stratified_kfold:
        tv_split = StratifiedKFold(n_splits=2)
        numerator = tv_split.split(Z_exp_all, A_exp_all)
        for train_index, val_index in numerator:
            X_t, X_v = X[train_index], X[val_index]
            Z_t, A_t, Y_t = X_t[:, :d_exp], X_t[:, d_exp], X_t[:, -1]
            Z_v, A_v, Y_v = X_v[:, :d_exp], X_v[:, d_exp], X_v[:, -1]
            break
    else:
        n_val = int(0.5 * X.shape[0])
        Z_t, A_t, Y_t = Z_exp_all[n_val:, :], A_exp_all[n_val:], Y_exp_all[n_val:]
        Z_v, A_v, Y_v = Z_exp_all[:n_val, :], A_exp_all[:n_val], Y_exp_all[:n_val]

    if exp_model == "aipw":
        if pi_func is None:
            raise ValueError("pi_func must be provided when exp_model='aipw'.")
        exp_model_inst = LinearRegression()
        exp_model_inst.fit(np.concatenate((A_t.reshape(-1, 1), Z_t), axis=1), Y_t)
        mu_pred_1 = exp_model_inst.predict(np.concatenate((np.ones((np.shape(Z_v)[0], 1)), Z_v), axis=1))
        mu_pred_0 = exp_model_inst.predict(np.concatenate((np.zeros((np.shape(Z_v)[0], 1)), Z_v), axis=1))
        pi = pi_func(Z_v)
        psi = (A_v / pi) * (Y_v - mu_pred_1) + mu_pred_1 - (((1 - A_v) / (1 - pi)) * (Y_v - mu_pred_0) + mu_pred_0)
        return np.mean(psi)

    if exp_model == "mean_diff":
        return Y_exp_all[A_exp_all == 1].mean() - Y_exp_all[A_exp_all == 0].mean()

    if exp_model == "response_func":
        if np.array_equal(A_exp_all, np.ones_like(A_exp_all)) or np.array_equal(A_exp_all, np.zeros_like(A_exp_all)):
            return 0.0
        exp_model_inst = LinearRegression()
        exp_model_inst.fit(np.concatenate((A_exp_all.reshape(-1, 1), Z_exp_all), axis=1), Y_exp_all)
        return exp_model_inst.coef_[0]

    raise ValueError(f"Unsupported exp_model: {exp_model}")


class model_class(torch.nn.Module):
    """Class to define models. Uses torch for extensibility."""

    def __init__(self, mode="mean", d_exp=None, d_obs=None, exp_model="aipw", stratified_kfold=False, rng=None, fit_interactions=True):
        super(model_class, self).__init__()
        assert mode in ["mean", "linear"], f"mode must be valid, got: {mode}"
        self.mode = mode
        self.theta = None
        self.theta_model = None
        self.d_exp = d_exp
        self.d_obs = d_obs
        self.exp_model = exp_model
        self.stratified_kfold = stratified_kfold
        self.rng = rng
        self.fit_interactions = fit_interactions

        if self.mode == "linear":
            assert d_obs is not None, "number of covariates of obs (d_obs) must be specified in the linear setting"
            n_coef = 1 + d_obs + (d_obs if self.fit_interactions else 0) + 1
            self.theta_model = torch.nn.Parameter(torch.zeros(n_coef, dtype=torch.float64))

    def forward(self):
        if self.mode == "linear":
            return self.theta_model

    def _build_obs_design(self, Z, A):
        a_col = torch.tensor(A, dtype=torch.float64).reshape(-1, 1)
        z_tensor = torch.tensor(Z, dtype=torch.float64)
        intercept = torch.ones((z_tensor.shape[0], 1), dtype=torch.float64)
        if self.fit_interactions:
            return torch.cat((a_col, z_tensor, a_col * z_tensor, intercept), dim=1)
        return torch.cat((a_col, z_tensor, intercept), dim=1)

    def mean_est(self, lambda_, X_exp, X_obs):
        return (1 - lambda_) * np.mean(X_exp) + lambda_ * np.mean(X_obs)

    def fit_model(self, lambda_, X_exp, X_obs, obs_weights=None, obs_outcomes=None):
        if self.mode == "mean":
            self.theta = self.mean_est
            return

        if self.mode != "linear":
            raise ValueError(f"Unsupported mode in fit_model: {self.mode}")

        from rct.sim_data import true_pi_func

        beta_exp_precompute = _compute_exp_minimizer_from_experimental_data(
            X_exp,
            mode=self.mode,
            exp_model=self.exp_model,
            stratified_kfold=self.stratified_kfold,
            d_exp=self.d_exp,
            rng=self.rng,
            pi_func=true_pi_func,
        )
        if lambda_ == 0:
            padded_mini = torch.zeros_like(self.theta_model.detach())
            padded_mini[0] = beta_exp_precompute
            self.theta_model = torch.nn.Parameter(padded_mini)
            return

        Z = X_obs[:, : self.d_obs]
        A = X_obs[:, self.d_obs]
        Y = X_obs[:, -1] if obs_outcomes is None else np.asarray(obs_outcomes, dtype=float).reshape(-1)
        if obs_weights is None:
            obs_weights = np.ones(X_obs.shape[0], dtype=float)
        obs_weights = np.asarray(obs_weights, dtype=float)
        obs_weights = obs_weights / np.mean(obs_weights)
        weight_tensor = torch.tensor(obs_weights, dtype=torch.float64).reshape(-1, 1)

        design_obs = self._build_obs_design(Z, A)
        e1 = torch.zeros(design_obs.shape[1], dtype=torch.float64)
        e1[0] = 1
        e1 = e1.reshape(-1, 1)
        weighted_design = weight_tensor * design_obs
        weighted_outcome = weight_tensor * torch.tensor(Y, dtype=torch.float64).reshape(-1, 1)
        l_matrix = (1 - lambda_) * e1 @ e1.T + lambda_ / X_obs.shape[0] * design_obs.T @ weighted_design
        r_matrix = (1 - lambda_) * beta_exp_precompute * e1 + lambda_ / X_obs.shape[0] * design_obs.T @ weighted_outcome
        det = torch.det(l_matrix)
        if torch.isclose(det, torch.tensor(0.0, dtype=det.dtype), atol=1e-7):
            solution = torch.linalg.pinv(l_matrix) @ r_matrix
            self.theta_model = torch.nn.Parameter(solution.reshape(-1))
        else:
            solution = torch.linalg.solve(l_matrix, r_matrix)
            self.theta_model = torch.nn.Parameter(solution.reshape(-1))

    def beta(self, lambda_=None, X_exp=None, X_obs=None):
        if self.mode == "mean":
            return self.theta(lambda_, X_exp, X_obs)
        if self.mode == "linear":
            return self.theta_model[0]
        raise ValueError(f"Unsupported mode in beta: {self.mode}")

    def predict_tau(self, X):
        if self.mode != "linear":
            raise ValueError("predict_tau is only supported in linear mode.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        coef = self.theta_model.detach().cpu().numpy().reshape(-1)
        coef_a = coef[0]
        if self.fit_interactions:
            coef_ax = coef[1 + self.d_obs : 1 + 2 * self.d_obs]
        else:
            coef_ax = np.zeros(self.d_obs, dtype=float)
        return coef_a + X @ coef_ax

    def estimate_ate(self, X):
        return float(np.mean(self.predict_tau(X)))
