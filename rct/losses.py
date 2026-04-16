"""Loss components for RCT-target CVCI."""

from __future__ import annotations

import numpy as np
import torch

from rct.models import _compute_exp_minimizer_from_experimental_data
from rct.sim_data import true_pi_func


def compute_exp_minmizer(X, mode="linear", exp_model="aipw", stratified_kfold=False, d_exp=None, rng=None):
    """Compute treatment effect estimate from experimental data."""
    return _compute_exp_minimizer_from_experimental_data(
        X,
        mode=mode,
        exp_model=exp_model,
        stratified_kfold=stratified_kfold,
        d_exp=d_exp,
        rng=rng,
        pi_func=true_pi_func,
    )


def L_exp(beta, X, mode="mean", beta_exp_precompute=None, exp_model="aipw", stratified_kfold=False, d_exp=None, rng=None):
    """Compute the experimental-data loss."""
    if mode == "mean":
        return np.mean((X - beta) ** 2)
    if mode == "linear":
        if beta_exp_precompute is None:
            if d_exp is None:
                raise ValueError("please specify d_exp in L_exp")
            beta_exp = compute_exp_minmizer(
                X,
                mode=mode,
                exp_model=exp_model,
                stratified_kfold=stratified_kfold,
                d_exp=d_exp,
                rng=rng,
            )
            beta_exp = torch.tensor(beta_exp)
        else:
            beta_exp = torch.tensor(beta_exp_precompute)
        return (beta_exp - beta) ** 2
    raise ValueError(f"Unsupported mode in L_exp: {mode}")


def L_obs(theta_model, X, mode="mean", d_obs=None, obs_weights=None, obs_outcomes=None):
    """Compute the observational-data loss."""
    if mode == "mean":
        if obs_weights is None:
            obs_mean = np.mean(X)
        else:
            obs_weights = np.asarray(obs_weights, dtype=float)
            obs_weights = obs_weights / np.sum(obs_weights)
            obs_mean = np.sum(obs_weights * X)
        return np.mean((obs_mean - theta_model) ** 2)

    if mode == "linear":
        if d_obs is None:
            raise ValueError("please specify d_obs in L_obs")
        X = torch.tensor(X, dtype=torch.float64)
        Z = X[:, :d_obs]
        A = X[:, d_obs]
        Y = X[:, -1] if obs_outcomes is None else torch.tensor(obs_outcomes, dtype=torch.float64).reshape(-1)
        n_non_intercept = int(theta_model.shape[0] - 1)
        if n_non_intercept == 1 + 2 * d_obs:
            design = torch.cat([A.view(-1, 1), Z, A.view(-1, 1) * Z], dim=1)
        elif n_non_intercept == 1 + d_obs:
            design = torch.cat([A.view(-1, 1), Z], dim=1)
        else:
            raise ValueError(
                "Unexpected theta_model dimension in L_obs: "
                f"{theta_model.shape[0]} for d_obs={d_obs}."
            )
        X_pred = torch.matmul(design, theta_model[:-1]) + theta_model[-1]
        sq_error = (X_pred - Y) ** 2
        if obs_weights is None:
            return torch.mean(sq_error)
        obs_weights = np.asarray(obs_weights, dtype=float)
        obs_weights = obs_weights / np.sum(obs_weights)
        return torch.sum(torch.tensor(obs_weights, dtype=torch.float64) * sq_error)

    raise ValueError(f"Unsupported mode in L_obs: {mode}")


def combined_loss(
    theta,
    X_exp,
    X_obs,
    lambda_,
    mode="mean",
    beta_exp_precompute=None,
    exp_model="aipw",
    stratified_kfold=False,
    d_exp=None,
    d_obs=None,
    rng=None,
    obs_weights=None,
    obs_outcomes=None,
):
    """Compute combined experimental + observational loss."""
    if mode == "mean":
        return (1 - lambda_) * L_exp(theta.beta(lambda_, X_exp, X_obs), X_exp, mode=mode) + lambda_ * L_obs(
            theta(lambda_, X_exp, X_obs),
            X_obs,
            mode=mode,
            obs_weights=obs_weights,
            obs_outcomes=obs_outcomes,
        )

    if mode == "linear":
        if d_obs is None:
            raise ValueError("please specify d_obs in combined_loss")
        loss_exp = L_exp(
            theta.beta(),
            X_exp,
            mode=mode,
            beta_exp_precompute=beta_exp_precompute,
            exp_model=exp_model,
            stratified_kfold=stratified_kfold,
            d_exp=d_exp,
            rng=rng,
        )
        loss_obs = L_obs(
            theta.theta_model,
            X_obs,
            mode=mode,
            d_obs=d_obs,
            obs_weights=obs_weights,
            obs_outcomes=obs_outcomes,
        )
        return (1 - lambda_) * loss_exp + lambda_ * loss_obs

    raise ValueError(f"Unsupported mode in combined_loss: {mode}")
