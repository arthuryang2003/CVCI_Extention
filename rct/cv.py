"""Cross-validation utilities for RCT-target CVCI."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold

from rct.losses import L_exp
from rct.models import model_class


def cross_validation(
    X_exp,
    X_obs,
    lambda_vals,
    mode="mean",
    k_fold=None,
    d_exp=None,
    d_obs=None,
    exp_model="aipw",
    stratified_kfold=False,
    random_state=None,
    rng=None,
    obs_weights=None,
    obs_outcomes=None,
):
    """Calculate CV objective for each lambda and return the selected model."""
    if k_fold is None:
        cross_validator = LeaveOneOut()
    else:
        if stratified_kfold:
            cross_validator = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=random_state)
        else:
            cross_validator = KFold(n_splits=k_fold, shuffle=True, random_state=random_state)

    Q_values = np.zeros_like(lambda_vals)
    for i, lambda_ in enumerate(lambda_vals):
        current_Q = 0
        if stratified_kfold:
            numerator = cross_validator.split(X_exp, X_exp[:, d_exp])
        else:
            numerator = cross_validator.split(X_exp)
        for train_index, val_index in numerator:
            model = model_class(
                mode=mode,
                d_exp=d_exp,
                d_obs=d_obs,
                exp_model=exp_model,
                stratified_kfold=stratified_kfold,
                rng=rng,
            )
            X_train, X_val = X_exp[train_index], X_exp[val_index]
            model.fit_model(lambda_, X_train, X_obs, obs_weights=obs_weights, obs_outcomes=obs_outcomes)
            with torch.no_grad():
                l_exp_fold = L_exp(
                    model.beta(lambda_, X_train, X_obs),
                    X_val,
                    mode=mode,
                    exp_model=exp_model,
                    stratified_kfold=stratified_kfold,
                    d_exp=d_exp,
                    rng=rng,
                )
                current_Q += l_exp_fold.item()
        Q_values[i] += current_Q

    Q_values /= X_exp.shape[0]
    lambda_opt = lambda_vals[np.argmin(Q_values)]
    theta_opt = model_class(
        mode=mode,
        d_exp=d_exp,
        d_obs=d_obs,
        exp_model=exp_model,
        stratified_kfold=stratified_kfold,
        rng=rng,
    )
    theta_opt.fit_model(lambda_opt, X_exp, X_obs, obs_weights=obs_weights, obs_outcomes=obs_outcomes)
    return Q_values, lambda_opt, theta_opt
