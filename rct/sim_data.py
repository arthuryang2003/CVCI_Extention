"""Synthetic data utilities for RCT-target simulations."""

from __future__ import annotations

import numpy as np


def generate_data(n, d, true_mu_coef, true_te, pi_func, noise=0.1, rng=None):
    """Generate with-covariate synthetic data."""
    if rng is None:
        X = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=n)
        pi = pi_func(X)
        A = np.array([np.random.binomial(1, pi[i]) for i in range(n)])
        Y = X @ true_mu_coef + A * true_te + noise * np.random.randn(n)
    else:
        X = rng.multivariate_normal(np.zeros(d), np.eye(d), size=n)
        pi = pi_func(X)
        A = np.array([rng.binomial(1, pi[i]) for i in range(n)])
        Y = X @ true_mu_coef + A * true_te + noise * rng.normal(size=n)
    return X, A, Y


def true_pi_func(X):
    """Default true propensity score function."""
    return 0.5 * np.ones(X.shape[0])


def tilde_pi_func(X):
    """Misspecified propensity score function."""
    return 0.2 * np.ones(X.shape[0])
