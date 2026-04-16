"""Baseline estimators for RCT-target experiments."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats


def t_test_normal_baseline(x_exp, x_obs, alpha_threshold=0.05, equal_var=True):
    """No-covariate t-test baseline with optional pooled fallback."""
    t_test_result = stats.ttest_ind(x_exp, x_obs, equal_var=equal_var)
    if t_test_result.pvalue < alpha_threshold:
        return np.mean(x_exp)
    return np.mean(np.concatenate((x_exp, x_obs), axis=0))
