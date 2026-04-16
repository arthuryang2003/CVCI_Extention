"""Backward-compatible exports for legacy scripts.

Prefer importing from the new modules directly:
- rct.models
- rct.losses
- rct.cv
- rct.sim_data
- rct.data
- rct.baselines
"""

from rct.baselines import t_test_normal_baseline
from rct.cv import cross_validation
from rct.data import lalonde_get_data
from rct.losses import L_exp, L_obs, combined_loss, compute_exp_minmizer
from rct.models import model_class
from rct.sim_data import generate_data, tilde_pi_func, true_pi_func

__all__ = [
    "model_class",
    "compute_exp_minmizer",
    "L_exp",
    "L_obs",
    "combined_loss",
    "cross_validation",
    "generate_data",
    "true_pi_func",
    "tilde_pi_func",
    "lalonde_get_data",
    "t_test_normal_baseline",
]
