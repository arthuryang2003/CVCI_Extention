# OBS Target Data Module

This folder contains an obs-target causal inference framework built on LaLonde data.

Main components:
- `obs/data.py`: data loading and standardized obs-target dataframe preparation
- `obs/models.py`: linear `w(x)` and `eta(x)` model components
- `obs/plugins.py`: selection correction plugins (`IPSW`, `CW`, `IV`, `Shadow`)
- `obs/estimator.py`: unified estimator

Core formulation:
- Train `w(x)` on OBS outcome model
- Build RCT grounding pseudo effect
- Train `eta(x)` on RCT bias target (`pseudo_effect - w_hat`), optionally corrected/reweighted by plugin
- Final estimate: `tau_hat(x) = w_hat(x) + eta_hat(x)`
