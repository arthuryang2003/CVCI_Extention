"""Smoke examples for selection-IV and shadow plugins on OBS-target flow.

Usage:
    python experiments/smoke_selection_plugins.py --obs-source cps --n-rct 200 --n-obs 800
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from methods.plugins.selection_iv_plugin import SelectionIVPlugin
from methods.plugins.shadow_plugin import ShadowPlugin
from obs.estimator import ObsTargetBaseEstimator
from utils.lalonde_utils import get_lalonde_default_covariates, load_lalonde_csv, load_lalonde_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lalonde-path", type=str, default="lalonde.csv")
    parser.add_argument("--obs-source", type=str, default="cps", choices=["cps", "psid"])
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--n-rct", type=int, default=200)
    parser.add_argument("--n-obs", type=int, default=800)
    return parser.parse_args()


def _subsample(df, n: int, seed: int):
    if n <= 0 or n >= df.shape[0]:
        return df.copy()
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    raw = load_lalonde_csv(args.lalonde_path)
    x_cols = get_lalonde_default_covariates(raw)
    df_rct, df_obs, _ = load_lalonde_split(
        target_mode="obs",
        obs_source=args.obs_source,
        x_cols=x_cols,
        lalonde_path=args.lalonde_path,
    )
    df_rct = _subsample(df_rct, args.n_rct, args.seed)
    df_obs = _subsample(df_obs, args.n_obs, args.seed + 1)

    iv_plugin = SelectionIVPlugin(verbose=False)
    iv_plugin.fit(df_rct=df_rct, df_obs=df_obs, x_cols=x_cols, a_col="T", y_col="Y", g_col="G")
    iv_weights = iv_plugin.get_rct_weights(df_rct)

    shadow_plugin = ShadowPlugin(verbose=False, allow_empty_fallback=True, shadow_mc_samples=300)
    shadow_est = ObsTargetBaseEstimator(plugin=shadow_plugin, model_type="linear", random_state=args.seed)
    shadow_est.fit(df_rct=df_rct, df_obs=df_obs, x_cols=x_cols, a_col="T", y_col="Y", g_col="G")

    payload = {
        "iv": {
            "selected_iv_cols": iv_plugin.summary().get("selected_iv_cols"),
            "weight_summary": iv_plugin.summary().get("weight_summary"),
            "n_weights": int(np.asarray(iv_weights).shape[0]) if iv_weights is not None else 0,
        },
        "shadow": {
            "selected_shadow_cols": shadow_plugin.summary().get("selected_shadow_cols"),
            "shadow_target_diagnostics": shadow_plugin.summary().get("shadow_target_diagnostics"),
            "eta_train_size": int(shadow_est.summary().get("eta_train_size", 0)),
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
