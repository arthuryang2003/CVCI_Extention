"""Shadow plugin wrapper for OBS-target second-stage corrected targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from methods.plugins.base import SelectionCorrectionPlugin
from methods.shadow import (
    build_shadow_corrected_targets_for_rhc,
    fit_shadow_pipeline,
    screen_shadow_candidates,
)


@dataclass
class ShadowPlugin(SelectionCorrectionPlugin):
    """Shadow plugin for OBS-target RHC: modifies second-stage bias target ``tau_shadow - w_hat``."""

    name: str = "shadow"
    shadow_candidate_cols: Optional[Sequence[str]] = None
    x_cols_for_shadow_screen: Optional[Sequence[str]] = None
    use_treatment_in_screen: bool = True
    association_threshold: float = 0.02
    residual_independence_threshold: float = 0.02
    or_model_type: str = "exp_linear"
    clip_min: float = 0.05
    clip_max: float = 20.0
    verbose: bool = True
    shadow_mc_samples: int = 2000
    allow_empty_fallback: bool = False
    random_state: int = 2024

    def _log(self, message: str):
        if self.verbose:
            print(f"[ShadowPlugin] {message}")

    def fit(
        self,
        df_rct: pd.DataFrame,
        df_obs: pd.DataFrame,
        x_cols,
        a_col,
        y_col,
        g_col,
    ):
        _ = self.shadow_candidate_cols, self.x_cols_for_shadow_screen, self.use_treatment_in_screen
        _ = self.or_model_type, self.clip_min, self.clip_max

        x_cols = list(x_cols)
        df_all = pd.concat([df_rct, df_obs], axis=0, ignore_index=True)

        screening = screen_shadow_candidates(
            df_all,
            X_cols=x_cols,
            t_col=a_col,
            y_col=y_col,
            g_col=g_col,
            relevance_threshold=float(self.association_threshold),
            independence_threshold=float(self.residual_independence_threshold),
            allow_empty_fallback=bool(self.allow_empty_fallback),
        )
        selected_shadow_cols = list(screening["selected_shadow_cols"])
        xc_cols = list(screening["Xc_cols"])
        xz_cols = list(selected_shadow_cols)

        shadow_models = fit_shadow_pipeline(
            df_all,
            Xc_cols=xc_cols,
            Xz_cols=xz_cols,
            t_col=a_col,
            y_col=y_col,
            g_col=g_col,
        )

        self.shadow_models_ = shadow_models
        self.selected_shadow_cols_ = selected_shadow_cols
        self.xc_cols_ = list(shadow_models["Xc_cols"])
        self.xz_cols_ = list(shadow_models["Xz_cols"])
        self.x_cols_ = x_cols
        self.a_col_ = a_col
        self.y_col_ = y_col
        self.g_col_ = g_col
        self.fitted_ = True

        self.diagnostics_ = {
            "plugin": self.name,
            "selected_shadow_cols": self.selected_shadow_cols_,
            "Xc_cols": self.xc_cols_,
            "Xz_cols": self.xz_cols_,
            "screening_logs": screening["screening_logs"],
            "screening": screening,
            "shadow_mc_samples": int(self.shadow_mc_samples),
            "allow_empty_fallback": bool(self.allow_empty_fallback),
        }
        self._log(f"Selected shadow cols: {self.selected_shadow_cols_}")
        return self

    def get_rct_weights(self, df_rct: pd.DataFrame) -> Optional[np.ndarray]:
        _ = df_rct
        return None

    def get_corrected_bias_target(
        self,
        df_rct: pd.DataFrame,
        base_w_hat: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        if not self.fitted_:
            return None
        if base_w_hat is None:
            raise ValueError("ShadowPlugin requires base_w_hat for corrected second-stage RHC targets.")

        corrected = build_shadow_corrected_targets_for_rhc(
            df_rct=df_rct,
            shadow_models=self.shadow_models_,
            w_hat_rct=np.asarray(base_w_hat, dtype=float),
            Xc_cols=self.xc_cols_,
            Xz_cols=self.xz_cols_,
            t_col=self.a_col_,
            M=int(self.shadow_mc_samples),
            random_state=int(self.random_state),
        )
        self.diagnostics_["shadow_target_diagnostics"] = {
            "mean_tau_shadow": corrected["diagnostics"]["mean_tau_shadow"],
            "std_tau_shadow": corrected["diagnostics"]["std_tau_shadow"],
            "mean_corrected_target": corrected["diagnostics"]["mean_corrected_target"],
            "std_corrected_target": corrected["diagnostics"]["std_corrected_target"],
        }
        return np.asarray(corrected["corrected_targets"], dtype=float)
