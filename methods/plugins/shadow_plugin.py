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
    predict_tau_shadow,
    screen_shadow_candidates_with_mode,
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
    shadow_relevance_group: Optional[str] = None
    random_state: int = 2024
    screening_mode: str = "screened"
    top_k: Optional[int] = None
    force_candidate_cols: Optional[Sequence[str]] = None

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

        if self.force_candidate_cols is not None:
            selected_shadow_cols = [str(c) for c in self.force_candidate_cols]
            missing = [c for c in selected_shadow_cols if c not in set(x_cols)]
            if missing:
                raise ValueError(f"ShadowPlugin force_candidate_cols contains unknown columns: {missing}")
            xc_cols = [c for c in x_cols if c not in set(selected_shadow_cols)]
            screening = {
                "selected_shadow_cols": selected_shadow_cols,
                "Xc_cols": xc_cols,
                "Xz_cols": selected_shadow_cols,
                "screening_logs": [
                    {
                        "column": c,
                        "selected": bool(c in set(selected_shadow_cols)),
                        "forced_selected": bool(c in set(selected_shadow_cols)),
                    }
                    for c in x_cols
                ],
                "relevance_threshold": float(self.association_threshold),
                "independence_threshold": float(self.residual_independence_threshold),
                "allow_empty_fallback": bool(self.allow_empty_fallback),
                "relevance_group": self.shadow_relevance_group,
                "source_g": 1,
                "target_g": 0,
                "shadow_direction": "rct_to_obs",
                "screening_mode": "forced",
                "top_k": None,
                "candidate_cols": x_cols,
                "force_candidate_cols": selected_shadow_cols,
            }
        else:
            screening = screen_shadow_candidates_with_mode(
                df_all,
                X_cols=x_cols,
                t_col=a_col,
                y_col=y_col,
                g_col=g_col,
                relevance_threshold=float(self.association_threshold),
                independence_threshold=float(self.residual_independence_threshold),
                allow_empty_fallback=bool(self.allow_empty_fallback),
                relevance_group=self.shadow_relevance_group,
                shadow_direction="rct_to_obs",
                source_g=1,
                target_g=0,
                screening_mode=str(self.screening_mode),
                top_k=self.top_k,
                force_candidate_cols=self.force_candidate_cols,
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
            shadow_direction="rct_to_obs",
            source_g=1,
            target_g=0,
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
            "shadow_direction": "rct_to_obs",
            "source_g": 1,
            "target_g": 0,
            "shadow_relevance_group": self.shadow_relevance_group,
            "screening_mode": str(self.screening_mode),
            "top_k": None if self.top_k is None else int(self.top_k),
            "force_candidate_cols": None
            if self.force_candidate_cols is None
            else [str(c) for c in self.force_candidate_cols],
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
            shadow_direction="rct_to_obs",
            source_g=1,
            target_g=0,
        )
        self.diagnostics_["shadow_target_diagnostics"] = {
            "mean_tau_shadow": corrected["diagnostics"]["mean_tau_shadow"],
            "std_tau_shadow": corrected["diagnostics"]["std_tau_shadow"],
            "mean_corrected_target": corrected["diagnostics"]["mean_corrected_target"],
            "std_corrected_target": corrected["diagnostics"]["std_corrected_target"],
        }
        return np.asarray(corrected["corrected_targets"], dtype=float)

    def get_regression_recovered_rct_signal(
        self,
        df_rct: pd.DataFrame,
        df_obs: Optional[pd.DataFrame] = None,
        x_cols: Optional[Sequence[str]] = None,
        a_col: str = "T",
        y_col: str = "Y",
        g_col: str = "G",
        raw_pseudo_effect: Optional[np.ndarray] = None,
        base_w_hat: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        _ = df_obs, x_cols, a_col, y_col, g_col, raw_pseudo_effect, base_w_hat
        if not self.fitted_:
            return None

        rng = np.random.default_rng(int(self.random_state))
        mu1_list = []
        mu0_list = []
        tau_list = []
        for _, row in df_rct.iterrows():
            tau_obj = predict_tau_shadow(
                self.shadow_models_,
                xc_vec=row[self.xc_cols_].to_numpy(dtype=float) if self.xc_cols_ else np.asarray([], dtype=float),
                xz_vec=row[self.xz_cols_].to_numpy(dtype=float) if self.xz_cols_ else np.asarray([], dtype=float),
                M=int(self.shadow_mc_samples),
                rng=rng,
                shadow_direction="rct_to_obs",
                source_g=1,
                target_g=0,
            )
            mu1_list.append(float(tau_obj["mu1_shadow"]))
            mu0_list.append(float(tau_obj["mu0_shadow"]))
            tau_list.append(float(tau_obj["tau_shadow"]))

        tau_shadow = np.asarray(tau_list, dtype=float).reshape(-1)
        if np.any(~np.isfinite(tau_shadow)):
            raise ValueError("ShadowPlugin produced non-finite regression recovered RCT signal.")

        self.diagnostics_["shadow_regression_recovery_diagnostics"] = {
            "mean_mu1_shadow": float(np.mean(mu1_list)),
            "std_mu1_shadow": float(np.std(mu1_list)),
            "mean_mu0_shadow": float(np.mean(mu0_list)),
            "std_mu0_shadow": float(np.std(mu0_list)),
            "mean_tau_shadow": float(np.mean(tau_shadow)),
            "std_tau_shadow": float(np.std(tau_shadow)),
            "shadow_direction": "rct_to_obs",
            "source_g": 1,
            "target_g": 0,
        }
        return tau_shadow
