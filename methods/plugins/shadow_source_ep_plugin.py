"""Simplified shadow-route extended-participation plugin for OBS-target correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from methods.plugins.base import SelectionCorrectionPlugin
from methods.shadow import screen_shadow_candidates_with_mode
from methods.shadow_source_ep import fit_shadow_source_ep_pipeline
from utils.weight_utils import ensure_1d_float, finalize_weights, weight_summary


@dataclass
class ShadowSourceEPPlugin(SelectionCorrectionPlugin):
    """OBS-target plugin using simplified shadow-route participation probabilities.

    This is an engineering wrapper around ``P(G=1 | Xc, T, Y)``. It does not
    implement full shadow identification; it only preserves the intended output
    interface for RHC sample reweighting.
    """

    name: str = "shadow_source_ep"
    association_threshold: float = 0.02
    residual_independence_threshold: float = 0.02
    clip_min: float = 0.05
    clip_max: float = 20.0
    allow_empty_fallback: bool = False
    shadow_relevance_group: Optional[str] = None
    random_state: int = 2024
    screening_mode: str = "screened"
    top_k: Optional[int] = None
    force_candidate_cols: Optional[Sequence[str]] = None
    verbose: bool = True

    def _log(self, message: str):
        if self.verbose:
            print(f"[ShadowSourceEPPlugin] {message}")

    def fit(self, df_rct, df_obs, x_cols, a_col, y_col, g_col):
        x_cols = list(x_cols)
        df_all = pd.concat([df_rct, df_obs], axis=0, ignore_index=True)

        if self.force_candidate_cols is not None:
            selected_shadow_cols = [str(c) for c in self.force_candidate_cols]
            missing = [c for c in selected_shadow_cols if c not in set(x_cols)]
            if missing:
                raise ValueError(f"ShadowSourceEPPlugin force_candidate_cols contains unknown columns: {missing}")
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

        fitted = fit_shadow_source_ep_pipeline(
            df=df_all,
            Xc_cols=xc_cols,
            Xz_cols=selected_shadow_cols,
            treatment_col=a_col,
            outcome_col=y_col,
            source_col=g_col,
            target="obs",
            clip=float(self.clip_min),
            random_state=int(self.random_state),
            return_model=True,
        )

        raw_weights = ensure_1d_float(fitted["sample_weight"][: df_rct.shape[0]])
        weights = finalize_weights(raw_weights, clip_min=self.clip_min, clip_max=self.clip_max, normalize=True)
        if np.any(~np.isfinite(weights)):
            raise ValueError("ShadowSourceEPPlugin produced non-finite weights.")

        self.weights_ = weights
        self.selection_model_ = fitted["model"]
        self.selected_shadow_cols_ = selected_shadow_cols
        self.xc_cols_ = xc_cols
        self.fitted_ = True
        self.diagnostics_ = {
            "plugin": self.name,
            "method": self.name,
            "selected_shadow_cols": self.selected_shadow_cols_,
            "Xc_cols": self.xc_cols_,
            "Xz_cols": self.selected_shadow_cols_,
            "probability_features": list(fitted["probability_features"]),
            "screening_logs": screening["screening_logs"],
            "screening": screening,
            "pi_shadow_mean": float(np.mean(fitted["pi_shadow"])),
            "pi_shadow_min": float(np.min(fitted["pi_shadow"])),
            "pi_shadow_max": float(np.max(fitted["pi_shadow"])),
            "mean_weight": float(np.mean(weights)),
            "clip_min": float(self.clip_min),
            "clip_max": float(self.clip_max),
            "shadow_direction": "rct_to_obs",
            "source_g": 1,
            "target_g": 0,
            "shadow_relevance_group": self.shadow_relevance_group,
            "screening_mode": str(self.screening_mode),
            "top_k": None if self.top_k is None else int(self.top_k),
            "force_candidate_cols": None
            if self.force_candidate_cols is None
            else [str(c) for c in self.force_candidate_cols],
            "weight_summary": weight_summary(self.weights_),
            "description": (
                "Simplified engineering shadow probability plugin for OBS-target RHC. "
                "Learns P(G=1|Xc,T,Y) and uses odds weights to reweight RCT toward OBS."
            ),
        }
        self._log(f"Selected shadow cols: {self.selected_shadow_cols_}")
        return self

    def get_rct_weights(self, df_rct: pd.DataFrame):
        _ = df_rct
        return self.weights_

    def get_corrected_bias_target(self, df_rct: pd.DataFrame, base_w_hat: Optional[np.ndarray] = None):
        _ = df_rct, base_w_hat
        return None
