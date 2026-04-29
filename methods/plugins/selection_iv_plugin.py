"""Selection-IV plugin for OBS-target second-stage reweighting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from methods.iv import fit_iv_pipeline, select_iv_candidates_with_mode
from methods.plugins.base import SelectionCorrectionPlugin
from utils.weight_utils import ensure_1d_float, finalize_weights, weight_summary


@dataclass
class SelectionIVPlugin(SelectionCorrectionPlugin):
    """
    Selection-IV plugin (source-selection IV, not treatment IV).

    Workflow:
    1. Screen IV candidates from covariates.
    2. Fit formal IV pipeline to obtain per-sample source-selection correction weights.
    3. Return RCT weights for OBS-target second-stage bias learning.
    """

    name: str = "selection_iv"
    iv_candidate_cols: Optional[Sequence[str]] = None
    x_cols_for_iv_screen: Optional[Sequence[str]] = None
    use_treatment_in_screen: bool = True
    relevance_threshold: float = 0.02
    exclusion_threshold: float = 0.02
    clip_min: float = 0.05
    clip_max: float = 20.0
    model_type: str = "logistic"
    screening_mode: str = "screened"
    top_k: Optional[int] = None
    force_candidate_cols: Optional[Sequence[str]] = None
    verbose: bool = True

    def _log(self, message: str):
        if self.verbose:
            print(f"[SelectionIVPlugin] {message}")

    def fit(self, df_rct, df_obs, x_cols, a_col, y_col, g_col):
        df_all = pd.concat([df_rct, df_obs], axis=0, ignore_index=True)
        candidate_cols = list(self.iv_candidate_cols) if self.iv_candidate_cols is not None else list(x_cols)
        if self.x_cols_for_iv_screen is not None:
            candidate_cols = [c for c in candidate_cols if c in set(self.x_cols_for_iv_screen)]
        if not candidate_cols:
            raise ValueError("SelectionIVPlugin has empty candidate_cols after screening configuration.")

        screening_result = select_iv_candidates_with_mode(
            df_all,
            candidate_cols=candidate_cols,
            t_col=a_col,
            y_col=y_col,
            g_col=g_col,
            relevance_threshold=float(self.relevance_threshold),
            exclusion_threshold=float(self.exclusion_threshold),
            allow_empty_fallback=True,
            screening_mode=str(self.screening_mode),
            top_k=self.top_k,
            force_candidate_cols=self.force_candidate_cols,
        )
        selected_iv_cols = list(screening_result["selected_iv_cols"])
        xc_cols = list(screening_result["Xc_cols"])

        all_weights = fit_iv_pipeline(
            df_all,
            Xc_cols=xc_cols,
            Xz_cols=selected_iv_cols,
            t_col=a_col,
            y_col=y_col,
            g_col=g_col,
            weight_clip_min=self.clip_min,
            weight_clip_max=self.clip_max,
        )
        weights = ensure_1d_float(all_weights[: df_rct.shape[0]])
        weights = finalize_weights(weights, clip_min=self.clip_min, clip_max=self.clip_max, normalize=True)

        if np.any(~np.isfinite(weights)):
            raise ValueError("SelectionIVPlugin produced non-finite weights.")

        self.selected_iv_cols_ = list(selected_iv_cols)
        self.weights_ = weights
        self.fitted_ = True
        self.diagnostics_ = {
            "plugin": self.name,
            "screening": screening_result,
            "selected_iv_cols": self.selected_iv_cols_,
            "Xc_cols": xc_cols,
            "Xz_cols": self.selected_iv_cols_,
            "screening_mode": str(self.screening_mode),
            "top_k": None if self.top_k is None else int(self.top_k),
            "force_candidate_cols": None
            if self.force_candidate_cols is None
            else [str(c) for c in self.force_candidate_cols],
            "weight_summary": weight_summary(self.weights_),
        }
        return self

    def get_rct_weights(self, df_rct: pd.DataFrame) -> Optional[np.ndarray]:
        if not self.fitted_:
            return None
        _ = df_rct
        return self.weights_

    def get_corrected_bias_target(self, df_rct: pd.DataFrame, base_w_hat: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        _ = df_rct, base_w_hat
        return None
