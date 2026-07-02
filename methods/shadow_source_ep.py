"""Simplified shadow-route extended participation probability utilities.

This is an engineering placeholder, not full shadow-variable identification.
The current implementation directly learns ``P(G=1 | Xc, T, Y)`` and keeps the
output interface stable so it can later be replaced by a proper shadow-based
OR / selection-bias-function identification method.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _validate_shadow_source_ep_inputs(
    df: pd.DataFrame,
    Xc_cols: Sequence[str],
    Xz_cols: Optional[Sequence[str]],
    treatment_col: str,
    outcome_col: str,
    source_col: str,
) -> Dict[str, object]:
    xc_cols = [str(col) for col in Xc_cols]
    xz_cols = [] if Xz_cols is None else [str(col) for col in Xz_cols]

    if not xc_cols:
        raise ValueError("Xc_cols must be non-empty for shadow_source_ep.")

    required = [source_col, treatment_col, outcome_col, *xc_cols, *xz_cols]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for shadow_source_ep: {missing}")

    unique_g = set(pd.Series(df[source_col]).dropna().astype(float).unique().tolist())
    if not unique_g.issubset({0.0, 1.0}):
        raise ValueError(f"{source_col} must be binary 0/1 for shadow_source_ep, got values={sorted(unique_g)}.")
    if len(unique_g) < 2:
        raise ValueError(f"{source_col} must contain both 0 and 1 for shadow_source_ep.")

    probability_features = list(dict.fromkeys([*xc_cols, treatment_col, outcome_col]))
    return {
        "Xc_cols": xc_cols,
        "Xz_cols": xz_cols,
        "probability_features": probability_features,
    }


def fit_shadow_source_ep_pipeline(
    df: pd.DataFrame,
    Xc_cols: Sequence[str],
    Xz_cols: Optional[Sequence[str]] = None,
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    source_col: str = "G",
    target: str = "obs",
    clip: float = 0.05,
    random_state: int = 42,
    return_model: bool = False,
):
    """Fit a simplified shadow-route participation model.

    The method intentionally does not use shadow variables ``Xz`` as training
    inputs. It only learns ``pi_shadow(Xc, T, Y) = P(G=1 | Xc, T, Y)`` while
    recording ``Xz_cols`` as selected shadow columns for downstream summaries.
    """
    validated = _validate_shadow_source_ep_inputs(
        df=df,
        Xc_cols=Xc_cols,
        Xz_cols=Xz_cols,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        source_col=source_col,
    )
    probability_features = list(validated["probability_features"])
    selected_shadow_cols = list(validated["Xz_cols"])

    clip = float(clip)
    if not 0.0 < clip < 0.5:
        raise ValueError(f"clip must be in (0, 0.5), got {clip}.")

    x_frame = df[probability_features].copy()
    g_vec = df[source_col].to_numpy(dtype=float)

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_transformer, probability_features)],
        remainder="drop",
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(max_iter=10000, random_state=random_state)),
        ]
    )
    model.fit(x_frame, g_vec)

    pi_shadow = model.predict_proba(x_frame)[:, 1]
    pi_shadow = np.clip(np.asarray(pi_shadow, dtype=float), clip, 1.0 - clip)
    weight_to_rct = pi_shadow / (1.0 - pi_shadow)
    weight_to_obs = (1.0 - pi_shadow) / pi_shadow

    target_normalized = str(target).lower()
    if target_normalized == "rct":
        sample_weight = weight_to_rct
    elif target_normalized == "obs":
        sample_weight = weight_to_obs
    else:
        raise ValueError(f"target must be 'rct' or 'obs', got {target}.")

    for name, values in {
        "pi_shadow": pi_shadow,
        "weight_to_rct": weight_to_rct,
        "weight_to_obs": weight_to_obs,
        "sample_weight": sample_weight,
    }.items():
        if np.any(~np.isfinite(values)):
            raise ValueError(f"shadow_source_ep produced non-finite {name}.")

    return {
        "pi_shadow": pi_shadow,
        "weight_to_rct": weight_to_rct,
        "weight_to_obs": weight_to_obs,
        "sample_weight": sample_weight,
        "selected_shadow_cols": selected_shadow_cols,
        "probability_features": probability_features,
        "model": model if return_model else None,
    }


def add_shadow_source_ep_columns(
    df: pd.DataFrame,
    Xc_cols: Sequence[str],
    Xz_cols: Optional[Sequence[str]] = None,
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    source_col: str = "G",
    target: str = "obs",
    clip: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a dataframe copy with shadow_source_ep probability columns added."""
    fitted = fit_shadow_source_ep_pipeline(
        df=df,
        Xc_cols=Xc_cols,
        Xz_cols=Xz_cols,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        source_col=source_col,
        target=target,
        clip=clip,
        random_state=random_state,
        return_model=False,
    )
    enriched = df.copy()
    enriched["shadow_source_ep_pi"] = fitted["pi_shadow"]
    enriched["shadow_source_ep_weight_to_rct"] = fitted["weight_to_rct"]
    enriched["shadow_source_ep_weight_to_obs"] = fitted["weight_to_obs"]
    enriched["shadow_source_ep_weight"] = fitted["sample_weight"]
    return enriched
