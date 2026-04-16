"""Top-level package exports for RCT utilities."""

from rct.data import (
    CATEGORY_COLUMNS,
    LALONDE_COLUMNS,
    LalondeBuildSummary,
    add_lalonde_engineered_features,
    build_lalonde_dataframe,
    collect_lalonde_txt_files,
    ensure_required_columns,
    generate_lalonde_csv,
    infer_group_from_filename,
    load_lalonde_csv,
    read_lalonde_txt_file,
    split_obs_target_groups,
    lalonde_get_data,
)

__all__ = [
    "LALONDE_COLUMNS",
    "CATEGORY_COLUMNS",
    "LalondeBuildSummary",
    "infer_group_from_filename",
    "read_lalonde_txt_file",
    "collect_lalonde_txt_files",
    "build_lalonde_dataframe",
    "add_lalonde_engineered_features",
    "load_lalonde_csv",
    "split_obs_target_groups",
    "ensure_required_columns",
    "generate_lalonde_csv",
    "lalonde_get_data",
    "RCTTargetBaseEstimator",
]


def __getattr__(name: str):
    if name == "RCTTargetBaseEstimator":
        from rct.estimator import RCTTargetBaseEstimator as _RCTTargetBaseEstimator

        return _RCTTargetBaseEstimator
    raise AttributeError(f"module 'rct' has no attribute {name!r}")
