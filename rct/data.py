"""LaLonde CSV helpers for RCT package.

Canonical data logic is shared at ``utils.lalonde_utils``.
This module keeps the RCT-facing API stable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from utils.lalonde_utils import (
    CATEGORY_COLUMNS,
    LALONDE_COLUMNS,
    LalondeBuildSummary,
    add_lalonde_engineered_features,
    build_lalonde_dataframe,
    collect_lalonde_txt_files,
    ensure_required_columns,
    infer_group_from_filename,
    load_lalonde_csv,
    read_lalonde_txt_file,
    split_obs_target_groups,
)


def get_repo_root() -> Path:
    """Return repository root path, inferred from this module location."""
    return Path(__file__).resolve().parents[1]


def generate_lalonde_csv(
    data_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    pattern: str = "*.txt",
) -> LalondeBuildSummary:
    """Generate ``lalonde.csv`` from raw TXT files."""
    repo_root = get_repo_root()
    data_dir_path = Path(data_dir) if data_dir is not None else repo_root / "data"
    txt_files = collect_lalonde_txt_files(data_dir_path, pattern=pattern)
    lalonde = build_lalonde_dataframe(txt_files)

    output = Path(output_path) if output_path is not None else repo_root / "lalonde.csv"
    lalonde.to_csv(output, index=False)

    return LalondeBuildSummary(
        output_path=str(output),
        n_rows=int(lalonde.shape[0]),
        n_files=len(txt_files),
        groups=sorted(lalonde["group"].astype(str).unique().tolist()),
    )
