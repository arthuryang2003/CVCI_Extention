"""
Backward-compatible wrapper for generating ``lalonde.csv``.

Recommended new command:
    python -m rct.experiments.generate_lalonde_csv
Legacy command kept for compatibility:
    python rct/read_lalonde_data.py
"""

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rct.data import generate_lalonde_csv


if __name__ == "__main__":
    summary = generate_lalonde_csv()
    print(
        {
            "output_path": summary.output_path,
            "n_rows": summary.n_rows,
            "n_files": summary.n_files,
            "groups": summary.groups,
        }
    )
