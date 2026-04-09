"""CLI entrypoint to generate ``lalonde.csv`` from raw TXT files."""

from __future__ import annotations

import argparse
import json

from rct.data import generate_lalonde_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate lalonde.csv from raw LaLonde TXT files.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing source TXT files. Defaults to <repo_root>/data.",
    )
    parser.add_argument("--pattern", type=str, default="*.txt", help="Glob pattern for source files.")
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output CSV path. Defaults to <repo_root>/lalonde.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_lalonde_csv(data_dir=args.data_dir, output_path=args.output_path, pattern=args.pattern)
    print(
        json.dumps(
            {
                "output_path": summary.output_path,
                "n_rows": summary.n_rows,
                "n_files": summary.n_files,
                "groups": summary.groups,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
