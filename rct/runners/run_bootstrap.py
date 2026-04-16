"""Runner for bootstrap LaLonde experiments."""

from __future__ import annotations

import runpy
import sys


def main() -> None:
    sys.argv = ["rct.experiments.lalonde_cv_bootstrap", *sys.argv[1:]]
    runpy.run_module("rct.experiments.lalonde_cv_bootstrap", run_name="__main__")


if __name__ == "__main__":
    main()
