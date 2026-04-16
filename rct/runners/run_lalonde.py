"""Unified LaLonde runner for common RCT experiment scripts."""

from __future__ import annotations

import argparse
import runpy
import sys


TASK_TO_MODULE = {
    "cv": "rct.experiments.lalonde_cv",
    "intro_mean": "rct.experiments.lalonde_intro_mean",
    "intro_linear": "rct.experiments.lalonde_intro_linear",
    "plugin_linear": "rct.experiments.lalonde_cvci_plugin_linear",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_TO_MODULE.keys()), default="plugin_linear")
    return parser.parse_known_args()


def main() -> None:
    args, module_args = parse_args()
    sys.argv = [TASK_TO_MODULE[args.task], *module_args]
    runpy.run_module(TASK_TO_MODULE[args.task], run_name="__main__")


if __name__ == "__main__":
    main()
