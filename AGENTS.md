# Repository Guidelines

## Project Structure & Module Organization
Core Python packages are organized by target domain: `rct/` (RCT-target estimators, runners, and experiments), `obs/` (OBS-target estimator stack), and `methods/plugins/` (shared plugin implementations such as `ipsw`, `cw`, `iv`, and `shadow`).

Top-level experiment orchestration lives in `run_lalonde_experiment.py` and `experiments/` (benchmark scripts, outputs, and figures). Data inputs are under `data/` and `lalonde.csv` at repo root. Utility helpers are in `utils/`; analysis/plot scripts are in `analysis/`.

## Build, Test, and Development Commands
Use module-style execution from repository root:

- `python -m rct.runners.example_use`: quick functional sanity check.
- `python run_lalonde_experiment.py --target-mode rct --method cvci --obs-source cps --plugin iv`: run one configurable LaLonde experiment.
- `python -m rct.runners.run_lalonde --task cv`: run recommended LaLonde CV task.
- `python -m rct.experiments.generate_lalonde_csv --data-dir data --output-path lalonde.csv`: regenerate the merged CSV from raw `.txt` files.
- `python experiments/smoke_selection_plugins.py`: smoke-test plugin selection paths.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, snake_case for functions/variables/modules, PascalCase for classes, and explicit keyword arguments in experiment entrypoints. Keep modules focused by layer (`data`, `models`, `losses`, `plugins`, `estimator`).

No formatter/linter config is checked in; match surrounding style and keep imports grouped and deterministic.

## Testing Guidelines
There is no dedicated `tests/` suite yet. Validate changes with targeted runnable scripts:

- Run `python -m rct.runners.example_use` after estimator changes.
- Run `python experiments/smoke_selection_plugins.py` after plugin/screening edits.
- For experiment pipeline edits, run one seed via `run_lalonde_experiment.py` and verify JSON output fields (`ate_hat`, `rmse`, plugin summary).

Name new smoke/regression scripts as `experiments/smoke_<feature>.py`.

## Commit & Pull Request Guidelines
Recent history uses short, imperative messages (for example, `iv modify`, `code structure modify`). Prefer clearer but similarly concise subjects such as `rct: refactor plugin screening`.

PRs should include:
- Scope summary (which package: `rct`, `obs`, `methods`, or `experiments`).
- Reproduction commands run locally.
- Output artifact paths when results change (for example `experiments/*.json`, `experiments/figures/*.png`).
- Linked issue or experiment note when applicable.

## Data & Artifact Hygiene
Do not commit large, redundant generated outputs unless required for reproducibility. Keep raw inputs in `data/`, and write new experiment outputs under `experiments/` with descriptive, seed-aware filenames.
