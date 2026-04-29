# Repository Guidelines

## 项目结构与模块组织
核心 Python 包按目标域划分：`rct/`（RCT 目标估计器、runner 与实验）、`obs/`（OBS 目标估计器栈）、`methods/plugins/`（共享插件实现，如 `ipsw`、`cw`、`iv`、`shadow`）。

顶层实验编排在 `run_lalonde_experiment.py` 与 `experiments/`（基准脚本、输出与图表）。数据输入位于 `data/`，`lalonde.csv` 在仓库根目录。通用工具在 `utils/`；分析与绘图脚本在 `analysis/`。

## 构建、测试与开发命令
请在仓库根目录使用模块方式运行：

- `python -m rct.runners.example_use`：快速功能性自检。
- `python run_lalonde_experiment.py --target-mode rct --method cvci --obs-source cps --plugin iv`：运行一次可配置的 LaLonde 实验。
- `python -m rct.runners.run_lalonde --task cv`：运行推荐的 LaLonde CV 任务。
- `python -m rct.experiments.generate_lalonde_csv --data-dir data --output-path lalonde.csv`：从原始 `.txt` 重新生成合并 CSV。
- `python experiments/smoke_selection_plugins.py`：对插件选择路径做 smoke 测试。

## 代码风格与命名约定
遵循现有 Python 风格：4 空格缩进；函数/变量/模块使用 snake_case；类名使用 PascalCase；实验入口尽量使用显式关键字参数。模块按层分工（`data`、`models`、`losses`、`plugins`、`estimator`）并保持职责单一。

仓库未提交格式化/静态检查配置；请对齐周边代码风格，并保持导入分组清晰、顺序稳定。

## 测试指南
当前没有独立 `tests/` 测试套件。请通过有针对性的可运行脚本验证改动：

- 修改估计器后运行 `python -m rct.runners.example_use`。
- 修改插件或筛选逻辑后运行 `python experiments/smoke_selection_plugins.py`。
- 修改实验流水线后，用 `run_lalonde_experiment.py` 跑单个 seed，并检查 JSON 输出字段（`ate_hat`、`rmse`、插件摘要）。

新增 smoke/回归脚本命名建议：`experiments/smoke_<feature>.py`。

## 提交与 Pull Request 规范
近期历史提交多为简短祈使句（如 `iv modify`、`code structure modify`）。建议保持简洁但更明确，例如：`rct: refactor plugin screening`。

PR 建议包含：
- 变更范围摘要（涉及 `rct`、`obs`、`methods` 或 `experiments`）。
- 本地复现实验/验证命令。
- 结果变更时的产物路径（如 `experiments/*.json`、`experiments/figures/*.png`）。
- 对应 issue 或实验记录（如适用）。

## 数据与产物管理
除非为复现所必需，不要提交体积大且冗余的生成结果。原始输入保留在 `data/`，新增实验输出写入 `experiments/`，并使用包含 seed 信息的可读文件名。
