# Repository Guidelines

## 项目结构与模块组织
核心 Python 包按目标域划分：`rct/`（RCT 目标估计器、runner 与实验）、`obs/`（OBS 目标估计器栈）、`methods/plugins/`（共享插件实现，如 `ipsw`、`cw`、`iv`、`shadow`）。

顶层实验编排在 `run_experiment.py` 与 `experiments/`（基准脚本、输出与图表）。数据输入位于 `data/`，`lalonde.csv` 在仓库根目录。通用工具在 `utils/`；分析与绘图脚本在 `analysis/`。

## 运行环境约定
默认实验环境为 conda 环境 `CVCI`。后续命令优先使用：

- `conda run -n CVCI python ...`

## 构建、测试与开发命令
请在仓库根目录使用模块方式运行：

- `python -m rct.runners.example_use`：快速功能性自检。
- `python run_experiment.py --dataset lalonde --target-mode rct --method cvci --obs-source cps --plugin iv`：运行一次可配置实验。
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
- 修改实验流水线后，用 `run_experiment.py` 跑单个 seed，并检查 JSON 输出字段（`ate_hat`、`rmse`、插件摘要）。

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

## ACTG/JTPA semi-synthetic stress test 说明
当前 ACTG/JTPA 支持 `--data-mode real` 和 `--data-mode semi_synthetic`。`real + current` 必须保留原始构造：ACTG 使用 age-driven split，JTPA 使用 site/selection-driven split；real outcome 下没有 individual-level `tau_true`，因此 `proxy_ate` 下的 `rmse/proxy_abs_error` 只能解释为 ATE-level proxy error，不是真 CATE RMSE。

`semi_synthetic` 会保留真实 X/T/G，替换 Y 并生成 `tau_true`，用于 `truth_mode=synthetic_truth` 下计算真正 CATE RMSE。默认 outcome 是线性 clean 构造：

```text
tau_true = tau0 + tau_scale * x_score
Y = mu + T * tau_true + noise
```

这个默认构造对 RHC 很友好，因为 RHC 会先在 OBS 内直接学习 treatment effect；当 OBS 中的 apparent CATE 等于 `tau_true` 且结构近似线性时，RHC 的 direct learner 容易正确指定。为了更充分 stress-test selection correction，新增 OBS-specific treatment bias：

- `--semisynth-bias-mode none`：默认值，保持旧结果不变。
- `--semisynth-bias-mode obs_treatment_bias`：只对 OBS treated outcome 加线性 bias。
- `--semisynth-bias-mode nonlinear_obs_treatment_bias`：只对 OBS treated outcome 加更复杂的非线性 bias。
- `--semisynth-bias-mode localized_obs_treatment_bias`：只在 X score 高分位的 OBS treated 子区域加局部非线性 bias，用于降低 RHC 从 RCT correction 线性外推修正 bias 的能力。
- `--semisynth-bias-scale`：控制 bias 强度，建议先试 `0.5, 1.0, 1.5`。

对应构造为：

```text
obs_treatment_bias = (1 - G) * T * bias_scale * bias_score(X)
Y = mu + T * tau_true + obs_treatment_bias + noise
```

评估 truth 仍然是 `tau_true`。因此 RHC 在 OBS 内看到的是 `tau_true + bias_scale * bias_score(X)` 这种 apparent CATE，可能把 OBS-specific treatment bias 误学成 treatment effect heterogeneity。这个 stress test 的目的不是人为打压某个方法，而是构造更贴近 nonignorable source selection / treatment-confounding correction 要解决的问题。

组会快速实验脚本：

```bash
conda run -n CVCI python experiments/meeting_quick_eval.py \
  --dataset actg \
  --data-path data/actg.csv \
  --target-mode obs \
  --seeds 0 1 2 \
  --semisynth-bias-mode localized_obs_treatment_bias \
  --semisynth-bias-scale 1.0 \
  --output-dir results/meeting_actg_bias
```

JTPA 同理把 `--dataset jtpa --data-path data/jtpa.csv --output-dir results/meeting_jtpa_bias`。如果打开 bias 后 Ours 仍不优于 Integrative-R/RHC，下一步应做 correction ablation：oracle source propensity、oracle IV/Shadow columns、以及 adaptive correction strength。
