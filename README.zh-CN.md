# HTE 数据集上的产率预测与条件推荐

**[English](README.md)** · 中文

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23772914.svg)](https://doi.org/10.5281/zenodo.23772914)

投稿至 *Chemometrics and Intelligent Laboratory Systems* 的论文所配套的代码与结果表。

## 仓库内容

| 路径 | 内容 |
|---|---|
| `hte/` | 核心包：数据集加载、槽位映射、特征构造、模型 |
| `archive/single_dataset_full_benchmark.py` | 第二套流水线通路：单数据集、端到端（见文末一节） |
| `scripts/figures.py` | 渲染论文全部插图（450 dpi，Times New Roman） |
| `scripts/figures_paper2.py`、`scripts/tables.py` | 复合面板与结果表 |
| `results/` | **由 `run_all.bat` 生成**（Step 1-11），发布包不带 |
| `outputs/attrib13/` | **由 Step 9 生成**（13 数据集 × 12 格） |
| `outputs/c20_transfer_4donor/` | **由 Step 4 生成**（48 个跨数据集迁移单元） |
| `scripts/verify_numbers.py` | 用仓库内的表格重算论文引用的每一个数字 |
| `docs/`、`data/` | 方法说明与槽位映射 CSV |
| `tests/` | 单元测试 |
| `archive/single_dataset_full_benchmark.py` | 单数据集流水线通路——与论文相互独立，说明见文末 |
| `Single_dataset_full_benchmark.bat` | 该流水线的一键启动器 |

## 安装

```bat
setup_env.bat            :: 建 .venv、装 requirements.txt、跑 check_env.py
```

联网策略：先直连 PyPI 并测速，**失败或低于 60 KB/s 才切清华源**。
`setup_env.bat check` 只报告不安装；`setup_env.bat mirror` 直接从清华源装；
`setup_env.bat notorch` 跳过 torch（只有 Table 2 的 GNN 那一臂需要它）。

torch 只装**一次**，用 CPU 构建（`--index-url .../whl/cpu`）。它**故意不写进
`requirements.txt`**：PyPI 给 Windows 的默认轮子是 CUDA 版，写进去会导致
“先下 CUDA 版、再下 CPU 版”的重复下载。

## 复现

```bash
pip install -r requirements.txt
python scripts/figures.py            # 论文全部插图
python scripts/build_attrib13.py --tag A --per-dataset   # 归因表
python scripts/tables.py             # 结果表
python scripts/verify_numbers.py     # 校验论文中的每个数字
python tests/run_equivalence.py      # 包与单体流水线的等价性测试
```

## 数据

原始数据**不随仓库分发**：`fetch_datasets.bat` 会从各数据集的官方记录下载并完成预处理
（详见 `LICENSE-DATA`）。`data/` 下的两张表承载了所有"逐文件"决策，因此没有任何脚本写死数据集名：

* `col_mappings.csv` —— 原始列 → 槽位，以及逐文件覆盖（`(none)` 表示对某个文件关闭该槽位）；
* `alias_mapping.yaml` —— 别名 → 文件主干名，以及 `corpus:`、`ceiling:`、`donor:` 标记。

**新增一个数据集**：把 CSV 放进 `data/`，在 `alias_mapping.yaml` 里加别名，
在 `col_mappings.csv` 里加它的列（或覆盖项）即可。

## 许可

代码：MIT。我们的结果表与 `data/` 下两张映射表：CC BY 4.0。
**不随仓库再分发任何第三方数据**：请运行 `fetch_datasets.bat` 自行获取。
每个数据集保留其原始记录的许可；`LICENSE-DATA` 载有逐来源许可表、每一条的证据与各自的预处理要求。

## R² 区间的范围

`0.416–0.848` 覆盖的是**能计算噪声上限**的 11 个数据集。
13 个数据集整体区间为 `0.416–0.868`：多出的两个文件（`rxn_yields_SM`、`Summit_CN`）
没有噪声上限，因此不进入 capture ratio。

## 数字校验

`python scripts/verify_numbers.py` 会用 `results/` 下的表格重算 67 个数字，并与论文中打印的数值比较；
任何不一致都会以非零状态退出。早期流水线版本产出的表格**刻意不随仓库发布**：
两套数字并排放置，会让读者复现出论文并未引用的值。

## 复现已发表的表格

`python scripts/build_grid_ceiling.py` 会从 `data/*.csv` 重建 `results/grid13_v18.csv`（156 个单元）、
`results/table1_v18_final.csv`（表 1）与 `results/dataset_index.csv`。
与论文随附的版本（2026-10-27 运行，4 线程）对比：

| 量 | 复现情况 |
|---|---|
| `n_features`（156 个单元） | 156 / 156 完全一致 |
| `n_rows`、`mean_yield`（13 个数据集） | 13 / 13 完全一致 |
| 噪声上限 `R2max`（11 个数据集） | 11 / 11 完全一致（< 1e-15） |
| `n_cond`、`n_rep_groups`、`rep_frac` | 13 / 13 完全一致 |
| `r2_test`（156 个单元） | 120 个逐位一致，149 个在 1e-3 以内 |
| `r2_best`（13 个数据集） | 11 / 13 完全一致 |

**线程数。** 这里的一切都是四线程结果：冻结流水线在其头部固定四线程，
而四线程正是能复现 shipped grid 的设置（156 个单元中 120 个逐位一致；八线程只有 105 个，
XGBoost 39 个中 0 个）。`scripts/protocol.py` 默认固定四线程
（`CD_THREADS=N` 可改，`CD_THREADS=0` 表示不改动线程池）。
唯一例外是 `results/protocol_cv.csv`，那是一张八线程表——其生产脚本固定 8，
文件内也记录 `n_jobs = 8`——因此 `scripts/build_protocols.py` 自身固定 8。
冻结流水线的头部声称"线程数不改变任何已报告的数字"，但**对 XGBoost 与 MLP 并不成立**，
上面的 4 vs 8 一行即为反例。

**残差来自特征矩阵，而非机器。** 18 个非逐位一致的单元全部落在两个 AI_RCP 文件上，
且其中包含**对线程不敏感**的模型（Random Forest 4.9e-4、Gradient Boosting 3.0e-4），
因此线程数无法解释它们；把 `n_jobs` 从 1 扫到 32 也复现不出该数据集的 XGBoost 单元。
这张 grid 写于 2026-10-25 16:11，早于 19:03–20:53 的 equivalents/槽位重构，
所以它是在一个**仓库中已不复存在**的特征矩阵上拟合的；今天的 13 列集合与
equivalents 之前的 9 列集合都试过，均无法复现。
这也解释了为什么 `results/protocol_cv.csv`（2026-10-22）、`results/enrichment13.csv`（2026-10-24）
与 `outputs/c20_transfer_4donor/transfer_matrix.csv`（2026-10-24）能复现各自的协议却复现不出数字，
而 21:13 写出的归因表可以**精确**复现。`docs/NUMBERS_PROVENANCE.md` 逐条列出了每个产物的日期与结论。

`scripts/build_enrichment.py`、`scripts/build_transfer.py` 与 `scripts/build_protocols.py`
会打印各自计算出的数字；仓库内的表保留的是论文值，并由 `scripts/verify_numbers.py` 校验这些值。

---

## 第二套流水线：单数据集端到端（与论文结果相互独立）

`archive/single_dataset_full_benchmark.py` 是第二套**自成体系的流水线通路**。
它与论文的表格互不依赖：输入**一份数据集**，输出属于它自己的证据包
（图 + CSV，每次约 60 个文件），落在 `output_singledataset/<时间戳>_<数据集名>/`。
复现论文**不需要**它，它产出的任何数字也**不会被论文引用**。

```bat
Single_dataset_full_benchmark.bat "(Summit)Reizman_Suzuki_385.csv"
```

一次运行能得到：

* **12 格基准** —— 在本数据集上训练每一个（架构 × 编码）单元，按留出 R² 排名，并打印成带序号的表格；
* **交互选题**（7 个问题，全部支持回车取默认）—— 决定画哪些单元、以及哪些成员组成集成；
  这个选择同时驱动图 2–4、图 7–8 **和**贝叶斯集成，因此二者不可能互相矛盾；
* **归因分析** —— 置换重要度（训练集 vs 留出集）、SHAP 依赖图与交互图，逐个成员输出；
* **虚拟库排序** —— 在试剂轴上穷举枚举，**每条用量轴展开为 `DOSE_POINTS`（默认 10）个点**，
  按期望改进（EI）排序；
* **用量-响应曲线** 以 PDF 输出，每个 top 候选一条曲线；
* **自审计输出** —— 噪声上限、校准斜率、留出覆盖率，以及当"条件审计本身不可信"时的显式告警。

它与论文流水线共用 `hte/` 包（槽位、网格、模型、特征），但有独立入口、独立输出，
且对已发表的数字没有任何影响。

### 结果文件夹里有什么

一次运行会在 `output_singledataset/<YYYYMMDD_HHMMSS>_<数据集名>/` 下写出**约 60 个文件**。
命名规则统一：`Figure<N>_<英文名>[-<架构 (编码)>].<png|csv|pdf>`，因此**按文件名就能看出
是哪张图、由哪个（模型 × 编码）单元产出**。

| 文件（模式） | 数量 | 内容 |
|---|---|---|
| `Figure1_Model_Comparison.png` | 1 | 四面板：R²（带噪声上限虚线）、RMSE、累计误差、5 折 CV |
| `Figure2_ActiveLearning_ResponseSurface- <单元>.png` | 每选中单元 1 张 | 该单元的学习/响应面 |
| `Figure3_Model_Diagnostics_LearningCurve_Residuals- <单元>.png` | 同上 | 学习曲线与残差诊断 |
| `Figure4_Feature_Importance_Attribution- <单元>.png` + `_panel_data.csv` | 同上 | 置换重要度（4a）、灵敏度（4b）、训练 vs 留出（4c）、预测-实测分布（4d） |
| `Figure5_Comprehensive_Substrate_and_Solvent_Analysis.png` + `_panel_data.csv` | 1 | 小提琴分布、底物×催化剂、催化剂×底物热图、SMILES PCA |
| `Figure6_Direct_Ensemble_Ranking.png` + `_top500.csv` + `_enrichment_audit.csv` | 3 | 真实反应的集成排序、Top-20 表、富集审计 |
| `Figure7_Deep_SHAP_Attribution- <单元>.png` + `_panel_data.csv` | 每选中单元 1 张 | SHAP 依赖、SHAP 交互、力分析、全局归因 |
| `Figure8_Partial_Dependence_Plots- <单元>.png` + `_panel_data.csv` | 同上 | 偏依赖三条曲线 |
| `Figure9_Bayesian_Optimization-Ensemble.png` + `_panel_data.csv` | 2 | 收敛曲线、预测热图、第二组交互、用量优化曲线 |
| `Figure10_Top20_Predicted_Conditions-Ensemble.png/.csv` | 2 | 按预测产率排序的 Top-20 条件（含各成员预测、置信区间） |
| `Figure10b_Top20_ByExpectedImprovement-Ensemble.png/.csv` | 2 | 按 **EI** 排序的 Top-20（与 10a 回答不同问题） |
| `Figure11_Loading_Curves_Top<N>.pdf/.csv` | 2 | 每个 top 候选一条用量-响应曲线（矢量 PDF） |
| `grid12_metrics.csv` | 1 | **12 格的完整指标**（模型、编码、帧、特征数、训练/留出 R²、RMSE、耗时、状态） |
| `model_evaluation_metrics.csv` | 1 | 选中成员的 CV / 学习曲线 / 置换重要度汇总 |
| `model_predictions_all.csv` | 1 | 逐行预测（真实 + 留出），供二次分析 |
| `timings.csv` | 1 | 分项耗时（训练 / SHAP / PDP / 扫描 …） |
| `choices.json` | 1 | **本次的 7 个选择**，可用于复现同一组图 |

> 两点说明：①每张图旁的 `_panel_data.csv` 是该图**画图用的原始数值**，便于核对，不必从像素里猜；
> ②`choices.json` 可直接重放——把它放进 `CD_CHOICES` 环境变量再跑一次，就会得到同一组图。
