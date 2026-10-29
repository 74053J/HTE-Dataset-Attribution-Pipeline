# 论文数字同步表（2026-10-29 全流程重算）

- 来源：`run_all.bat` 全流程实跑（Step 1–11），环境 Python 3.10.2 / **xgboost-cpu 3.2.0** / shap 0.49.1 / 4 线程
- 对照基准：`repo_v2/results/*`（论文随附表）
- 门禁脚本：`scripts/verify_numbers.py` → **63 项，53 通过 / 10 变化**
- 原始解析：`verify_checks_parsed.tsv`；门禁日志：`verify_paper.log`

## 1. 需要改到论文里的 10 个数字

| 位置 | 量 | 论文值 | 本轮值 |
|---|---|---|---|
| [2] accuracy, ceiling and capture (sections 3.2 and 4) | best R2 max (all 13) | 0.867 | 0.8684 |
| [3] attribution shares and first-rank counts (section 3.3, figure 3) | share reactant_1 | 17.3 | 17.3360 |
| [3] attribution shares and first-rank counts (section 3.3, figure 3) | share reactant_2 | 16.69 | 16.5319 |
| [3] attribution shares and first-rank counts (section 3.3, figure 3) | share ligand | 8.8 | 8.8461 |
| [3] attribution shares and first-rank counts (section 3.3, figure 3) | first-rank reactant_1 | 49 | 50 |
| [4] encoding pairs (section 3.3) | rho median | 0.85 | 0.8455 |
| [4] encoding pairs (section 3.3) | L1 median (percentage points) | 26.09 | 27.0450 |
| [9] the remaining published tables | grid_stats encoding_effect gap | 0.165877 | 0.1683 |
| [9] the remaining published tables | grid_stats mlp_int_vs_xgb gap | -0.242452 | -0.2450 |
| [9] the remaining published tables | grid_stats histgb_vs_xgb gap | -0.050768 | -0.0512 |

## 2. 全部 63 项（当前值 vs 论文值）

| 章节 | 检查项 | 论文值 | 本轮值 | 变化 |
|---|---|---|---|---|
| [1] dataset inventory (Table 1) | datasets | 13 | 13 |  |
| [1] dataset inventory (Table 1) | datasets with a ceiling | 11 | 11 |  |
| [1] dataset inventory (Table 1) | model-dataset cells | 156 | 156 |  |
| [2] accuracy, ceiling and capture (sections 3.2 and 4) | best R2 min (all 13) | 0.417 | 0.4161 |  |
| [2] accuracy, ceiling and capture (sections 3.2 and 4) | best R2 max (all 13) | 0.867 | 0.8684 | **改** |
| [2] accuracy, ceiling and capture (sections 3.2 and 4) | best R2 max (11 with ceiling) | 0.848 | 0.8482 |  |
| [2] accuracy, ceiling and capture (sections 3.2 and 4) | ceiling min | 0.576 | 0.5755 |  |
| [2] accuracy, ceiling and capture (sections 3.2 and 4) | ceiling max | 0.971 | 0.9711 |  |
| [2] accuracy, ceiling and capture (sections 3.2 and 4) | capture mean (%) | 86.35 | 86.3456 |  |
| [2] accuracy, ceiling and capture (sections 3.2 and 4) | capture median (%) | 87.34 | 87.3376 |  |
| [2] accuracy, ceiling and capture (sections 3.2 and 4) | rho(ceiling, R2) | 0.709 | 0.7091 |  |
| [2] accuracy, ceiling and capture (sections 3.2 and 4) | rho(n_rows, R2) | -0.1 | -0.1000 |  |
| [3] attribution shares and first-rank counts (section 3.3, figure 3) | share reactant_1 | 17.3 | 17.3360 | **改** |
| [3] attribution shares and first-rank counts (section 3.3, figure 3) | share reactant_2 | 16.69 | 16.5319 | **改** |
| [3] attribution shares and first-rank counts (section 3.3, figure 3) | share temperature | 11.2 | 11.1859 |  |
| [3] attribution shares and first-rank counts (section 3.3, figure 3) | share base | 10.53 | 10.5365 |  |
| [3] attribution shares and first-rank counts (section 3.3, figure 3) | share ligand | 8.8 | 8.8461 | **改** |
| [3] attribution shares and first-rank counts (section 3.3, figure 3) | share catalyst_eq | 2.85 | 2.8376 |  |
| [3] attribution shares and first-rank counts (section 3.3, figure 3) | share solvent_eq | 0.006 | 0.0085 |  |
| [3] attribution shares and first-rank counts (section 3.3, figure 3) | first-rank reactant_1 | 49 | 50 | **改** |
| [3] attribution shares and first-rank counts (section 3.3, figure 3) | first-rank reactant_2 | 46 | 46 |  |
| [3] attribution shares and first-rank counts (section 3.3, figure 3) | first-rank temperature | 30 | 30 |  |
| [3] attribution shares and first-rank counts (section 3.3, figure 3) | first-rank additive | 11 | 11 |  |
| [4] encoding pairs (section 3.3) | integer vs one-hot pairs | 65 | 65 |  |
| [4] encoding pairs (section 3.3) | pairs sharing the top slot | 35 | 35 |  |
| [4] encoding pairs (section 3.3) | rho median | 0.85 | 0.8455 | **改** |
| [4] encoding pairs (section 3.3) | L1 median (percentage points) | 26.09 | 27.0450 | **改** |
| [5] permutation vs SHAP (section 3.3) | perm-vs-SHAP mean | 0.899 | 0.8967 |  |
| [5] permutation vs SHAP (section 3.3) | perm-vs-SHAP median | 0.9273 | 0.9273 |  |
| [5] permutation vs SHAP (section 3.3) | cells | 156 | 156 |  |
| [6] cross-dataset transfer (section 3.5) | transfer cells | 48 | 48 |  |
| [6] cross-dataset transfer (section 3.5) | transfer cells negative | 48 | 48 |  |
| [6] cross-dataset transfer (section 3.5) | transfer worst | -4.94 | -4.9403 |  |
| [6] cross-dataset transfer (section 3.5) | transfer best | -0.1 | -0.0813 |  |
| [7] aggregate ledger agrees with the tables | json spearman_rho | 0.7090909090909092 | 0.7091 |  |
| [7] aggregate ledger agrees with the tables | json median_capture | 0.8733756028758433 | 0.8734 |  |
| [7] aggregate ledger agrees with the tables | json mean_capture | 0.8634563066707177 | 0.8635 |  |
| [7] aggregate ledger agrees with the tables | json best_r2.min | 0.4160983095467068 | 0.4161 |  |
| [7] aggregate ledger agrees with the tables | json best_r2.max | 0.8683665208571499 | 0.8684 |  |
| [7] aggregate ledger agrees with the tables | json rho(n_rows, R2) | -0.10000000000000002 | -0.1000 |  |
| [9] the remaining published tables | grid_stats rows | 6 | 6.0000 |  |
| [9] the remaining published tables | grid_stats encoding_effect gap | 0.165877 | 0.1683 | **改** |
| [9] the remaining published tables | grid_stats mlp_int_vs_xgb gap | -0.242452 | -0.2450 | **改** |
| [9] the remaining published tables | grid_stats histgb_vs_xgb gap | -0.050768 | -0.0512 | **改** |
| [9] the remaining published tables | enrichment rows | 13 | 13.0000 |  |
| [9] the remaining published tables | enrichment mean ef_10 | 3.1977 | 3.1976 |  |
| [9] the remaining published tables | bootstrap rows | 26 | 26.0000 |  |
| [9] the remaining published tables | protocol rows | 3 | 3.0000 |  |
| [9] the remaining published tables | protocol random mean R2 | 0.757918 | 0.7579 |  |
| [9] the remaining published tables | protocol substrate-disjoint mean R2 | -0.151465 | -0.1515 |  |
| [9] the remaining published tables | protocol ordered mean R2 | -0.79944 | -0.7994 |  |
| [9] the remaining published tables | graph_baseline rows | 20 | 20.0000 |  |
| [9] the remaining published tables | novel_pairs rows | 5 | 5.0000 |  |
| [9] the remaining published tables | novel_pairs xgb_native R2 | 0.58335 | 0.5833 |  |
| [9] the remaining published tables | unseen_reactants rows | 5 | 5.0000 |  |
| [9] the remaining published tables | unseen_reactants n_unseen | 5829 | 5829.0000 |  |
| [9] the remaining published tables | bo_panel rows | 1113 | 1113.0000 |  |
| [9] the remaining published tables | attrib13_long rows | 1392 | 1392.0000 |  |
| [9] the remaining published tables | attrib13_long cells | 156 | 156.0000 |  |
| [8] held-out real chemistry (section 3.4) | heldout k=1 ranking | 95.0 | 95.0000 |  |
| [8] held-out real chemistry (section 3.4) | heldout k=1 random | 12.3 | 12.3000 |  |
| [8] held-out real chemistry (section 3.4) | heldout rho(sd,err) | 0.115 | 0.1153 |  |
| [8] held-out real chemistry (section 3.4) | heldout coverage 90 | 47.3 | 47.3091 |  |

## 3. 由这 10 个变化连带改写的正文表述

| 原文表述 | 建议改成 |
|---|---|
| R² 区间 `0.417–0.867`（13 个数据集） | `0.416–0.868` |
| R² 区间 `0.417–0.848`（11 个有上限的数据集） | `0.416–0.848` |
| “49 / 46 / 30 / 11 of 156 cells” | “50 / 46 / 30 / 11 of 156 cells” |
| “median ρ = 0.850, median L1 = 26.1 pp” | “median ρ = 0.846, median L1 = 27.0 pp” |
| “perm-vs-SHAP mean +0.90” | “+0.897”（median 0.9273 不变） |
| 归因占比 “17.31 / 16.69 / 11.20 / 10.53 / 8.80” | “17.34 / 16.53 / 11.19 / 10.54 / 8.85” |
| `grid_stats` 三个 gap：0.165877 / −0.242452 / −0.050768 | 0.1683 / −0.2450 / −0.0512 |

## 4. 已在仓库内同步的文件

- `README.md` / `README.zh-CN.md`：R² 区间 → 0.416–0.848 / 0.416–0.868
- `docs/NUMBERS_AND_PIPELINE.md`：归因占比、首位计数、median ρ / L1、perm-vs-SHAP 均值
- `COMPARISON_vs_paper_20261029.md`：完整台账（含两轮：修改前 / 按论文口径重算后）

## 5. 没有同步的部分（需要你提供文件）

论文正文（`.docx`）不在本仓库内。工作区里能查到的最新稿是
`_work\v247\v2.30.before_figfix.docx`（10-25 22:28），而 `CILS-格式包-20261026` 的评审记录提到过 v3.19——
**v3.x 正文不在这台机器的工作区里**。把当前正文的路径给我，我可以按上表逐条替换并生成修订版（保留原文件）。

## 6. 已应用到论文正文（2026-10-29 22:15）

- 原稿：`D:\Documents\1Study\HKSchoolApply\科研项目\论文-英文版-A4-v3.46-CILS投稿版.docx`（**未改动**）
- 新稿：`…\论文-英文版-A4-v3.47-numbers-synced-20261029.docx`（**1,570 KB**）
- 清单：`…\论文-v3.47-数字同步改动清单.txt`

实际改动：**9 个段落 / 15 处数字 + Table 1 的 7 个单元格**（逐条见清单）。
校验：新值全部出现（17.34% / 16.53% / 8.85% / 11.19% / 10.54% / 2.84% / 50 of 156 (32%) /
0.846 / 27.0 pp / +0.897 / (0.868) / 0.416–0.848），旧值 11 项全部清除。

**未改**：§3.3 段 55 的 `grid_stats` 文字（0.0518/0.0314/0.0147/0.2327/0.161 与 gate 的
0.0512/−0.2450/0.1683 不是同一组定义）；Table 2 结构基线（Doyle 修复改变了条件特征，
需重跑第 6 步后逐格核对）。
