# Wiring Edge 错误归因工具设计

日期：2026-08-10

## 1. 目标

在不读取最终 42 张封存手绘测试图、不调用 LLM、不重新执行 YOLO/OCR 的前提下，利用现有 50 张开发集的人工连接 GT、`strict_jj` 预测结果和 wiring trace，把 wiring edge 的 FP/FN 自动归因到可操作的根因类别。

下一轮算法改动以 wiring edge micro-F1 为主指标。在 Precision 不明显下降的条件下优先提高 Recall。错误归因只决定下一轮优先改进对象，不直接改变模型或连接阈值。

## 2. 非目标

- 不在本阶段训练新的神经网络或连接边分类器。
- 不根据最终 42 张测试图选择阈值或修改算法。
- 不重新评估 OCR、数值识别或 LLM 评分。
- 不把每一条由错误并网派生的 FP 都错误解释为独立根因。
- 不修改人工 GT。

## 3. 输入及数据边界

默认输入目录为：

`results/wiring_reliability_full_20260810_merged/`

使用以下内容：

- `predictions/strict_jj/*.json`：50 张最佳配置预测；
- `wiring_traces_strict_jj/*.json`：候选连接接受/拒绝证据；
- `benchmark/result/*_gt.txt`：人工端口连接 GT；
- `benchmark/detections/*.json`：组件、端口标签和坐标；
- benchmark 对应的 50 张源图：只用于绘制诊断叠加图。

运行前必须验证：图像 stem 完全一致、数量恰好为 50、预测使用 `strict_jj`、LLM 已跳过、失败数为 0。任一条件不满足时停止，不生成容易误导的部分报告。

## 4. 方案选择

采用基于 GT 和 wiring trace 的规则归因方案。相比纯人工逐图分类，它可重复、可统计；相比直接训练连接边分类器，它不需要先构建新的错误标签数据集，并能先识别当前规则系统的主要失败模式。

人工复核只用于检查最差 10 张可视化和验证自动类别是否合理，不作为主统计来源。

## 5. 数据流

1. 使用与 `run_experiments.evaluate()` 相同的 IoU 组件匹配和端口身份映射，分别构建 GT 网络和预测网络。
2. 将每个网络展开成无向端口对集合，计算 TP、FP 和 FN。
3. 为每条 FP/FN 收集端点组件、端口坐标、预测网络、GT 网络及相关 wiring trace 事件。
4. 先识别网络级错误合并和网络拆分，再对根因边做局部归因。
5. 输出边级表、图像级表、类别汇总、最差 10 张诊断图和 Markdown 报告。

## 6. FN 归因优先级

每条 FN 只分配一个主类别，按以下顺序判断：

1. `component_unmatched`：GT 端点对应组件未被当前预测组件匹配。
2. `port_unmatched`：组件已匹配，但对应端口身份或端口索引无法映射。
3. `no_port_skeleton`：端口搜索半径内没有可用骨架起点。
4. `skeleton_break`：存在骨架起点，但在最大追踪范围内没有到达 GT 网络的节点或另一端口。
5. `candidate_rejected`：trace 中存在指向正确网络的候选，但被 `no_skeleton_path`、`crosses_component`、`would_short_component` 或 `ambiguous_crossing` 拒绝。
6. `candidate_not_generated`：组件和端口均可映射，但 trace 中不存在指向正确网络的候选。
7. `network_split_unresolved`：证据不足以定位更具体局部原因，但 GT 网络被预测拆分。

记录 `secondary_reason` 保存更细的拒绝原因，主类别统计保持互斥。

## 7. FP 归因优先级

先比较预测网络与 GT 网络的重叠关系。如果一个预测网络包含来自两个或更多 GT 网络的端口，将其标记为 `network_merge`，并在该预测网络内寻找最早或置信证据最弱的跨 GT 接受事件作为 `root_event`。

该根因事件按下列类别进一步分类：

1. `wrong_port_to_junction`：错误 P2J 把端口接入其他 GT 网络。
2. `wrong_port_to_port`：错误 P2P/LOS/close-port 连接两个不同 GT 网络。
3. `wrong_junction_merge`：错误 JJ 合并两个不同 GT 网络。
4. `crossing_ambiguity`：错误连接经过无连接点交叉位置或由交叉语义产生。
5. `component_crossing`：接受边穿过其他元器件包围盒。
6. `unattributed_merge`：能够确认网络误合并，但 trace 不能唯一定位根因事件。
7. `local_false_edge`：没有形成多网络合并的孤立 FP。

同一错误预测网络中，由根因合并产生的其余端口对标记为 `cascade_fp`，并通过 `root_event_id` 指向根因。全局报告同时给出 edge FP 数与 root error 数，算法优先级按 root error 数和造成的 cascade FP 数综合排序。

## 8. 输出

默认输出目录：

`results/wiring_error_attribution_20260810/`

包含：

- `edge_errors.csv`：每条 FP/FN、主类别、次级原因、端点、网络标识和根因事件；
- `image_summary.csv`：每张图的 TP/FP/FN/F1、根因数量及首要类别；
- `category_summary.json`：按错误类别汇总的 edge 数、root 数、图像数和占比；
- `annotated_worst10/*.jpg`：最差 10 张诊断图；
- `wiring_error_report.md`：全局结论和下一轮改进建议；
- `run_metadata.json`：输入路径、代码版本、输入文件哈希、图像数和生成时间。

诊断图颜色固定为：TP 绿色、FP 橙色、FN 红色、被判定为根因的连接或断点紫色。图例必须直接画在图内。

## 9. 实现边界

新增独立模块处理端口身份映射、网络展开、根因归因与统计。现有 `run_experiments.evaluate()` 复用该模块的基础边集合函数，避免实验指标与诊断工具出现两套定义。渲染逻辑保持独立，归因核心不依赖 OpenCV，便于单元测试。

命令行提供明确输入、输出参数和 `--expected-count 50`。默认拒绝覆盖非空输出目录；只有显式 `--resume` 且元数据与输入哈希一致时允许继续。

## 10. 测试策略

单元测试覆盖：

- 完全正确网络，无错误输出；
- 单个 FN 的七类归因；
- P2J、P2P 和 JJ 根因造成的网络误合并；
- 一个根因产生多条 `cascade_fp`；
- 同一物理网络中端口顺序变化不影响边集合；
- 输入缺图、重复 stem、错误配置或不匹配哈希时失败；
- 中文绝对路径下读取和写图正常。

集成测试使用小型合成 benchmark 验证 CSV、JSON、Markdown 和诊断图数量。随后在现有 50 张开发集执行一次正式归因，并人工抽查最差 10 张。

## 11. 完成标准

本阶段完成需同时满足：

1. 50 张输入全部分析，无静默遗漏或失败；
2. 每条 FP/FN 恰好有一个主类别；
3. `edge_errors.csv` 汇总出的 TP/FP/FN 与既有 `strict_jj` 结果 `305/537/1002` 完全一致；
4. 每个 `cascade_fp` 都能指向一个根因，不能定位时明确标记 `unattributed_merge`；
5. 最差 10 张诊断图完成生成并人工抽查；
6. 自动测试和语法检查通过；
7. 报告明确给出占比最高的根因，以及下一轮只改哪一个问题。

## 12. 后续阶段

错误归因完成后进入新的第二步：针对占比最高且可修复的根因设计一个最小算法改动。先在最差约 10 张开发图上迭代，通过后再跑完整 50 张对照实验。只有完整开发集结果满足 F1 提升且 Precision 不明显下降，才锁定代码和阈值并启封最终 42 张测试图。
