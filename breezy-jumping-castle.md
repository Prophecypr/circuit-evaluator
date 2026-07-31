# 手绘电路图 LLM 评价方案

## Context
项目目标是：手绘电路图 → 图像检测（提取连接关系+元件数值）→ **LLM评价** → 评分/反馈。
当前处于设计阶段，尚无代码。用户收集了6篇LLM相关论文，需要评估这些论文的参考价值，并设计LLM评价的技术方案。

## 论文价值评估

| 论文 | 相关度 | 可借鉴点 |
|------|--------|----------|
| MLLM-as-a-Judge | ⭐⭐⭐⭐⭐ 最核心 | LLM做裁判的三种模式（Scoring/Pair/Ranking）、偏见分析、多次评估取均值 |
| EQUATOR | ⭐⭐⭐⭐ | 确定性评分框架、二元打分、向量数据库+标准答案比对、小LLM做评估器 |
| SmartCourse | ⭐⭐⭐ | Prompt设计（注入上下文）、自定义多维指标（PlanScore/PersonalScore） |
| NL-to-SQL Benchmark | ⭐⭐⭐ | 多维度评价体系（合规率/幻觉率/一致性）、结构化输出要求 |
| AIoT Voice Assistant | ⭐⭐ | Pipeline架构参考、结构化JSON输出约束 |
| Local LLMs Programming | ⭐ | 方法论参考，与电路评价任务关联不大 |

**结论**：论文非常有用。MLLM-as-a-Judge + EQUATOR 给出了两条互补路线——主观语义评价（LLM打分）和客观匹配评价（向量相似度）。建议两条路线结合使用。

## 整体架构

```
[手绘电路图] → [CV检测模块] → [结构化电路描述JSON] → [LLM评价模块] → [评分+反馈]
                                        │
                                        ├── 路线A: 确定性匹配（有标准答案时）
                                        │    └── 图匹配/数值误差/余弦相似度
                                        │
                                        └── 路线B: LLM-as-a-Judge（语义评价）
                                             └── 多维度打分 + 配对比较 + 评语
```

## 实现计划

### Phase 1: 定义数据格式（电路描述的标准JSON结构）

定义两种JSON Schema：
1. **检测结果JSON** — CV模块产出，描述从手绘图提取的电路信息
2. **标准答案JSON** — 用于对比的正确答案（教师提供或预先定义）

```json
// 检测结果示例
{
  "circuit_id": "001",
  "components": [
    {"id": "R1", "type": "resistor", "value": "10kΩ", "confidence": 0.92},
    {"id": "V1", "type": "voltage_source", "value": "5V", "confidence": 0.88}
  ],
  "connections": [
    {"from": "V1.positive", "to": "R1.pin1", "type": "wire"},
    {"from": "R1.pin2", "to": "GND", "type": "wire"}
  ],
  "topology": "series_circuit"
}
```

### Phase 2: 实现路线A — 确定性评价

参考 EQUATOR 的做法，适用于有标准答案的场景。

**文件**：`evaluation/deterministic_eval.py`

**核心逻辑**：
1. 标准化元件名称后做集合匹配（元件检测召回率/精确率）
2. 连接关系图匹配（改自图编辑距离，降低计算复杂度用邻接矩阵比对）
3. 数值误差计算（考虑单位换算：1kΩ vs 1000Ω）
4. 输出：`{recall, precision, f1, value_mape, topology_match}`

**关键决策**：
- 不用向量数据库（EQUATOR的做法），因为电路结构天然适合图/集合比较
- 数值匹配需要模糊容差（±5%或±10%）

### Phase 3: 实现路线B — LLM-as-a-Judge

参考 MLLM-as-a-Judge 的 Scoring + Pair Comparison 模式。

**文件**：`evaluation/llm_judge.py`

**Prompt设计**（参考SmartCourse的做法——注入完整上下文）：

```
System: 你是电路分析专家。请根据以下标准评价学生手绘电路的检测结果。
  评分维度：
  1. 元件识别完整性 (0-10): 是否遗漏或多余元件
  2. 连接关系正确性 (0-10): 连线是否正确
  3. 数值识别准确性 (0-10): 标注数值是否准确
  4. 电路拓扑合理性 (0-5): 整体电路逻辑是否合理
  总分: 35分

  输出JSON: {"scores": {...}, "total": N, "feedback": "...", "errors": [...]}

User: 
  标准答案: {ground_truth_json}
  检测结果: {detection_result_json}
  请逐项评分并给出反馈。
```

**关键设计选择**：
- 使用本地LLM（Ollama + llama3.1:8b / qwen2.5:7b），参考多篇论文均验证了7-8B模型在结构化评价任务上的可行性
- 要求JSON输出，便于程序解析（参考AIoT论文的tool-call约束思路）
- 每条评价重复3次取中位数（参考MLLM-as-a-Judge的做法）

### Phase 4: 评估器主控模块

**文件**：`evaluation/evaluator.py`

整合路线A和路线B：
```python
def evaluate(detection_result, ground_truth=None, use_llm=True):
    result = {}
    if ground_truth:
        result["deterministic"] = deterministic_eval(detection_result, ground_truth)
    if use_llm:
        result["llm_judge"] = llm_evaluate(detection_result, ground_truth)
    return result
```

### Phase 5: 评价指标仪表盘

**文件**：`evaluation/metrics.py`

多维度汇总（参考NL-to-SQL论文的7指标框架）：
- 元件级：Precision / Recall / F1
- 数值级：MAPE（平均绝对百分比误差）
- 连接级：连接正确率
- LLM级：各维度得分 + 一致性（多次评估的方差）
- 综合：加权总分

### Phase 6: 验证方案

1. 构造5个测试电路（简单串联、并联、混联、含错误、含遗漏）
2. 对每个电路准备标准答案JSON + 模拟检测结果（含正确和带噪声的版本）
3. 验证：
   - 确定性评价能否区分"正确检测"和"错误检测"
   - LLM评价与确定性评价的一致性
   - LLM重复评估的一致性（3次的标准差）

## 关键风险与应对

| 风险 | 应对 |
|------|------|
| LLM幻觉（编造不存在的元件） | 要求LLM只评价检测结果中存在的元件，不自行推断 |
| LLM偏见（偏好特定格式） | 多次评估取均值；Prompt中强调客观性 |
| 图匹配复杂度高 | 先用集合匹配筛掉明显错误的，再对疑似正确的做精确图比较 |
| 数值单位不统一 | 预处理阶段统一单位（parse到SI基本单位） |

## 文件清单

```
evaluation/
├── __init__.py
├── schema.py              # 电路JSON的dataclass定义
├── deterministic_eval.py  # 路线A：基于规则/结构的确定性评分
├── llm_judge.py          # 路线B：LLM-as-a-Judge
├── evaluator.py           # 主控，整合A+B
├── metrics.py             # 指标计算与汇总
├── prompts.py             # LLM prompt模板
└── test_evaluation.py     # 验证脚本
```
