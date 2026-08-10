# CRNN-v2 手绘电路数值 OCR 设计

## 1. 背景与已验证问题

当前主流水线仍使用 CRNN，固定加载 `runs/ocr_crnn_machine/best.pt`。现有权重在不同数据域上的实测结果如下：

| 权重 | 验证域 | Exact Match | CER |
|---|---|---:|---:|
| `runs/ocr_crnn/best.pt` | Digitize-HCD 原域 | 89.01% | 4.83% |
| `runs/ocr_crnn/best.pt` | CGHD 手绘域 | 4.34% | 78.77% |
| `runs/ocr_crnn_machine/best.pt` | CGHD 手绘域 | 2.11% | 107.37% |
| `runs/ocr_crnn_machine/best_old_cghd.pt` | CGHD 手绘域 | 17.00% | 71.11% |

这证明 CRNN 结构可以识别手绘数值，但现有权重存在训练域、字符集、模型结构和验证集不一致问题。历史 `val_loss` 不能代替在当前独立验证集上的 Exact Match 和 CER。

## 2. 目标

训练一个面向手绘电路与手机拍摄图片的 `CRNN-v2-handwritten` 数值识别模型，在不使用 LLM、不污染 42 张最终测试图的前提下，提高元器件数值串的完整正确率。

首轮成功门槛：

- 按绘图者隔离的 CGHD 验证集：Normalized Exact Match ≥ 80%。
- 同一验证集：Normalized CER ≤ 10%。
- `Ω`、`μF`、`nF`、`kΩ`、`MΩ`、`V`、`H`、`A`、`Hz` 主要单位分组准确率全部报告。
- 新模型必须明显优于现用 `machine` 权重，否则不接入主流水线。

## 3. 范围与非目标

本阶段包含：

- 手绘电路数值裁剪的数据重建、清洗、分组划分和增强。
- 统一 CRNN 结构、字符集、检查点元数据和推理预处理。
- Kaggle GPU 训练包、独立验证脚本和可复现报告。
- 新旧权重的公平对比与可回滚接入。

本阶段不包含：

- LLM 评价或 LLM 辅助 OCR。
- 使用 42 张最终测试图的图片、裁剪、GT 或错误分析进行训练、调参或模型选择。
- IC 型号等大字符集文本识别；该任务继续由 Tesseract 或后续专用模型处理。
- 立即替换为 SVTR、PARSeq 或 TrOCR。只有 CRNN-v2 未达门槛时才启动架构比较。

## 4. 数据设计

### 4.1 数据来源

- CGHD v16：`E:\circuit_image\cghd-zenodo-16`，当前有 33 个 `drafter_*` 目录。
- Digitize-HCD 历史裁剪：`data/ocr_training`，11,642 张。
- 合成稀有单位样本：只用于补足 `μ`、`Ω`、小数点和工程前缀的覆盖，必须记录生成种子和参数。

42 张用户手绘图只是最终外部测试集，不得出现在任何训练或验证清单中。

### 4.2 重建与溯源

重新从 CGHD XML 提取裁剪，每条样本保留：

- `sample_id`
- `drafter_id`
- `source_image`
- `source_xml`
- `bbox`
- `raw_label`
- `normalized_label`
- `source_sha256`
- `crop_sha256`
- `split`

不再仅使用无溯源的 `cghd_000001.png` 文件名。

### 4.3 划分规则

- 以 `drafter_id` 为最小隔离单位，同一绘图者不得同时出现在训练集和验证集。
- 划分名单固定写入配置文件，种子固定为 42。
- Digitize-HCD 因缺少原始绘图者溯源，只加入训练集，不作主验证集。
- 主验证集只使用未见 CGHD 绘图者，防止随机裁剪划分造成书写风格泄漏。

### 4.4 泄漏防护

- 训练前比对原图和裁剪哈希，确保 42 张测试图及其派生物不在数据包中。
- 训练结束前不读取 `测试集实验_v1/value_gt_template.xlsx` 作为训练决策依据。
- 模型、阈值、字符归一化规则和预处理在最终测试前一次性冻结。

## 5. 标签与字符归一化

模型识别对象是“数值+电气单位”，不是任意文本。

归一化规则：

- `µ` 和 ASCII `u` 归一为 Greek small mu `μ` (`U+03BC`)。
- `K` 在 kilo 语义下归一为 `k`；`M` 保留为 mega，`m` 保留为 milli。
- 欧姆单位统一为 `Ω` (`U+03A9`)。
- 移除数值两端空格，但不删除小数点、正负号和斜杠。
- `3V3 -> 3.3V` 等是语义规范化，在原始 OCR 结果之外另存 `normalized_prediction`，不覆盖原始预测。
- 不符合数值语法且属于 IC 型号的标签从 CRNN-v2 数值训练中排除。

字符集必须由清洗后数据自动验证，配置、训练和推理共用同一份字符集，不得在三处手写不同字符串。

## 6. 模型与训练

### 6.1 结构

- 保留 CNN + 2 层 BiLSTM + CTC 的 CRNN 主体。
- 固定输入高度 32，首轮输入宽度为 160。
- 使用单一、可版本化的预处理实现，训练和推理调用同一函数。
- 检查点必须保存 `model_version`、`architecture`、`chars`、`img_h`、`img_w`、`normalization_version`、`split_id`、`seed`、`epoch`、验证指标和 Git commit。

### 6.2 增强

增强仅用于训练集：

- 轻微旋转、仿射和透视变换。
- 变尺度、左右/上下裁剪抖动和不对称 padding。
- 高斯模糊、轻微运动模糊、JPEG 压缩、传感器噪声。
- 亮度、对比度、局部阴影和纸张背景变化。
- 笔画膨胀、腐蚀和断裂模拟。

增强参数写入 YAML/JSON 配置，不在 notebook 单元格中散落硬编码。

### 6.3 优化

- AdamW + CTC loss。
- 最大 40 epochs，以 Normalized Exact Match 为主指标保存最优模型。
- 早停 patience 为 7。
- 稀有单位类型使用加权采样，但验证指标不加权。
- 固定随机种子 42，启用可复现设置并记录 Kaggle GPU 类型、PyTorch 版本和运行时间。

## 7. Kaggle 训练包

训练包应包含：

- 可单击 Run All 的 Kaggle notebook。
- 训练/验证 manifest 与哈希清单。
- 版本化配置文件。
- 模型、数据集、增强、评估和导出模块。
- 训练日志 CSV、学习曲线、单位分组指标、错误案例 CSV 和混淆统计。
- `best.pt`、`last.pt`、`metrics.json`、`run_config.json`、`environment.txt`。

训练包不包含 API Key、Kaggle Token、42 张测试图或其 GT。

## 8. 评估设计

对每个候选权重报告：

- Raw Exact Match。
- Normalized Exact Match。
- Raw CER。
- Normalized CER。
- 数字部分完全正确率。
- 单位完全正确率。
- 按 `Ω`、`μF`、`nF`、`kΩ`、`MΩ`、`V`、`H`、`A`、`Hz` 分组的样本数和准确率。
- 常见编辑错误和错误样例。

CER 定义为：

```text
CER = (substitutions + deletions + insertions) / number_of_ground_truth_characters
```

CER 可能高于 100%，因为一个短标签可能被插入很多额外字符。

## 9. 接入与回滚

- 新权重写入 `runs/ocr_crnn_hand_v2/best.pt`，不覆盖任何旧权重。
- 主流水线的 OCR 权重路径改为配置项，默认值只在新模型通过门槛后切换。
- 接入前对旧权重、新权重和预处理配置做带时间戳备份。
- 保留一键回滚到原 `runs/ocr_crnn_machine/best.pt` 的能力。
- 接入后先跑独立验证集，不立即跑 42 张最终测试图。

## 10. 异常与完整性检查

以下任一情况必须终止训练：

- manifest 中有不存在的裁剪。
- 训练集与验证集共享 `drafter_id`、`source_sha256` 或 `crop_sha256`。
- 出现字符集以外的未规范化标签。
- 数据包中出现 42 张最终测试图的哈希。
- 检查点字符集、输入尺寸或结构与推理代码不兼容。
- Kaggle 运行未产生完整的配置、指标和环境文件。

## 11. 验收流程

1. 数据完整性和泄漏检查通过。
2. Kaggle notebook 能从空环境 Run All 完成训练和导出。
3. 独立脚本能加载导出的 `best.pt` 并复现 notebook 指标。
4. 新模型通过第 2 节门槛后，备份并接入主流水线。
5. 冻结权重、预处理、后处理和阈值。
6. 最后一次性运行 42 张外部测试图，生成投稿指标。

## 12. 已确认决策

- 用户选择方案 1：保留 CRNN 并重训手绘数值模型。
- 训练平台使用 Kaggle GPU。
- LLM 不参与 OCR 训练、识别或模型选择。
- 42 张测试图保持冻结，不用于训练或调参。
