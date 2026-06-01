# 电路图智能评价系统

自动识别手绘电路图 → 提取元件+数值 → LLM 专业评价。

## 快速开始

```powershell
cd E:\ClaudeCode\电路图智能评价系统

# 评价手绘电路图
python -m src.vision.image_eval test3.jpg

# 文字描述直接评价
python eval.py

# 带 Ngspice 仿真
python eval.py circuit_04_no_r --sim
```

## 目录结构

```
电路图智能评价系统/
├── README.md                   # 本文档
├── eval.py                     # 一键评价入口（文字描述）
├── requirements.txt            # Python 依赖
├── circuit0.sp                 # 标准答案 SPICE 网表
├── CLAUDE.md                   # Claude Code 项目指南
│
├── src/
│   ├── llm.py                  # DeepSeek API 封装
│   │
│   ├── phase1_verify/          # 方案B：端到端 LLM 评价
│   │   ├── evaluate.py         #   文字描述 → LLM → JSON 报告
│   │   └── prompts.py          #   评价 prompt 模板
│   │
│   ├── phase2_pipeline/        # 方案A：结构化流水线
│   │   ├── schema.py           #   Pydantic 数据模型
│   │   ├── converter.py        #   文字 → 结构化 JSON
│   │   ├── safety.py           #   规则安全检查
│   │   ├── correctness.py      #   LLM 功能正确性
│   │   ├── quality.py          #   LLM 质量评分
│   │   └── report.py           #   流水线编排
│   │
│   ├── phase3_spice/           # SPICE 仿真模块
│   │   ├── netlist.py          #   Circuit → SPICE 网表
│   │   └── simulate.py         #   Ngspice 执行
│   │
│   ├── phase4_netlist/         # 方案D：网表优先评价
│   │   └── evaluate.py         #   文字 → SPICE → 仿真 → LLM
│   │
│   └── vision/                 # CV 视觉模块
│       ├── image_eval.py       # ★ 主评价管道：YOLO+CRNN+LLM
│       ├── wire_detect.py      #   连线检测（Hough+Union-Find）
│       ├── train_ocr.py        #   CRNN 手写 OCR 训练/推理
│       ├── train_yolo.py       #   YOLO 元件检测训练
│       ├── convert_cghd.py     #   CGHD→YOLO 格式转换
│       └── visualize.py        #   标注可视化
│
├── data/
│   ├── samples/                # 10 个电路文字样本
│   ├── schemas/                # JSON Schema
│   ├── ocr_training/           # CRNN OCR 训练数据 (11,642 crops)
│   └── Netlistify/             # Netlistify 源码（参考）
│
├── runs/
│   ├── detect/
│   │   ├── circuit_real/weights/best.pt   # 元件检测 17类 mAP 98.5%
│   │   ├── circuit_text/weights/best.pt   # 文字检测 1类  mAP 99.4%
│   │   ├── circuit_port/weights/best.pt   # 接线点检测 1类 mAP 98.9%
│   │   └── cghd_56cls/weights/best.pt     # CGHD 56类（训练中）
│   └── ocr_crnn/best.pt                   # 手写OCR CRNN val_loss=0.084
│
├── tests/                     # 14 个单元测试
│
├── test2.jpg ~ test5.jpg      # 测试用机画电路图
├── C193_D2_P1.png             # 测试用手绘电路图
└── circuit_19/42/47/99/100    # HCD 手绘图（在E:\circuit_image\）
```

## 训练好的模型

| 用途 | 路径 | 类别数 | 准确率 |
|------|------|:--:|:--:|
| 手绘元件检测 | `runs/detect/circuit_real/weights/best.pt` | 17 | mAP 98.5% |
| 手写文字区域 | `runs/detect/circuit_text/weights/best.pt` | 1 | mAP 99.4% |
| 接线点检测 | `runs/detect/circuit_port/weights/best.pt` | 1 | mAP 98.9% |
| 手写OCR | `runs/ocr_crnn/best.pt` | 28 chars | val_loss 0.084 |
| CGHD 全类 | `runs/detect/cghd_56cls/weights/best.pt` | 56 | 训练中 |

## 数据集

| 数据集 | 位置 | 图片数 | 用途 |
|------|------|:--:|------|
| Digitize-HCD | `E:\circuit_image\` | 1,277 手绘 | 元件检测 + 文字标注 |
| CGHD (GTDB-HD) | `E:\circuit_image\cghd-zenodo-13\` | 2,837 | 56类训练 + 端口标注 + junction |
| Netlistify | `E:\circuit_image\Netlistify\` | 100,000 机画 | GT SPICE 网表 |
| Tesseract | `E:\Tesseract-OCR\` | - | 印刷体 OCR 引擎 |

## 评价方案对比

| 方案 | 输入 | 处理 | 致命错误检出率 |
|------|------|------|:--:|
| Phase 1 (B) | 文字描述 | LLM 直接评价 | 100% |
| Phase 2 (A) | 文字描述 | LLM→JSON→规则+LLM | 受 converter 影响 |
| Phase 3 | 文字描述 | LLM→SPICE→Ngspice→LLM | 仿真增强 |
| Phase 4 (D) | 文字描述 | LLM 生成 SPICE→解析→LLM | 与 Phase 1 持平 |
| Vision | 图像 | YOLO+CRNN→LLM | 取决于 OCR 准确率 |

## 外部依赖

| 软件 | 路径 | 用途 |
|------|------|------|
| Python 3.13 | `C:\Python313` | 运行环境 |
| Ngspice | `E:\ngspice\bin\ngspice.exe` | SPICE 仿真 |
| Tesseract 5.5 | `E:\Tesseract-OCR\tesseract.exe` | 印刷体 OCR |

## 已知局限

- 手绘图连线检测：基础 CV 方法不可靠，需专用检测模型（DETR）
- YOLO 元件检测：在非 HCD 风格的图上泛化有限（CGHD 训练后可改善）
- CRNN OCR：手写准确率 85-90%，`k`与`Ω`易混淆
- 端口检测：使用数据集端口相对坐标推算，非实际检测
