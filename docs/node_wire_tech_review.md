# 手绘电路图节点与连线检测技术 — 论文调研与改进方案

> 基于12篇论文（2019–2026）的系统调研，针对当前 CNA=15%/GA=12%/PC=32% 提出5项可操作改进。

---

## 一、当前系统接线流水线回顾

### 架构：YOLO检测 → 骨架引导接线 → Union-Find建图 → LLM评审

```
Step 1: YOLO(CGHD 61类) 检测元件 + junction + text
Step 2: CRNN OCR 文字识别 + 元件值匹配
Step 3: 端口坐标生成 (硬编码 PORT_POSITIONS + Sobel方向检测)
Step 4: 端口网格吸附 (_snap_ports_to_grid)
Step 5: Junction 处理 (snap + Union-Find合并)
Step 6: P2J — 端口→Junction (最近邻 + 骨架回退 + P2P fallback)
Step 7: JJ  — Junction→Junction (对齐检测 + 骨架验证 + NN过滤)
Step 8: 路由 (Manhattan避障)
Step 9: Union-Find 建图 → 连通组
Step 10: LLM 结构化评审
```

### 消融实验配置

| 配置 | use_skeleton | use_sobel | use_nn_filter | use_skel_jj | use_close_port | use_force_connect | use_los |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Ours (默认) | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Baseline | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| w/o_Sobel | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| w/_Skeleton | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| CCL | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

### 准确率（56张全量）

| 配置 | CNA | GA | PC |
|------|-----|-----|-----|
| Ours | 0.115 | 0.108 | 0.345 |
| w/_Skeleton | 0.150 | 0.122 | 0.317 |
| Baseline | 0.131 | 0.093 | 0.334 |
| CCL | ~0.35 | ~0.08 | ~0.20 |

---

## 二、12篇论文逐篇技术对比

### 元件检测 + 节点/连线技术一览

| # | 论文 | 年份 | 元件检测 | 连线检测方法 | 节点检测方法 | 连接匹配策略 | 网表准确率 |
|---|------|------|---------|-------------|-------------|-------------|-----------|
| 1 | **Peker et al.** (IEEE Access) | 2026 | YOLOv8L+v11L+v11X (97.5%晶体管, 99.2%接地) | 骨架化 + 轮廓分析 | **基于规则的轮廓与连接分析** (300电路100%准确率) | 掩码重叠优先 → 最近邻回退 + SPICE仿真二元验证 | 印刷93.3%, 手绘85.3% |
| 2 | **Reddy & Panicker** (SNCS) | 2022 | YOLOv5 (mAP 98.2%) | **Hough变换** → 按斜率分类水平/垂直 → 交点检测 | 膨胀 + 轮廓检测 + **K-means聚类** | 距离映射 (每终端连接最近节点) | 重建80% |
| 3 | **Bohara et al.** (IEEE Access) | 2024 | YOLOv8m (mAP 96.7%) | **Harris角点 + MLSD线段检测** (95.71%) | K-means聚类分组 | 图遍历关联文本与元件边界框 | **95.71%** |
| 4 | **Bayer et al.** (ICPRAM) | 2023 | **Mask R-CNN** (Detectron2) 实例分割+关键点 | 实例分割直接检测导线掩码 | 非导线多边形=节点, 导线多边形=边 | 几何关键点匹配 | 概念验证 |
| 5 | **Uzair et al.** (DICTA) | 2023 | Faster R-CNN (F1=94.24%) | 连通分量分析 | 连通分量分析 | 文本标签→元件(欧氏距离+单位识别) | 82% |
| 6 | **Wategaonkar et al.** (ICDCS) | 2024 | YOLOv8m (mAP 99.07%) | **连接点斜率角度判别** (水平=180°/360°±10%, 垂直=90°/270°±10%) | 连接点识别 + 线对齐 | 元件中心位于水平/垂直线之间 | — |
| 7 | **Putak et al.** (MIPRO) | 2023 | YOLOv5m (mAP 99.07%) | 连接点斜率角度判别 (同上) | 连接点 + 线对齐 | 元件中心线间判断 | — |
| 8 | **Mohan et al.** (AISP) | 2022 | **端点分析法**(线长比) + HOG+SVM | 骨架化 → 端点提取 | **移除元件区域 → 轮廓检测** | 端点匹配 (最近节点) | — |
| 9 | **Chen et al.** (arXiv) | 2026 | **Gemini 2.5 Pro** (多模态LLM) + YOLO辅助 | LLM直接理解拓扑 | LLM隐式理解 | LLM生成ngspice网表 + 仿真验证迭代 | **97.59%** |
| 10 | **Mathur & Achar** (NEWCAS) | 2024 | Faster R-CNN/YOLOv5 + ResNet-50方向(99.97%) | — (不涉及连线) | — | — | — |
| 11 | **Ahmed et al.** (Data in Brief) | 2025 | 数据集(17类, 18,602实例) | 2D高斯热图端口定位 | 线交叉点单独标注 | 端口热图 + 距离匹配 | — |
| 12 | **Roy et al.** (Scientific Reports) | 2025 | CBAM-DenseNet-121 (91.15%) | — (不涉及连线) | — | — | — |

### 关键技术分类统计

| 技术类别 | 使用论文数 | 代表方法 |
|---------|:---:|------|
| **深度学习元件检测** | 7 | YOLOv5/v8/v11, Faster R-CNN, Mask R-CNN |
| **传统CV做节点/连线** | 5 | 骨架化+轮廓, Hough变换, 连通分量 |
| **Hough变换检测线段** | 4 | HoughLinesP + 按斜率分类 |
| **K-means聚类节点** | 3 | 合并邻近交点消除手绘抖动 |
| **骨架化预处理** | 3 | Zhang-Suen细化 → 轮廓追踪 |
| **实例分割(Mask R-CNN)** | 2 | 像素级元件/导线分离 |
| **LLM多模态理解** | 1 | Gemini 2.5 Pro端到端 |

---

## 三、连线检测 — 四条技术路线

```
路线A: Hough/MLSD 直线检测
  边缘/角点 → Hough/MLSD → 斜率分类 → 线段合并
  ├── Reddy(2022): Hough + 斜率分类
  └── Bohara(2024): Harris + MLSD (比Hough高8.57%)

路线B: 骨架化 + 轮廓追踪
  二值化 → 骨架化(1px) → 轮廓检测 → 端点提取
  ├── Peker(2026): 完整预处理管线 → 轮廓分析, 节点100%
  └── Mohan(2022): 骨架化 → 端点 → 匹配

路线C: 连接点斜率判别
  YOLO检测连接点 → 计算两点斜率 → H/V判定 → 线对齐
  ├── Wategaonkar(2024): 水平=180°/360°±10%, 垂直=90°/270°±10%
  └── Putak(2023): 同思路

路线D: 实例分割直接检测导线 (仅Bayer 2023)
  Mask R-CNN → 元件掩码+导线掩码 → 关键点 → 图构建
```

## 四、节点检测 — 三类方法

```
方法1: 传统CV管线 (5篇)
  灰度化 → 去噪/CLAHE → 自适应阈值 → 形态学 → 骨架化 → 轮廓检测
  └── Peker: 确定性规则, 300电路100%准确率

方法2: 几何特征 + 聚类 (2篇)
  Hough线 → 按斜率分类 → 交点 → 膨胀填隙 → K-means聚类
  └── Reddy: 92%节点识别率; Bohara: 95.71%网表

方法3: 深度学习分割 (2篇)
  Mask R-CNN → 元件掩码+导线掩码 → 非导线=节点, 导线=边
  └── Bayer: 端到端图提取
```

---

## 五、5项可操作改进方案

### 改进1：端口坐标吸附到骨架端点

| 项目 | 内容 |
|------|------|
| **来源** | Peker(2026) 终端分割模型思路 + Reddy(2022) 终端识别方法 |
| **针对问题** | 硬编码`PORT_POSITIONS`在手绘电路里偏差10-20px，是PC=32%的主因 |
| **改动量** | ~30行 |
| **预期提升** | CNA +8~12% |
| **风险** | 低 |

**方法**：在`unified_pipeline.py`的端口生成后（Step 5c之前），对每个端口坐标在骨架图像上搜索最近的骨架端点，用骨架端点替换硬编码位置：

```
对每个元件的每个端口(px, py):
  在骨架图像上, 以(px, py)为中心, 半径15px搜索骨架像素
  如果找到骨架端点(仅1个邻居):
    将端口坐标修正为骨架端点坐标
  如果找到骨架内部点(≥2个邻居):
    沿骨架向外追踪到最近的端点
    将端口坐标修正为该端点
```

**插入位置**：`unified_pipeline.py` Step 5c (`_snap_ports_to_grid`) 之前。

---

### 改进2：P2J骨架优先匹配

| 项目 | 内容 |
|------|------|
| **来源** | Peker(2026) "掩码重叠优先 → 最近邻回退" |
| **针对问题** | 纯最近邻P2J在密集电路里产生大量FP |
| **改动量** | ~15行 |
| **预期提升** | FP -15~20% |
| **风险** | 低 |

**方法**：在Step 6的P2J循环中，对每对(端口, junction)先检查骨架连通性。如果骨架连通，直接锁定该连接，不再参与最近邻竞争：

```python
# 在 Step 6 的 for jx, jy in junctions: 循环内, d < best_d 判断之前插入:
if skeleton is not None and config["use_skeleton"]:
    if _verify_skeleton_path(skeleton, px, py, jx, jy,
                             margin=3, min_ratio=0.50):
        best_j = (jx, jy)
        break  # 骨架连通 → 高置信度, 不再比较其他junction
```

**插入位置**：`unified_pipeline.py` Step 6 的 P2J 匹配循环内（约第1678行）。

---

### 改进3：MLSD线段检测替代/补充骨架验证

| 项目 | 内容 |
|------|------|
| **来源** | Bohara(2024) 消融实验: Hough→MLSD 从87.14%→95.71%(+8.57%) |
| **针对问题** | Zhang-Suen骨架在手绘不规则线条上退化成碎片，JJ骨架验证拒绝率过高 |
| **改动量** | ~50行 |
| **预期提升** | JJ准确率 +10% |
| **风险** | 中（需调MLSD参数） |

**方法**：在`_extract_skeleton()`之外新增`_extract_mlsd()`函数，在`_verify_skeleton_path`失败时回退检查MLSD线段：

```python
def _extract_mlsd(gray_img):
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines, _, _, _ = lsd.detect(gray_img)
    if lines is None:
        return []
    return [tuple(map(int, [l[0][0], l[0][1], l[0][2], l[0][3]]))
            for l in lines]

def _verify_mlsd_path(mlsd_segments, x1, y1, x2, y2, margin=8):
    """检查两junction间是否存在MLSD线段覆盖"""
    for sx1, sy1, sx2, sy2 in mlsd_segments:
        # 检查线段是否近似连接两个junction
        d11 = math.hypot(x1-sx1, y1-sy1)
        d12 = math.hypot(x1-sx2, y1-sy2)
        d21 = math.hypot(x2-sx1, y2-sy1)
        d22 = math.hypot(x2-sx2, y2-sy2)
        if min(d11, d12) < margin and min(d21, d22) < margin:
            return True
    return False
```

**插入位置**：`_verify_skeleton_path` 调用处增加 fallback 逻辑。

---

### 改进4：K-means局部聚类替代全局网格吸附

| 项目 | 内容 |
|------|------|
| **来源** | Reddy(2022) + Bohara(2024) 都用K-means处理手绘抖动 |
| **针对问题** | 全局网格吸附把不同元件的端口硬拉到同一行列，破坏局部结构 |
| **改动量** | ~40行 |
| **预期提升** | PC +5~8% |
| **风险** | 低 |

**方法**：在`_snap_ports_to_grid`之后，对同一连通组内的端口做K-means聚类：

```
对每个连通组内的所有端口坐标:
  if 组内端口数 <= 2: 跳过 (不需要聚类)
  用K-means (k=实际应该合并的簇数) 聚类
  每个簇内的端口 → 用聚类中心替换
```

**插入位置**：`unified_pipeline.py` Step 5c 之后、Step 6 之前，或嵌入`_snap_ports_to_grid`的 Step 3 逻辑中。

---

### 改进5：SPICE仿真验证闭环

| 项目 | 内容 |
|------|------|
| **来源** | Peker(2026) SPICE二元验证 + Chen(2026) ngspice迭代修正 |
| **针对问题** | 无法判断接线结果是否正确，LLM在错误信息上做评价 |
| **改动量** | ~200行 |
| **预期提升** | 论文核心亮点（差分点） |
| **风险** | 高（需Ngspice/PySpice集成，网表语法生成） |

**方法**：对接PySpice/Ngspice，让生成的网表在SPICE里仿真：

```
接线结果 → 生成SPICE网表 → Ngspice仿真
  ├── 仿真成功 → 接线大概率正确 → LLM评价更可信
  └── 仿真崩溃 → 接线有错 → 反馈给LLM修正/标记不可信
```

这个改进的论文价值大于工程价值——让你的系统与Reddy/Peker形成差异化："他们停在图像→网表，我们到LLM评价+仿真验证"。

---

## 六、改进优先级与实施路线

```
第一轮 (1-2天): 改进1 + 改进2
  ├── 端口吸附骨架端点 (CNA +8~12%)
  ├── P2J骨架优先匹配 (FP -15~20%)
  └── 跑消融 → 预期CNA从15%→22-25%

第二轮 (2-3天): 改进3 + 改进4
  ├── MLSD线段检测 (JJ +10%)
  ├── K-means聚类 (PC +5~8%)
  └── 跑消融 → 预期CNA从25%→30-35%

第三轮 (视投稿deadline): 改进5
  └── SPICE验证闭环 → 论文revision或补充实验
```

### 预期叠加效果

| 阶段 | CNA | GA | PC |
|------|-----|-----|-----|
| 当前 (Ours) | 0.115 | 0.108 | 0.345 |
| +改进1+2 | ~0.22-0.25 | ~0.18-0.20 | ~0.40-0.45 |
| +改进3+4 | ~0.30-0.35 | ~0.25-0.30 | ~0.45-0.50 |

---

## 七、论文定位建议

### 与已有工作的差异化

| | Reddy(2022) | Uzair(2023) | Peker(2026) | Bohara(2024) | **本系统** |
|---|---|---|---|---|---|
| 元件检测 | YOLOv5 | Faster R-CNN | YOLOv8+v11 | YOLOv8m | YOLO(CGHD 61类) |
| 连线方法 | Hough+K-means | CCL | 骨架+轮廓 | Harris+MLSD | 骨架引导P2J+JJ |
| 节点方法 | Hough交点+聚类 | 连通分量 | 基于规则 | K-means | Union-Find建图 |
| 网表准确率 | 80% | 82% | 85.3% | 95.71% | — |
| LLM评价 | ✗ | ✗ | ✗ | ✗ | **✓** |
| 仿真验证 | ✗ | ✗ | ✓(SPICE) | ✓(PySpice) | (规划中) |

### 创新点陈述

1. **骨架引导的多阶段接线算法**：P2J(端口到结点) + JJ(结点到结点) + LOS(视线连接) + Close-port + Force-connect 的8步接线流水线
2. **接线结果驱动LLM电路评审**：不是纯网表提取系统，而是从接线走向LLM评价
3. **多LLM并行评审框架**：合规性检查 + DFMEA风险分析 + 经验性原理评估的三维结构化评审

### 当前最大风险

> **接线准确率不够高 (CNA=15%)，LLM在错误信息上做判断，LLM vs 人工专家 Spearman 相关系数可能做不出显著结果。**

**缓解策略**：先做一轮LLM纠错能力实验（用当前最好的w/_Skeleton配置输出30张图的接线结果给LLM，看LLM能否指出接线错误）。如果LLM展现出纠错能力，这就是论文的意外发现——"即使接线不完美，LLM仍能给出有价值的评审"。

---

## 八、相关论文完整列表

| # | 论文 | 作者 | 发表 | 年份 |
|---|------|------|------|------|
| 1 | A Fully Automated SPICE-Compatible Netlist Extraction From Image Using Deep Learning and Image Preprocessing Techniques | Peker et al. | IEEE Access | 2026 |
| 2 | Hand-Drawn Electrical Circuit Recognition Using Object Detection and Node Recognition | Reddy & Panicker | SN Computer Science (Springer) | 2022 |
| 3 | Deep Learning-Based Framework for Power Converter Circuit Identification and Analysis | Bohara & Krishnamoorthy | IEEE Access | 2024 |
| 4 | Instance Segmentation Based Graph Extraction for Handwritten Circuit Diagram Images | Bayer et al. | ICPRAM | 2023 |
| 5 | Automated Netlist Generation from Offline Hand-Drawn Circuit Diagrams | Uzair et al. | DICTA | 2023 |
| 6 | Circuit Vision: Fast and Efficient Digitization of Complex Hand-Drawn Circuit using YOLOv8 | Wategaonkar et al. | ICDCS | 2024 |
| 7 | Electrical Scheme Digitization Using Deep Learning Methods | Putak et al. | MIPRO | 2023 |
| 8 | Generation of Netlist from a Hand-drawn Circuit through Image Processing and Machine Learning | Mohan et al. | AISP | 2022 |
| 9 | Enhancing Large Language Models for End-to-End Circuit Analysis Problem Solving | Chen et al. | arXiv | 2026 |
| 10 | Recognition of Electronic Component Orientations from Hand-Drawn Circuit Schematics through a Two-Stage Machine Learning System | Mathur & Achar | NEWCAS | 2024 |
| 11 | Digitize-HCD: A Dataset for Digitization of Handwritten Circuit Diagrams | Ahmed et al. | Data in Brief | 2025 |
| 12 | JUHCCR-v1: A Database for Hand-Drawn Electrical and Electronics Circuit Component Recognition | Roy et al. | Scientific Reports | 2025 |
