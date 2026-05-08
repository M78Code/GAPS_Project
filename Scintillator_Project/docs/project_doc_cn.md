# 闪烁体波形位置推定项目文档

**更新日期**：2026年5月6日  
**负责人**：李（引继自大場章徳同学卒业论文）

---

## 1. 项目概述

### 1.1 背景

本项目继承自大場章徳同学（2022级，学号202204429）的卒业研究。大場同学已实现基于传统信号处理方法（CFD法、THR法、TLE法、电荷比法）的粒子位置推定，并建立了完整的实验数据集。

本项目的目标是：**用深度学习（1D CNN）方法直接从波形数据推定粒子位置，超越传统方法的精度。**

### 1.2 物理背景

闪烁体探测器（Scintillator Detector）两端各安装一个光电倍增管（PMT），分别对应 CH0（左端）和 CH1（右端）。当粒子穿过闪烁体时，光子同时向两端传播，到达时间差（Δt）和幅度比（R₀）携带粒子位置信息。

```
  [CH0 PMT] ←───────── 闪烁体 (125 cm) ─────────→ [CH1 PMT]
      │                                                  │
      │←── 位置越靠左，CH0信号越强、到达越早 ──────────────│
      │←── 位置越靠右，CH1信号越强、到达越早 ──────────────│
```

**Δt 的物理单调性**（已通过数据验证）：

| 位置 | 平均 Δt (index) | 说明 |
|------|----------------|------|
| 15 cm | +27.7 | CH1晚到，靠近左端 |
| 75 cm | ≈ 0 | 中间，两端等距 |
| 140 cm | −30.4 | CH1早到，靠近右端 |

---

## 2. 数据集

### 2.1 原始数据

| 属性 | 说明 |
|------|------|
| 格式 | `.dat` 二进制文件 |
| 来源 | `dataset/raw/` 和 `dataset/tar_zip/A/` |
| 总大小 | 约 900 MB |
| 总事件数 | 23,602 events |
| 文件数 | 15个（run7~run21） |

**文件命名规则**：`runNN_XX.dat`，其中 `XX` 为粒子通过位置（cm），直接作为标签。

### 2.2 各位置数据量

| 位置（cm） | 文件名 | 事件数（test集） | 备注 |
|-----------|--------|----------------|------|
| 15 | run9_15.dat | 904 | 主力样本 |
| 25 | run10_25.dat | 38 | 样本极少 |
| 35 | run11_35.dat | 39 | 样本极少 |
| 45 | run12_45.dat | 677 | 主力样本 |
| 55 | run8_55.dat | 46 | 样本偏少 |
| 65 | run7_65.dat | 34 | 样本极少 |
| 75 | run13_75.dat | 49 | 样本偏少 |
| 80 | run20_80.dat | 575 | 主力样本 |
| 85 | run14_85.dat | 44 | 样本偏少 |
| 95 | run19_95.dat | 81 | 样本偏少 |
| 105 | run18_105.dat | 59 | 样本偏少 |
| 115 | run17_115.dat | 71 | 样本偏少 |
| 125 | run16_125.dat | 722 | 主力样本 |
| 135 | run15_135.dat | 37 | 样本极少（最大瓶颈）|
| 140 | run21_140.dat | 179 | 样本偏少 |

**注意**：样本量严重不均衡，15/45/80/125cm 的样本占总量主体，135cm 仅 37 个事件是最大难点。

### 2.3 数据划分

| 集合 | 事件数 | 比例 |
|------|--------|------|
| 训练集（train） | 16,514 | 70% |
| 验证集（val） | 3,533 | 15% |
| 测试集（test） | 3,555 | 15% |

划分文件位于 `dataset/split/`，格式为 JSON。

### 2.4 波形数据结构

每个 event 包含 4 个通道，每通道 1024 个采样点：
- **CH0**：左端 PMT 信号（有效）
- **CH1**：右端 PMT 信号（有效）
- **CH2 / CH3**：触发信号（忽略）

脉冲特性：
- 脉冲方向为**负方向**（光电倍增管输出特性）
- 峰值集中在 index **550~650**（全1024点的第55%~63%处）
- 脉冲宽度约 **74个采样点**
- **[450~800] 固定窗口覆盖率：99.97%**（全数据集仅 6 个事件超出）

---

## 3. 预处理流程

### 3.1 `.dat` 文件解析

**脚本**：`src/preprocessor.py`，`src/data_parse/dat_file_reader.py`

解析流程：
1. 读取二进制 `.dat` 文件
2. 跳过 Header（Baseline、Amplitude、Charge、LeadingEdgeTime）
3. 提取 CH0、CH1 波形（各 1024 点）
4. 从文件名提取位置标签（`XX`）
5. 输出为 JSON 格式，保存至 `dataset/processed/`

### 3.2 数据集划分

**脚本**：`src/data_parse/split_dataset.py`

按 70/15/15 随机划分，保存为 `dataset/split/train.json`、`val.json`、`test.json`。

### 3.3 Dataset 类（ScintillatorDataset）

**文件**：`src/data_parse/scintillator_dataset.py`

```python
ScintillatorDataset(json_path, normalize=True, roi=(450, 800))
```

处理步骤（按顺序）：
1. 读取 JSON，提取所有 events
2. 将 CH0、CH1 合并为 shape `[2, 1024]` 的 NumPy 数组
3. **ROI 截取**（如 `roi=(450,800)`）：`waveform = waveform[:, 450:800]` → shape `[2, 350]`
4. **联合归一化**：`max_abs = np.max(np.abs(waveform))`，两通道共用同一最大值归一化

**为什么用联合归一化（而非独立归一化）？**

独立归一化会将每个通道分别除以自己的最大值，破坏两通道间的幅度比例关系。  
电荷比 R₀ = ln(Q_R / Q_L) 是重要的位置信息（携带 R₀ 信息）。联合归一化保留了 CH0/CH1 的相对幅度，使模型能够学习到这一物理特征。

### 3.4 DataLoader

**文件**：`src/data_parse/data_loader.py`

```python
make_dataloaders(split_dir, batch_size=64, num_workers=0)
```

| 参数 | train | val/test | 原因 |
|------|-------|----------|------|
| shuffle | True | False | 训练打乱防止顺序偏差，验证/测试需要复现 |
| batch_size | 64 | 64 | 显存与精度的平衡 |
| num_workers | 0 | 0 | Mac 上 0 最安全 |

---

## 4. 模型架构

### 4.1 CNN1D

**文件**：`src/models/cnn1d.py`

轻量级 1D CNN，专为波形回归设计。

```
输入: [batch, 2, 350]  (2通道 × ROI 350点)
  │
  ├─ Conv1d(2→32, k=7) + BN + ReLU + MaxPool(2)   → [batch, 32, 175]
  ├─ Conv1d(32→64, k=5) + BN + ReLU + MaxPool(2)  → [batch, 64, 87]
  ├─ Conv1d(64→128, k=3) + BN + ReLU + MaxPool(2) → [batch, 128, 43]
  │
  ├─ AdaptiveAvgPool1d(1)  → [batch, 128, 1]  (与输入长度无关)
  ├─ Squeeze              → [batch, 128]
  │
  ├─ Linear(128→64) + ReLU
  └─ Linear(64→1)
  
输出: [batch, 1]  (预测位置，单位 cm)
```

**参数量**：44,257（约 44K，非常轻量）

**关键设计**：`AdaptiveAvgPool1d(1)` 使模型对输入长度无感知，ROI 窗口大小可灵活调整而无需修改模型结构。

---

## 5. 训练配置

**脚本**：`src/scripts/train.py`

### 当前配置（300 epoch 版本）

| 参数 | 值 |
|------|----|
| EPOCHS | 300 |
| BATCH_SIZE | 64 |
| LEARNING_RATE | 3e-4 |
| 优化器 | Adam |
| 损失函数 | MSELoss（单位 cm²）|
| 学习率调度 | StepLR(step_size=75, gamma=0.7) |
| 设备 | MPS（Apple Silicon）|

**学习率衰减路径**：

| Epoch 区间 | 学习率 |
|-----------|--------|
| 1~75 | 0.000300 |
| 76~150 | 0.000210 |
| 151~225 | 0.000147 |
| 226~300 | 0.000103 |
| 300后第1个 epoch | 0.000072 |

**模型保存策略**：每次 val_loss 创新低时保存，文件名格式 `{timestamp}_best_model.pth`。

---

## 6. 评估

**脚本**：`src/scripts/evaluate.py`

评估流程：
1. 自动检测 `results/` 下时间戳最新的模型（正则过滤，防止加载错误文件）
2. 对测试集推理，计算 MAE、RMSE、R²
3. 按位置分别统计 RMSE
4. 保存结果至 `results/{timestamp}_test_results.txt`

**评估指标**：
- **MAE**：平均绝对误差（对大误差不敏感）
- **RMSE**：均方根误差（对大误差更敏感，与大場论文对比标准一致）
- **R²**：决定系数（越接近 1 越好）

---

## 7. 传统方法基准

来源：大場章徳卒业论文 PDF（`清水研_202204429_大場章徳_卒業研究概要書.pdf`）表1

| 手法 | 单独 σ | 融合后 σ | 说明 |
|------|--------|---------|------|
| 電荷比法 | 11.6 cm | — | ln(Q_R/Q_L) → 位置 |
| THR法 | 4.99 cm | 4.95 cm | 固定阈值触发时刻差 |
| CFD法 | 5.30 cm | **4.93 cm** | 恒分数鉴别，最优 |
| TLE法 | 5.55 cm | 5.13 cm | 前沿时刻 |

**CFD 融合法（4.93 cm）是深度学习需要超越的基准目标。**

### 传统方法说明

**CFD法（Constant Fraction Discriminator）**：
恒分数鉴别器，触发时刻定义为信号达到峰值某一固定比例时的时刻（如 20%），消除了幅度变化对时间测量的影响，是最精确的时刻触发方法。

**THR法（Threshold）**：
固定阈值法，信号超过预设电压阈值时触发，受信号幅度影响较大。

**TLE法（Trailing Leading Edge）**：
前沿时刻法，取信号上升沿（或下降沿）某个特定点定义触发时刻。

**电荷比法（Charge Ratio）**：
计算 R₀ = ln(Q_R / Q_L)（两通道积分电荷之比的对数），通过标定曲线推算位置。单独使用精度较差，但与时刻差法融合后可改善。

**残差加权融合（大場核心贡献）**：
将多种方法的位置估算结果按加权残差方式融合，各方法取长补短。CFD+电荷比融合后达到 4.93 cm。

---

## 8. 实验结果演进

### 8.1 训练轮次与 RMSE 对比

| 阶段 | 改进内容 | Test RMSE | 相比上一阶段 |
|------|---------|-----------|-------------|
| 基线 | 原始CNN，全1024点，独立归一化 | 5.52 cm | — |
| +ROI截取 | 固定窗口 [450~800]，输入350点 | 5.26 cm | ↓ 0.26 cm |
| +联合归一化 | 保留CH0/CH1幅度比（R₀信息） | 5.03 cm | ↓ 0.23 cm |
| 200 epoch | step_size=60, gamma=0.7 | **4.87 cm** | ↓ 0.16 cm ✅超越传统 |
| 300 epoch | step_size=75, gamma=0.7 | **4.78 cm** | ↓ 0.09 cm |

**传统最优（CFD融合）：4.93 cm**  
**当前深度学习最优：4.78 cm（领先 0.15 cm，提升 3.0%）**

### 8.2 300 epoch 各位置误差（最新结果）

| 位置 | RMSE | N（测试集） | 评价 |
|------|------|-----------|------|
| 15 cm | 4.84 cm | 904 | 良好 |
| 25 cm | 8.15 cm | 38 | 样本不足，误差偏大 |
| 35 cm | 7.78 cm | 39 | 样本不足，误差偏大 |
| 45 cm | 3.67 cm | 677 | 最优 |
| 55 cm | 5.47 cm | 46 | 中等 |
| 65 cm | 5.71 cm | 34 | 样本不足 |
| 75 cm | 6.99 cm | 49 | 样本偏少 |
| 80 cm | 4.12 cm | 575 | 良好 |
| 85 cm | 6.05 cm | 44 | 中等 |
| 95 cm | 6.13 cm | 81 | 中等 |
| 105 cm | 5.65 cm | 59 | 中等 |
| 115 cm | 5.36 cm | 71 | 中等 |
| 125 cm | 4.49 cm | 722 | 良好 |
| **135 cm** | **10.94 cm** | **37** | **最大瓶颈** |
| 140 cm | 4.57 cm | 179 | 良好 |

---

## 9. 问题分析与根因

### 9.1 CNN 初始性能差的根因

1. **输入噪声过多**：全 1024 点中约 66% 是纯噪声（基线区域），干扰特征提取
2. **独立归一化破坏物理信息**：各通道独立归一化后，CH0/CH1 幅度比（R₀）信息被抹除
3. **样本不均衡**：135cm（N=37）等位置样本极少，模型未能充分学习

### 9.2 改进措施效果

| 问题 | 解决方案 | 效果 |
|------|---------|------|
| 输入噪声 | ROI固定窗口[450~800] | RMSE 5.52→5.26 |
| R₀信息丢失 | 联合归一化（两通道共用max_abs） | RMSE 5.26→5.03 |
| 学习不充分 | 增加 epoch（200/300） | RMSE 5.03→4.87→4.78 |
| 样本不均衡 | **待实施**：数据增强 | 预期改善135cm |

---

## 10. 目录结构

```
Scintillator_Project/
├── dataset/
│   ├── raw/                        # 原始 .dat 文件（15个）
│   ├── processed/                  # 解析后的 JSON（单通道）
│   │   └── processed_ch0_and_ch1_to_json/  # 双通道 JSON
│   ├── split/                      # train/val/test.json
│   ├── DATA_OVERVIEW.md            # 数据集说明
│   └── TODO_from_obana.md          # 大場同学的建议事项
│
├── src/
│   ├── preprocessor.py             # .dat → JSON 预处理主脚本
│   ├── data_parse/
│   │   ├── dat_file_reader.py      # 二进制文件解析器
│   │   ├── split_dataset.py        # 数据集划分
│   │   ├── scintillator_dataset.py # PyTorch Dataset（含ROI+归一化）
│   │   └── data_loader.py          # DataLoader 工厂函数
│   ├── models/
│   │   └── cnn1d.py                # CNN1D 模型定义
│   └── scripts/
│       ├── train.py                # 训练脚本
│       ├── evaluate.py             # 测试集评估脚本
│       ├── evaluate_traditional.py # 传统方法对比（CFD/THR/TLE等）
│       ├── check_waveforms.py      # 波形统计分析（ROI窗口验证）
│       └── visualize.py            # 可视化脚本
│
├── results/
│   ├── {timestamp}_best_model.pth  # 各次训练最优模型
│   ├── {timestamp}_test_results.txt # 测试结果
│   └── plots/                       # 可视化图像
│
├── logs/
│   └── {timestamp}.log             # 训练日志
│
├── runs/                           # TensorBoard 日志
└── docs/                           # 文档（本文件）
```

---

## 11. 运行说明

### 环境要求

- Python 3.12
- PyTorch（支持 MPS / CUDA / CPU）
- NumPy、TensorBoard

### 训练

```bash
cd /path/to/parent_of_Scintillator_Project
python -m Scintillator_Project.src.scripts.train
```

训练日志保存至 `logs/{timestamp}.log`，最优模型保存至 `results/{timestamp}_best_model.pth`。

### 评估

```bash
python -m Scintillator_Project.src.scripts.evaluate
```

自动加载 `results/` 下时间戳最新的模型，结果保存至 `results/{timestamp}_test_results.txt`。

### 波形分析

```bash
python -m Scintillator_Project.src.scripts.check_waveforms
```

输出峰值位置统计、Δt 单调性验证、ROI 窗口覆盖率。

---

## 12. 下一步计划

### Step 18：数据增强（优先级最高）

**目标**：改善 135cm（N=37）、25cm（N=38）、35cm（N=39）等样本稀疏位置的误差

**方案 A**：对稀疏位置的波形加入高斯噪声，生成新样本
```python
# 生成增强样本
noise = np.random.normal(0, 0.01, waveform.shape)  # σ=1% 幅度
aug_waveform = waveform + noise
```

**方案 B**：在 DataLoader 中对稀疏位置过采样（oversample）
```python
# 按样本量逆比例设置采样权重
weights = 1.0 / position_counts[labels]
sampler = WeightedRandomSampler(weights, num_samples=len(dataset))
```

**预期效果**：135cm 位置 RMSE 从 10.94 cm 降至 7 cm 以下；整体 RMSE 有望进一步降至 4.5 cm 以下。

---

## 附录：关键实验记录

| 日期 | 实验内容 | val RMSE | test RMSE | 备注 |
|------|---------|---------|----------|------|
| 2026-05-04 | CNN 基线（100ep, 全1024点） | — | 5.52 cm | 初始结果 |
| 2026-05-06 | +ROI[450~800] | — | 5.26 cm | Step15完成 |
| 2026-05-06 | +联合归一化（100ep） | 5.38 cm | 5.03 cm | Step16/17完成 |
| 2026-05-06 | 200 epoch（step=60） | — | **4.87 cm** | 超越传统最优 |
| 2026-05-06 | 300 epoch（step=75） | 5.19 cm | **4.78 cm** | 当前最优 |
