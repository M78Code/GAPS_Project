# 闪烁体波形位置推定 - 项目完整指导

**项目名**：闪烁体检出器波形数据的深度学习位置推定  
**来源**：大場同学卒业论文引继  
**开始日期**：2026-04-25  
**当前阶段**：数据处理  

---

## 📋 项目概览

### 任务目标
- **输入**：闪烁体检出器4通道波形数据（2048维）
- **输出**：粒子通过位置（回归，15～140cm范围）
- **目标**：用深度学习超越卒论的传统方法精度

### 数据规模
- **数据量**：15个run文件，总 ~900MB
- **位置覆盖**：15cm～140cm（15个位置点）
- **事件总数**：~5000+个粒子事件
- **关键特征**：CH0（左端）+ CH1（右端）波形信号

---

## 🚀 项目步骤（共14步）

### **Phase 1: 数据处理** ✓ 进行中

#### **Step 1：实现.dat文件读取函数** ✅ 已完成
**文件**：`Scintillator_Project/src/data_parse/dat_file_reader.py`

- ✅ 解析文本格式.dat文件
- ✅ 提取EVENT ID（从`=== EVENT N ===`）
- ✅ 读取4通道波形数据（各1024样本）
- ✅ 从文件名提取位置标签（run7_65.dat → 65cm）
- ✅ 返回所有events的列表

**输出格式**：
```python
[
  {'event_id': 1, 'CH0': [...], 'CH1': [...], 'CH2': [...], 'CH3': [...]},
  {'event_id': 2, 'CH0': [...], 'CH1': [...], 'CH2': [...], 'CH3': [...]},
  ...
]
```

#### **Step 2：批量读取和验证** ✅ 已完成
**文件**：`Scintillator_Project/src/data_parse/batch_process.py`

- ✅ 遍历`dataset/raw/`目录所有15个.dat文件
- ✅ 调用DatFileReader处理每个文件
- ✅ 保存为JSON格式：`dataset/processed/runN_XX.json`
- ✅ 显示处理进度和结果

**输出格式**（JSON）：
```json
{
  "filename": "run7_65.dat",
  "position_label": 65.0,
  "position_unit": "cm",
  "num_events": 236,
  "samples_per_channel": 1024,
  "events": [
    {
      "event_id": 1,
      "CH0": [0.000694, 0.001869, ...],
      "CH1": [0.000437, 0.000808, ...],
      "CH2": [...],
      "CH3": [...]
    },
    ...
  ]
}
```

#### **Step 3：数据验证 & 统计** 👈 当前
**文件**：`Scintillator_Project/src/data_parse/verify_data.py`

- ⏳ 遍历所有JSON文件验证格式
- ⏳ 统计：总事件数、位置范围、样本数
- ⏳ 输出数据统计报告
- ⏳ 检验数据完整性

**预期输出**：
```
位置覆盖范围: 15cm - 140cm
总事件数: 5000+
平均每个位置: 333个事件
最小/最大: xxx/yyy
```

---

### **Phase 2: 数据预处理**

#### **Step 4：信号脉冲切割**（可选但推荐）
- 检测信号起点和终点
- 从1024样本中提取有效脉冲部分
- 测试是否改善精度

#### **Step 5：波形归一化**
- 幅度归一化：`(waveform - mean) / std`
- 位置标签归一化或保持原样
- 保存预处理后的数据

#### **Step 6：数据分割**
- train: 70%
- val: 15%
- test: 15%
- **重要**：按照位置分层分割，确保每个位置都有train/val/test

---

### **Phase 3: 数据集构建** ⏳

#### **Step 7：创建PyTorch Dataset类**
```python
class ScintillatorDataset(torch.utils.data.Dataset):
    def __init__(self, json_files, train=True):
        # 加载JSON数据
        # 返回 (waveform, position_label)
        pass
    
    def __getitem__(self, idx):
        return waveform, label  # [2, 1024], scalar
```

#### **Step 8：创建DataLoader**
- batch_size: 32或64
- shuffle: True（train）
- num_workers: 2-4

---

### **Phase 4: 模型设计与训练** 

#### **Step 9：设计DL模型**
**架构选项**：
1. **1D CNN**（推荐）
   - Input: [batch, 2, 1024]
   - Conv1d → MaxPool → Dense
   - Output: [batch, 1]


 Step 9 要写的是 src/models/cnn1d.py，结构如下：

  输入 [batch, 2, 1024]
      ↓
  Conv1d 卷积层（×3）  ← 提取波形特征
      ↓
  AdaptiveAvgPool     ← 压缩时间轴
      ↓
  全连接层 FC（×2）    ← 回归到位置
      ↓
  输出 [batch, 1]     ← 预测位置（cm）

  ---
  有几个问题需要你决定，然后我给出代码：

  Q1：输出要不要做归一化？
  - 方案A：直接输出原始位置（15~140 cm），损失函数用 MSELoss
  - 方案B：输出归一化到 [0,1]，训练时再还原

  建议选 A，更直观。

  Q2：模型复杂度
  - 轻量版（3层Conv，适合快速验证）
  - 标准版（4~5层Conv + Dropout，防过拟合）

  建议先选轻量版，跑通后再调。

2. **LSTM/GRU**
   - 适合时间序列
   - 可捕捉波形中的时间依赖

3. **Transformer**
   - 最新方法
   - 需要更多数据

#### **Step 10：实现训练循环**
- 损失函数：MSE（回归）或MAE
- 优化器：Adam
- 学习率：0.001 或自适应
- 早停机制（Early Stopping）

#### **Step 11：训练和验证**
- 记录train/val loss
- 绘制学习曲线
- 保存最优模型
- 监控过拟合

---

### **Phase 5: 性能评估** ⏳ 第二周末

#### **Step 12：测试集评估**
- 计算位置推定误差：MAE, RMSE, R²
- 统计：平均误差、最大误差、误差分布
- 按位置分析误差（某些位置更容易或更难）

#### **Step 13：可视化结果**
- 真实 vs 预测位置散点图
- 误差 vs 位置分布
- 误差直方图
- 与卒论结果对标

#### **Step 14：撰写技术报告**
- 数据描述
- 模型架构说明
- 超参数设置
- 结果展示
- 与卒论对标
- 局限性分析

---

## 📁 项目目录结构

```
GAPS_Project/
├── Scintillator_Project/
│   ├── dataset/
│   │   ├── raw/                 # 原始.dat文件（15个）
│   │   │   ├── run7_65.dat
│   │   │   ├── run8_55.dat
│   │   │   └── ...
│   │   ├── processed/           # 处理后的JSON文件
│   │   │   ├── run7_65.json
│   │   │   └── ...
│   │   └── split/               # 分割后的train/val/test
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   ├── src/
│   │   ├── data_parse/
│   │   │   ├── dat_file_reader.py      # Step 1
│   │   │   ├── batch_process.py        # Step 2
│   │   │   └── verify_data.py          # Step 3
│   │   ├── preprocessing/
│   │   │   ├── pulse_cutter.py         # Step 4
│   │   │   ├── normalizer.py           # Step 5
│   │   │   └── data_splitter.py        # Step 6
│   │   ├── dataset/
│   │   │   └── scintillator_dataset.py # Step 7-8
│   │   ├── models/
│   │   │   ├── cnn_model.py           # Step 9
│   │   │   └── lstm_model.py
│   │   ├── train.py                    # Step 10-11
│   │   ├── evaluate.py                 # Step 12
│   │   └── visualize.py                # Step 13
│   ├── notebooks/
│   │   └── analysis.ipynb
│   ├── results/
│   │   ├── models/
│   │   ├── plots/
│   │   └── report.md                   # Step 14
│   └── logs/
│
└── docs/
    └── scintillator_project_guide.md   # 本文件
```

---

## ⏱️ 时间估计

| 阶段 | 步骤 | 时间 | 状态 |
|------|------|------|------|
| Phase 1 | Step 1-3 | 2-3天 | ✅ 进行中 |
| Phase 2 | Step 4-6 | 3-4天 | ⏳ 等待 |
| Phase 3 | Step 7-8 | 1-2天 | ⏳ 等待 |
| Phase 4 | Step 9-11 | 5-7天 | ⏳ 等待 |
| Phase 5 | Step 12-14 | 3-4天 | ⏳ 等待 |
| **总计** | - | **2-3周** | - |

---

## 🎯 关键里程碑

- **4月30日**：Step 1-3完成，数据统计报告 ✅ 进行中
- **5月7日**：Step 4-8完成，Dataset和DataLoader准备好
- **5月14日**：Step 9-11完成，模型训练完成
- **5月21日**：Step 12-14完成，技术报告提交

---

## 📊 数据验证清单

- [ ] Step 3：运行verify_data.py，确认：
  - [ ] 所有15个JSON文件已生成
  - [ ] 总事件数 > 5000
  - [ ] 位置范围 15-140cm 完整
  - [ ] 每个位置都有样本
  - [ ] CH0, CH1, CH2, CH3都有1024个样本

---

## 🆘 常见问题

### Q1: .dat文件读取失败
**A**: 检查文件格式是否为文本。用`file run7_65.dat`查看。

### Q2: JSON生成的数据量太大
**A**: 可选：保存为NPZ格式（更紧凑），或只保存CH0+CH1。

### Q3: 事件数不足以训练
**A**: 如果<2000，考虑数据增强（旋转、缩放波形）。

---

## 📝 下一步行动

```
现在：Step 3 数据验证
↓
运行: python verify_data.py
↓
查看统计结果
↓
确认无异常后
↓
开始 Step 4-6（下周）
```

**现在就运行**：
```bash
cd /Users/lind/Desktop/ppt/GAPS_Project/Scintillator_Project/src/data_parse
python verify_data.py
```

---

**最后更新**：2026-04-25
